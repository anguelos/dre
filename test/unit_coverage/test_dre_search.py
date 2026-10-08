import itertools
import random
import re

import pytest
import torch

from dre import AbstractMatcher, Corpus, DFAMatcher, ExactObserver, JointSoftmaxObserver, L1Distance, PHOCEmbedder, Searcher, Tokenizer, ViterbiMatcher, compile_regex
from phoc import PHOC


def tokens(text, tokenizer=None):
    tokenizer = Tokenizer() if tokenizer is None else tokenizer
    return [text[start:end] for start, end in tokenizer.tokenize(tokenizer.encode(text)).T.tolist()]


@pytest.mark.parametrize(("text", "expected"), [
    ("Hi, what's up?!", ["Hi", ",", "what", "'s", "up", "?!"]),
    ("Don't \"quote\" me...", ["Don", "'t", "quote", "me", "..."]),
    ("well-known 'tis dogs' bones", ["well", "-", "known", "'tis", "dogs", "'", "bones"]),
    ("Wörld  \t3.5\n", ["Wörld", "3", ".", "5"]),
    ("", []),
    ("   ", []),
    ("?", ["?"]),
])
def test_tokenizer(text, expected):
    assert tokens(text) == expected


def test_kept_separators_are_merged_unless_asked_not_to():
    assert tokens("wait...?") == ["wait", "...?"]
    assert tokens("wait...?", Tokenizer(merge_kept=False)) == ["wait", ".", ".", ".", "?"]


def test_tokens_do_not_cross_documents():
    corpus = Corpus(["the end", "start here", "", "x"], lowercase=False)
    assert [[corpus.vocabulary[i] for i in document.tolist()] for document in corpus.documents] == [["the", "end"], ["start", "here"], [], ["x"]]
    assert corpus.char_span(1, 0, 2) == (0, 10)
    assert corpus.char_span(3, 0, 1) == (0, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_tokenizer_on_gpu():
    tokenizer = Tokenizer()
    codes = tokenizer.encode("Hi, what's up?! Don't \"quote\" me... 3.5")
    assert torch.equal(tokenizer.tokenize(codes.cuda()).cpu(), tokenizer.tokenize(codes))


def test_corpus_lowercases_observed_tokens():
    corpus = Corpus(["The Cat", "the cat"])
    assert corpus.documents[0].tolist() == corpus.documents[1].tolist()
    assert corpus.vocabulary.tokens == ["cat", "the"]


def test_search_reports_character_spans():
    texts = ["The Federal  Register, vol. 3", "no match here", "federal tax register; federal register"]
    corpus = Corpus(texts)
    searcher = Searcher(corpus, ExactObserver(corpus.vocabulary))
    hits = searcher.search("federal ( tax ) ? register")
    assert [(h.document, texts[h.document][h.start:h.end], h.probability) for h in hits] == [
        (0, "Federal  Register", 1.0), (2, "federal tax register", 1.0), (2, "federal register", 1.0)]
    assert searcher.search("federal ( tax ) ? register", top_n=1) == hits[:1]
    assert searcher.scores("federal register").tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        Searcher(corpus, ExactObserver(["federal"]))


def test_exact_hits_are_non_overlapping_matches_as_python_re_finds():
    rng = random.Random(0)
    texts = [" ".join(rng.choice("abc") for _ in range(rng.randint(0, 12))) for _ in range(40)]
    corpus = Corpus(texts)
    searcher = Searcher(corpus, ExactObserver(corpus.vocabulary))
    for regex, python_regex in [("a b", "ab"), ("a . c", "a.c"), ("a b +", "ab+"), ("( a | b ) c ?", "(?:a|b)c?"), ("b c *", "bc*")]:
        pattern = re.compile(python_regex)
        hits = searcher.search(regex)
        for hit in hits:
            assert pattern.fullmatch(texts[hit.document][hit.start:hit.end].replace(" ", "")), (regex, hit)
        for document, text in enumerate(texts):
            spans = sorted((h.start, h.end) for h in hits if h.document == document)
            assert all(end <= next_start for (_, end), (next_start, _) in zip(spans, spans[1:]))
            assert bool(spans) == bool(pattern.search(text.replace(" ", ""))), (regex, text)


def test_windowed_matches_equal_whole_document_matches():
    generator = torch.Generator().manual_seed(1)
    documents = [torch.randint(0, 6, (int(n),), generator=generator) for n in torch.randint(0, 30, (40,), generator=generator)]
    for regex in ["a b", "a . c", "a ( b | c ) ?", "b c +"]:
        automaton = compile_regex(regex)
        emissions = torch.rand(6, len(automaton.symbols), generator=generator) * (torch.rand(6, 1, generator=generator) > 0.4)
        emissions[:, automaton.any_mask] = 1
        whole = AbstractMatcher().matches(automaton, emissions, documents, min_probability=1e-6)
        windowed = ViterbiMatcher(min_emission=1e-6, max_segments=5, max_tokens=40).matches(automaton, emissions, documents, min_probability=1e-6)
        as_dict = lambda m: {(d, s, e): p for d, s, e, p in zip(*[c.tolist() for c in m])}
        whole, windowed = as_dict(whole), as_dict(windowed)
        assert whole.keys() == windowed.keys(), regex
        assert all(abs(whole[k] - windowed[k]) < 1e-6 for k in whole), regex
        best = AbstractMatcher().scores(automaton, emissions, documents)
        for (document, _, _), probability in whole.items():
            assert probability <= best[document] + 1e-6


def test_soft_search_finds_misspellings():
    texts = ["the federa1 regisrer of rules", "Federal Regster notices", "a register of federal rules"]
    corpus = Corpus(texts)
    searcher = Searcher(corpus, JointSoftmaxObserver(corpus.vocabulary, PHOCEmbedder(PHOC()), L1Distance()))
    hits = searcher.search("federal register")
    assert [(h.document, texts[h.document][h.start:h.end]) for h in hits] == [(0, "federa1 regisrer"), (1, "Federal Regster")]
    assert hits[0].probability > 0.9


DFA_REGEXES = ["a b", "a . c", "a b ?", "a b +", "( a | b ) * c", "b c *", "a ( b | . ) a"]


def random_documents(seed, vocabulary_size=5, n=40, longest=15):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocabulary_size, (int(k),), generator=generator) for k in torch.randint(0, longest, (n,), generator=generator)]


def categorical_emissions(automaton, vocabulary_size, generator):
    # random readings: each token is one query word or none, with probabilities summing to 1; ANY always holds
    words = automaton.word_mask
    readings = torch.softmax(3 * torch.randn(vocabulary_size, int(words.sum()) + 1, generator=generator), dim=1)
    emissions = torch.zeros(vocabulary_size, len(automaton.symbols))
    emissions[:, words] = readings[:, 1:]
    emissions[:, automaton.any_mask] = 1
    return emissions


@pytest.mark.parametrize("regex", DFA_REGEXES)
def test_dfa_equals_viterbi_on_exact_emissions(regex):
    automaton = compile_regex(regex)
    emissions = ExactObserver(["a", "b", "c", "x", "y"]).observe(automaton)
    documents = random_documents(0)
    as_set = lambda m: set(zip(*[c.tolist() for c in m]))
    for anchored in [False, True]:
        viterbi, dfa = ViterbiMatcher(anchored=anchored), DFAMatcher(anchored=anchored, max_segments=7, max_tokens=50)
        assert torch.equal(dfa.scores(automaton, emissions, documents), viterbi.scores(automaton, emissions, documents))
        assert as_set(dfa.matches(automaton, emissions, documents)) == as_set(viterbi.matches(automaton, emissions, documents))


@pytest.mark.parametrize("regex", DFA_REGEXES)
def test_anchored_dfa_is_the_total_probability_of_the_readings(regex):
    automaton = compile_regex(regex)
    generator = torch.Generator().manual_seed(2)
    emissions = categorical_emissions(automaton, 4, generator)
    documents = [d for d in random_documents(3, vocabulary_size=4, n=12, longest=5) if d.numel()]
    words = automaton.word_mask.nonzero().reshape(-1).tolist()
    readings = torch.cat([1 - emissions[:, words].sum(dim=1, keepdim=True), emissions[:, words]], dim=1)
    exact = ExactObserver([f"w{k}" for k in range(len(words) + 1)])  # reading r: "none" (w0) or word r
    symbols_of_reading = torch.zeros(len(words) + 1, len(automaton.symbols))
    symbols_of_reading[:, automaton.any_mask] = 1
    symbols_of_reading[torch.arange(1, len(words) + 1), words] = 1
    for document in documents:
        total = 0.0
        for sequence in itertools.product(range(len(words) + 1), repeat=document.numel()):
            matched = AbstractMatcher(anchored=True).scores(automaton, symbols_of_reading, [torch.tensor(sequence)])[0]
            total += float(matched) * float(torch.prod(readings[document, torch.tensor(sequence)]))
        assert abs(float(DFAMatcher(anchored=True).scores(automaton, emissions, [document])[0]) - total) < 1e-5, (regex, document)


@pytest.mark.parametrize("regex", DFA_REGEXES)
def test_dfa_match_probabilities_bound_the_best_path(regex):
    automaton = compile_regex(regex)
    emissions = categorical_emissions(automaton, 5, torch.Generator().manual_seed(4))
    documents = random_documents(5)
    ends = lambda m: {(d, e): p for d, _, e, p in zip(*[c.tolist() for c in m])}
    viterbi = ends(ViterbiMatcher(min_emission=1e-9).matches(automaton, emissions, documents, min_probability=1e-6))
    dfa = ends(DFAMatcher(min_emission=1e-9).matches(automaton, emissions, documents, min_probability=1e-6))
    assert all(dfa.get(end, 0.0) >= probability - 1e-6 for end, probability in viterbi.items())
    assert all(probability <= 1 + 1e-6 for probability in dfa.values())


def test_dfa_rejects_non_categorical_readings():
    automaton = compile_regex("a | b")
    with pytest.raises(ValueError):
        DFAMatcher().scores(automaton, torch.tensor([[0.8, 0.7]]), [torch.tensor([0])])


def test_searcher_with_dfa():
    texts = ["The Federal  Register, vol. 3", "no match here", "federal tax register; federal register"]
    corpus = Corpus(texts)
    observer = ExactObserver(corpus.vocabulary)
    assert Searcher(corpus, observer, DFAMatcher()).search("federal ( tax ) ? register") == Searcher(corpus, observer).search("federal ( tax ) ? register")


def symbol_letters(automaton):
    # one letter per single symbol, plus "none of them"; ANY is satisfied by every letter
    letters = torch.cat([torch.zeros(1, len(automaton.symbols), dtype=torch.bool), torch.eye(len(automaton.symbols), dtype=torch.bool)])
    letters[:, automaton.any_mask] = True
    return letters


def run_dfa(dfa, sequence):
    state, accepted = dfa.initial, []
    for letter in sequence:
        state = int(dfa.transitions[state, letter])
        accepted.append(bool(dfa.accept[state]))
    return accepted


def test_minimized_dfa_accepts_the_same_sequences():
    from test_dre_engine import random_regex
    rng = random.Random(6)
    for _ in range(150):
        automaton = compile_regex(random_regex(rng)[0])
        letters = symbol_letters(automaton)
        for search in [False, True]:
            dfa = automaton.determinize(letters, search=search)
            minimal = dfa.minimized()
            assert minimal.n_states <= dfa.n_states
            assert minimal.minimized().n_states == minimal.n_states
            for _ in range(10):
                sequence = [rng.randrange(letters.size(0)) for _ in range(rng.randint(1, 8))]
                assert run_dfa(minimal, sequence) == run_dfa(dfa, sequence)


def test_minimization_merges_equivalent_states():
    duplicated, single = compile_regex("a b | a b"), compile_regex("a b")
    assert duplicated.determinize(symbol_letters(duplicated)).minimized().n_states == single.determinize(symbol_letters(single)).minimized().n_states
    star = compile_regex("a * a *")
    assert star.determinize(symbol_letters(star)).minimized().n_states == 2  # accepting (only a's so far) and dead


def test_pretokenized_texts_split_on_whitespace_only():
    text = "don't  well-known \" 3.5 ?! café"
    assert tokens(text, Tokenizer.pretokenized()) == text.split()
