import itertools
import random
import re

import pytest
import torch

from dre import (END_OF_DOCUMENT, AbstractMatcher, Corpus, DFAMatcher, ExactObserver, JointSoftmaxObserver, L1Distance, PHOCEmbedder, Tokenizer,
                 ViterbiMatcher, Vocabulary, compile_regex)


def stitch(emissions, documents):
    # Documents of ids into `emissions` -> one sequence (ids + 1: id 0 is END_OF_DOCUMENT, which follows each document)
    # and the emissions with a row for it; plus each document's first position.
    sequence = torch.cat([torch.cat([d + 1, torch.zeros(1, dtype=torch.long)]) for d in documents]) if documents else torch.zeros(0, dtype=torch.long)
    starts = torch.cumsum(torch.tensor([0] + [d.numel() + 1 for d in documents]), 0)
    return torch.cat([torch.zeros(1, emissions.size(1)), emissions]), sequence, starts


def document_scores(matcher, automaton, emissions, documents):
    # Each document's best match (anchored: the probability that the whole document matches), from one run.
    emissions, sequence, starts = stitch(emissions, documents)
    p = matcher.run(automaton, emissions, sequence, None).end_probability
    if matcher.anchored:
        return torch.tensor([float(p[starts[n + 1] - 2]) if d.numel() else 0.0 for n, d in enumerate(documents)])
    return torch.tensor([float(p[starts[n]:starts[n + 1]].max()) if d.numel() else 0.0 for n, d in enumerate(documents)])


def document_matches(matcher, automaton, emissions, documents, min_probability=0.01):
    # Every match with at least min_probability as (document, start, end (exclusive), probability) tensors, positions
    # within the document.
    emissions, sequence, starts = stitch(emissions, documents)
    matches = matcher.run(automaton, emissions, sequence, min_probability)
    ends = ((matches.end_probability >= min_probability) & (matches.end_probability > 0)).nonzero().reshape(-1)
    document = torch.searchsorted(starts, ends, right=True) - 1
    return document, matches.start[ends] - starts[document], ends + 1 - starts[document], matches.end_probability[ends]


def tokens(text, tokenizer=None):
    tokenizer = Tokenizer() if tokenizer is None else tokenizer
    return [text[start:end] for start, end in tokenizer.tokenize(text).T.tolist()]


def phoc_observer(vocabulary):
    return JointSoftmaxObserver(vocabulary, PHOCEmbedder(), L1Distance())


@pytest.mark.parametrize(("text", "expected"), [
    ("Hi, what's up?!", ["Hi", ",", "what", "'s", "up", "?!"]),
    ("Don't \"quote\" me...", ["Don", "'t", "quote", "me", "..."]),
    ("well-known 'tis dogs' bones", ["well", "-", "known", "'tis", "dogs", "'", "bones"]),
    ("Wörld  \t3.5\n", ["Wörld", "3", ".", "5"]),
    ("", []),
    ("   ", []),
    ("?", ["?"]),
    (f"end.{END_OF_DOCUMENT}{END_OF_DOCUMENT}'s", ["end", ".", END_OF_DOCUMENT, END_OF_DOCUMENT, "'s"]),
])
def test_tokenizer(text, expected):
    assert tokens(text) == expected


def test_kept_separators_are_merged_unless_asked_not_to():
    assert tokens("wait...?") == ["wait", "...?"]
    assert tokens("wait...?", Tokenizer(merge_kept=False)) == ["wait", ".", ".", ".", "?"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_tokenizer_on_gpu():
    text = "Hi, what's up?! Don't \"quote\" me... 3.5"
    assert torch.equal(Tokenizer().tokenize(text, device="cuda"), Tokenizer().tokenize(text))


def test_hashed_vocabulary_is_exact():
    text = "Wörld Würld the The the 'tis Wörld"
    spans = Tokenizer().tokenize(text)
    vocabulary, ids = Vocabulary.from_spans(text, spans)
    assert [vocabulary[i] for i in ids.tolist()] == [text[s:e] for s, e in spans.T.tolist()]
    assert vocabulary.tokens == [END_OF_DOCUMENT] + sorted({"Wörld", "Würld", "the", "The", "'tis"})


def test_corpus_stitches_documents_with_end_of_document():
    texts = ["the end", "start here", "", "x"]
    corpus = Corpus(texts)
    assert [corpus.vocabulary[i] for i in corpus.token_ids.tolist()] == ["the", "end", END_OF_DOCUMENT, "start", "here", END_OF_DOCUMENT, END_OF_DOCUMENT, "x", END_OF_DOCUMENT]
    assert len(corpus) == 4 and [corpus.document(n) for n in range(4)] == texts
    assert Corpus([f"a{END_OF_DOCUMENT}b"]).document(0) == "a b"  # the separator cannot occur inside a document


def test_tokens_are_not_normalized_observers_decide():
    corpus = Corpus(["The Cat", "the cat"])
    assert corpus.vocabulary.tokens == [END_OF_DOCUMENT, "Cat", "The", "cat", "the"]
    assert corpus.scores("the cat").tolist() == [1, 1]  # ExactObserver compares case-insensitively


def test_search_reports_character_spans():
    texts = ["The Federal  Register, vol. 3", "no match here", "federal tax register; federal register"]
    corpus = Corpus(texts)
    hits = corpus.search("federal ( tax ) ? register")
    assert [(h.document, texts[h.document][h.start:h.end], h.text, h.probability) for h in hits] == [
        (0, "Federal  Register", "Federal  Register", 1.0), (2, "federal tax register", "federal tax register", 1.0),
        (2, "federal register", "federal register", 1.0)]
    assert corpus.search("federal ( tax ) ? register", order="position") == sorted(hits, key=lambda h: (h.document, h.start))
    assert corpus.scores("federal register").tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        corpus.search("federal", order="alphabetical")


def test_a_match_is_credited_once():
    corpus = Corpus(["x a b b b y", "a b"])
    assert [(h.document, h.text) for h in corpus.search("a b +")] == [(0, "a b b b"), (1, "a b")]
    assert corpus.matches("a b +").end_probability.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]


def test_anchored_corpus_scores_whole_documents():
    corpus = Corpus(["what time is it", "it is what", "what", ""], matcher=ViterbiMatcher(anchored=True))
    assert corpus.scores("what . *").tolist() == [1, 0, 1, 0]
    assert [(h.document, h.text) for h in corpus.search("what . *")] == [(0, "what time is it"), (2, "what")]


def test_exact_hits_are_non_overlapping_matches_as_python_re_finds():
    rng = random.Random(0)
    texts = [" ".join(rng.choice("abc") for _ in range(rng.randint(0, 12))) for _ in range(40)]
    corpus = Corpus(texts)
    for regex, python_regex in [("a b", "ab"), ("a . c", "a.c"), ("a b +", "ab+"), ("( a | b ) c ?", "(?:a|b)c?"), ("b c *", "bc*")]:
        pattern = re.compile(python_regex)
        hits = corpus.search(regex)
        for hit in hits:
            assert pattern.fullmatch(hit.text.replace(" ", "")), (regex, hit)
        for document, text in enumerate(texts):
            spans = sorted((h.start, h.end) for h in hits if h.document == document)
            assert all(end <= next_start for (_, end), (next_start, _) in zip(spans, spans[1:]))
            assert bool(spans) == bool(pattern.search(text.replace(" ", ""))), (regex, text)


def test_soft_search_finds_misspellings():
    texts = ["the federa1 regisrer of rules", "Federal Regster notices", "a register of federal rules"]
    hits = Corpus(texts, phoc_observer).search("federal register")
    assert [(h.document, h.text) for h in hits] == [(0, "federa1 regisrer"), (1, "Federal Regster")]
    assert hits[0].probability > 0.9


def test_windowed_matches_equal_whole_document_matches():
    generator = torch.Generator().manual_seed(1)
    documents = [torch.randint(0, 6, (int(n),), generator=generator) for n in torch.randint(0, 30, (40,), generator=generator)]
    for regex in ["a b", "a . c", "a ( b | c ) ?", "b c +"]:
        automaton = compile_regex(regex)
        emissions = torch.rand(6, len(automaton.symbols), generator=generator) * (torch.rand(6, 1, generator=generator) > 0.4)
        emissions[:, automaton.any_mask] = 1
        whole = document_matches(AbstractMatcher(), automaton, emissions, documents, min_probability=1e-6)
        windowed = document_matches(ViterbiMatcher(min_emission=1e-6, max_segments=5, max_tokens=40), automaton, emissions, documents, min_probability=1e-6)
        as_dict = lambda m: {(d, s, e): p for d, s, e, p in zip(*[c.tolist() for c in m])}
        whole, windowed = as_dict(whole), as_dict(windowed)
        assert whole.keys() == windowed.keys(), regex
        assert all(abs(whole[k] - windowed[k]) < 1e-6 for k in whole), regex
        best = document_scores(AbstractMatcher(), automaton, emissions, documents)
        for (document, _, _), probability in whole.items():
            assert probability <= best[document] + 1e-6


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
    emissions = ExactObserver(["a", "b", "c", "x", "y"]).observe(automaton)[1:]  # row 0 is END_OF_DOCUMENT
    documents = random_documents(0)
    as_set = lambda m: set(zip(*[c.tolist() for c in m]))
    for anchored in [False, True]:
        viterbi, dfa = ViterbiMatcher(anchored=anchored), DFAMatcher(anchored=anchored, max_segments=7, max_tokens=50)
        assert torch.equal(document_scores(dfa, automaton, emissions, documents), document_scores(viterbi, automaton, emissions, documents))
        assert as_set(document_matches(dfa, automaton, emissions, documents)) == as_set(document_matches(viterbi, automaton, emissions, documents))


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
            matched = document_scores(AbstractMatcher(anchored=True), automaton, symbols_of_reading, [torch.tensor(sequence)])[0]
            total += float(matched) * float(torch.prod(readings[document, torch.tensor(sequence)]))
        assert abs(float(document_scores(DFAMatcher(anchored=True), automaton, emissions, [document])[0]) - total) < 1e-5, (regex, document)


@pytest.mark.parametrize("regex", DFA_REGEXES)
def test_dfa_match_probabilities_bound_the_best_path(regex):
    automaton = compile_regex(regex)
    emissions = categorical_emissions(automaton, 5, torch.Generator().manual_seed(4))
    documents = random_documents(5)
    ends = lambda m: {(d, e): p for d, _, e, p in zip(*[c.tolist() for c in m])}
    viterbi = ends(document_matches(ViterbiMatcher(min_emission=1e-9), automaton, emissions, documents, min_probability=1e-6))
    dfa = ends(document_matches(DFAMatcher(min_emission=1e-9), automaton, emissions, documents, min_probability=1e-6))
    assert all(dfa.get(end, 0.0) >= probability - 1e-6 for end, probability in viterbi.items())
    assert all(probability <= 1 + 1e-6 for probability in dfa.values())


def test_dfa_rejects_non_categorical_readings():
    automaton = compile_regex("a | b")
    with pytest.raises(ValueError):
        document_scores(DFAMatcher(), automaton, torch.tensor([[0.8, 0.7]]), [torch.tensor([0])])


def test_corpus_with_dfa():
    texts = ["The Federal  Register, vol. 3", "no match here", "federal tax register; federal register"]
    assert Corpus(texts, matcher=DFAMatcher()).search("federal ( tax ) ? register") == Corpus(texts).search("federal ( tax ) ? register")


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


def test_default_normalization():
    from dre.default_mapping import default_mapper
    from dre import AbstractEmbedder
    mapper, embedder = default_mapper(), AbstractEmbedder()
    assert mapper("Ærzbiſchoff Würzburg") == "arzbischoff wurzburg"   # MUFI letters -> lower-case ASCII
    assert mapper("Hello,\tWorld!") == "hello. world."              # punctuation -> ".", whitespace -> " "
    assert mapper("well-known 1290") == "well-known 1290"            # "-" and digits are kept
    assert embedder.normalize("Wörld") == "world"               # decomposed characters are composed first


def test_case_sensitivity_is_the_mappers_choice():
    import string
    from pylelemmatize.fast_mapper import LemmatizerBMP
    from dre import AbstractEmbedder
    assert compile_regex("Federal register").symbols == ["Federal", "register"]  # query words stay as written
    texts = ["the federal register", "the Federal register"]
    assert Corpus(texts).scores("Federal register").tolist() == [1, 1]          # default: case-insensitive
    keep_case = LemmatizerBMP(mapping_dict={c: c for c in string.ascii_letters + " "})
    case_sensitive = Corpus(texts, lambda vocabulary: ExactObserver(vocabulary, AbstractEmbedder(keep_case)))
    assert case_sensitive.scores("Federal register").tolist() == [0, 1]


def test_compile_time_normalization_merges_equal_words():
    from dre import AbstractEmbedder
    normalize = AbstractEmbedder().normalize
    assert compile_regex("Federal | federal").symbols == ["Federal", "federal"]
    merged = compile_regex("( Federal | federal ) . <number>", normalize)
    assert merged.symbols == [".", "<number>", "federal"]
    corpus = Corpus(["the FEDERAL rules", "federal rules"])
    assert corpus.scores("Federal | federal").tolist() == [1, 1]
    assert compile_regex("Federal | federal", corpus.observer.normalize).symbols == ["federal"]


def test_tables_keep_their_own_characters():
    from dre import TableEmbedder
    table = TableEmbedder(["US", "us", "u.s."], torch.eye(3))
    assert table["US"].tolist() == [1, 0, 0] and table["us"].tolist() == [0, 1, 0] and table["u.s."].tolist() == [0, 0, 1]
    assert table["Us"].tolist() == [0, 0, 0]  # not in the table: the null embedding
