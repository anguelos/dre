import random

import pytest
import torch

from dre import (ANY, AbstractMatcher, CharCodeEmbedder, Corpus, CosineDistance, ExactObserver, JointSoftmaxObserver, L1Distance,
                 LevenshteinDistance, PHOCEmbedder, SigmoidObserver, TableEmbedder, ViterbiMatcher, Vocabulary, compile_regex)
from dre.abstract_matcher import batches_by_size, candidate_segments, longest_match
from dre.default_mapping import default_mapper
from phoc import PHOC


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


def search(regex, texts):
    return Corpus(texts).scores(regex).tolist()


def phoc_observer(vocabulary, **kwargs):
    return JointSoftmaxObserver(vocabulary, PHOCEmbedder(), L1Distance(), **kwargs)


@pytest.mark.parametrize(("regex", "text", "expected"), [
    ("federal register", "the federal register vol", 1),
    ("federal register", "federal income register", 0),
    ("federal register", "federal", 0),
    ("federal . register", "federal income register", 1),
    ("federal . register", "federal register", 0),
    ("federal income ? register", "federal register", 1),
    ("federal income ? register", "federal income register", 1),
    ("federal income ? register", "federal tax register", 0),
    ("federal ( income | tax ) register", "the federal tax register", 1),
    ("federal ( income | tax ) register", "federal fee register", 0),
    ("a b +", "x a b b b", 1),
    ("a b +", "a", 0),
    ("a x * b", "a b", 1),
    ("a x * b", "a x x x b", 1),
    ("( a | b ) *  c", "b a b c", 1),
    ("Federal", "federal", 1),
])
def test_exact_search(regex, text, expected):
    assert search(regex, [text]) == [expected]


def test_compile_structure():
    automaton = compile_regex("federal ( income | . ) register")
    assert automaton.symbols == [ANY, "federal", "income", "register"]
    assert automaton.any_mask.tolist() == [True, False, False, False]
    assert automaton.transitions.shape == (4, automaton.n_states, automaton.n_states)
    assert automaton.start.sum() >= 1 and automaton.accept.sum() == 1


@pytest.mark.parametrize("regex", ["", "a |", "( a", "a )", "* a", "^ a", "a1"])
def test_invalid_regex(regex):
    with pytest.raises(ValueError):
        compile_regex(regex)


def test_soft_scores_are_best_path_products():
    automaton = compile_regex("a ( b | c )")
    emissions = torch.zeros(4, 3)  # vocabulary: x, y, z, w; symbols a, b, c
    emissions[0] = torch.tensor([0.9, 0.0, 0.0])
    emissions[1] = torch.tensor([0.0, 0.5, 0.2])
    emissions[2] = torch.tensor([0.0, 0.1, 0.8])
    documents = [torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([1, 0]), torch.tensor([3, 3])]
    for matcher in [AbstractMatcher(), ViterbiMatcher()]:
        assert torch.allclose(document_scores(matcher, automaton, emissions, documents), torch.tensor([0.45, 0.72, 0.0, 0.0]))


def test_phoc_emissions():
    automaton = compile_regex("register . spinners")
    vocabulary = ["register", "registers", "banana", "spinners", "sphinnas"]
    table = phoc_observer(vocabulary).observe(automaton)[1:]  # row 0 is END_OF_DOCUMENT
    register, spinners, any_ = automaton.symbols.index("register"), automaton.symbols.index("spinners"), automaton.symbols.index(ANY)
    assert torch.equal(table[:, any_], torch.ones(5))
    assert table[0, register] > 0.99 and table[3, spinners] > 0.99
    assert table[0, register] > table[1, register] > table[2, register]
    assert table[4, spinners] > table[2, spinners]
    assert torch.equal(phoc_observer(vocabulary).observe(compile_regex(". .")), torch.ones(6, 1))


def test_vocabulary_distances_expand_to_document_distances():
    document = "the federal register , the regster of federal rules".split()
    observer = phoc_observer(Vocabulary.from_documents([document]), batch_size=3)
    observer.prepare(["federal", "register"])
    distances = torch.stack([observer._word_columns[w] for w in ["federal", "register"]], dim=1)
    phoc = PHOC(mapper=default_mapper())  # what PHOCEmbedder builds
    token_ids = observer.vocabulary.encode(document)
    assert torch.allclose(distances[token_ids], PHOC.distance(phoc(document), phoc(["federal", "register"])))  # every running token
    assert torch.allclose(observer._masses[token_ids], phoc(document).abs().sum(dim=1))


def test_cache_serves_many_automata():
    vocabulary = ["register", "regster", "federal", "banana"]
    shared = phoc_observer(vocabulary)
    shared.prepare(["federal", "register", "rules"])
    for regex in ["federal register", "register | rules", "federal"]:
        automaton = compile_regex(regex)
        assert torch.allclose(shared.observe(automaton), phoc_observer(vocabulary).observe(automaton))
    assert sorted(shared._word_columns) == ["federal", "register", "rules"]


def test_observe_sequence_equals_vocabulary_rows():
    sentence = "the regster of federal rules".split()
    observer = phoc_observer(Vocabulary.from_documents([sentence]))
    automaton = compile_regex("federal . register")
    assert torch.allclose(observer.observe_sequence(automaton, sentence), observer.observe(automaton)[observer.vocabulary.encode(sentence)])


def test_phoc_emissions_are_a_joint_softmax():
    automaton = compile_regex("register | registers")
    table = phoc_observer(["register", "registers", "banana"]).observe(automaton)[1:]
    assert torch.all(table.sum(dim=1) <= 1 + 1e-6)
    register, registers = automaton.symbols.index("register"), automaton.symbols.index("registers")
    assert table[0, register] > table[0, registers] > 0.001  # a close second query word takes a share
    assert table[2].sum() < 0.01  # an unrelated token goes to the background


def test_sigmoid_observer_is_the_original_cosine_sigmoid():
    words = ["king", "queen", "apple"]
    vectors = torch.tensor([[1.0, 0.2], [0.9, 0.4], [-0.3, 1.0]])
    observer = SigmoidObserver(words + ["unknown"], TableEmbedder(words, vectors), CosineDistance(), b=10.)
    table = observer.observe(compile_regex("king"))[1:, 0]
    similarity = torch.cosine_similarity(vectors, vectors[0:1], dim=1)
    assert torch.allclose(table[:3], 1 / (1 + torch.exp(-10 * similarity + 5)), atol=1e-6)
    assert table[3] < 0.01  # no vector: cosine distance 1


def test_levenshtein_distance_matches_dynamic_programming():
    def reference(a, b):
        row = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            previous, row[0] = row[0], i
            for j, cb in enumerate(b, 1):
                previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (ca != cb))
        return row[-1]
    rng = random.Random(0)
    tokens = ["", "a", "ab", "kitten", "sitting", "Grüße"] + ["".join(rng.choice("abc") for _ in range(rng.randint(0, 9))) for _ in range(60)]
    words = tokens[:20]
    embedder = CharCodeEmbedder(max_length=12)
    distance = LevenshteinDistance(chunk_elements=500)
    distances = distance(embedder.embed(tokens), embedder.embed(words))
    assert distances.tolist() == [[float(reference(t, w)) for w in words] for t in tokens]
    assert distance.mass(embedder.embed(tokens)).tolist() == [float(len(t)) for t in tokens]


@pytest.mark.parametrize(("regex", "longest"), [
    ("a", 1), ("a b", 2), ("a . c", 3), ("a b ? c", 3), ("a ( b | c d ) e", 4),
    ("a b *", None), ("a +", None), ("( a | . )", None), (". .", None), ("a ? .", None),
])
def test_longest_match(regex, longest):
    assert longest_match(compile_regex(regex)) == longest


def test_segments_are_windows_around_resembling_tokens():
    automaton = compile_regex("a . c")  # matches span 3 tokens: keep 2 tokens around each resembling token
    vocabulary_size = 4  # 0 = a, 1 = c, 2 and 3 = other words
    emissions = ExactObserver(["a", "c", "x", "y"]).observe(automaton)[1:]  # row 0 is END_OF_DOCUMENT
    documents = [torch.tensor([2, 2, 2, 0, 3, 1, 2, 2, 2, 2, 2, 2]), torch.tensor([3, 3]), torch.tensor([1, 2])]
    emissions, sequence, _ = stitch(emissions, documents)  # ids + 1; END_OF_DOCUMENT (0) after each document
    segments, positions = candidate_segments(automaton, emissions, sequence, 1e-4)
    assert [(s - 1).tolist() for s in segments] == [[2, 2, 0, 3, 1, 2, 2], [1, 2]]  # windows stop at END_OF_DOCUMENT
    assert positions.tolist() == [1, 16]
    assert emissions.size(0) == vocabulary_size + 1
    assert candidate_segments(automaton, emissions, torch.tensor([], dtype=torch.long), 1e-4)[0] == []


def test_windowed_scores_equal_whole_document_scores():
    generator = torch.Generator().manual_seed(0)
    documents = [torch.randint(0, 6, (int(n),), generator=generator) for n in torch.randint(0, 40, (50,), generator=generator)]
    for regex in ["a b", "a . c", "a ( b | c ) ?", "a . ? c", "b c +", "( a | . ) c"]:
        automaton = compile_regex(regex)
        emissions = torch.rand(6, len(automaton.symbols), generator=generator) * (torch.rand(6, 1, generator=generator) > 0.5)
        emissions[:, automaton.any_mask] = 1
        whole = document_scores(AbstractMatcher(), automaton, emissions, documents)
        assert torch.allclose(document_scores(ViterbiMatcher(min_emission=0.0), automaton, emissions, documents), whole), regex
        windowed = document_scores(ViterbiMatcher(min_emission=1e-9, max_segments=7, max_tokens=60), automaton, emissions, documents)
        assert torch.allclose(windowed, whole), regex
        pruned = document_scores(ViterbiMatcher(min_emission=0.3), automaton, emissions, documents)
        exact = whole >= 0.3
        assert torch.allclose(pruned[exact], whole[exact]) and torch.all(pruned <= whole + 1e-6), regex
        for anchored in [ViterbiMatcher(anchored=True), AbstractMatcher(anchored=True)]:
            assert torch.allclose(document_scores(anchored, automaton, emissions, documents), document_scores(AbstractMatcher(anchored=True), automaton, emissions, documents))


def test_batches_by_size():
    assert batches_by_size([5, 1, 3, 100, 2], max_segments=2, max_tokens=10) == [[1, 4], [2, 0], [3]]


def test_token_classes_and_apostrophes():
    automaton = compile_regex("what 's <number> <punct> ?")
    assert automaton.symbols == ["'s", "<number>", "<punct>", "what"]
    classes = {"<number>": str.isdigit, "<punct>": lambda t: t in {"?", "."}}
    vocabulary = ["what", "'s", "42", "?", "x"]
    table = ExactObserver(vocabulary, classes=classes).observe(automaton)[1:]  # row 0 is END_OF_DOCUMENT
    assert table[:, automaton.symbols.index("<number>")].tolist() == [0, 0, 1, 0, 0]
    assert table[:, automaton.symbols.index("<punct>")].tolist() == [0, 0, 0, 1, 0]
    assert table[:, automaton.symbols.index("'s")].tolist() == [0, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        ExactObserver(vocabulary).observe(automaton)
    soft = phoc_observer(vocabulary, classes=classes).observe(automaton)[1:]
    assert torch.equal(soft[:, automaton.symbols.index("<number>")], table[:, automaton.symbols.index("<number>")])
    with pytest.raises(ValueError):
        phoc_observer(vocabulary).observe(automaton)


def test_anchored_matching_covers_the_whole_document():
    automaton = compile_regex("a b")
    emissions = ExactObserver(["a", "b", "x"]).observe(automaton)[1:]  # row 0 is END_OF_DOCUMENT
    documents = [torch.tensor([0, 1]), torch.tensor([2, 0, 1]), torch.tensor([0, 1, 2]), torch.tensor([], dtype=torch.long)]
    assert document_scores(ViterbiMatcher(), automaton, emissions, documents).tolist() == [1, 1, 1, 0]
    assert document_scores(ViterbiMatcher(anchored=True), automaton, emissions, documents).tolist() == [1, 0, 0, 0]
    soft = torch.tensor([[0.9, 0.0], [0.0, 0.8], [0.1, 0.1]])
    assert torch.allclose(document_scores(ViterbiMatcher(anchored=True), automaton, soft, [torch.tensor([0, 1]), torch.tensor([0, 1, 2])]), torch.tensor([0.72, 0.0]))


def test_documents_without_resemblance_are_skipped():
    automaton = compile_regex("a")
    emissions = torch.tensor([[1.0], [0.0]])
    scores = document_scores(ViterbiMatcher(), automaton, emissions, [torch.tensor([1, 1]), torch.tensor([1, 0]), torch.tensor([], dtype=torch.long)])
    assert scores.tolist() == [0.0, 1.0, 0.0]


def random_regex(rng, depth=0):
    # (word regex, the same pattern as a Python character regex, binding level); every word is a single letter. The word
    # regex has parentheses only where precedence requires them (postfix 2 > concatenation 1 > alternation 0), so
    # comparing with the fully grouped Python regex also tests the parser's precedence.
    def operand(sub, level):
        return sub[0] if sub[2] >= level else f"( {sub[0]} )"
    choice = rng.random() if depth < 3 else 0.0
    if choice < 0.4:
        letter = rng.choice("abc")
        return letter, letter, 3
    left = random_regex(rng, depth + 1)
    if choice < 0.6:
        operator = rng.choice("*?+")
        return f"{operand(left, 3)} {operator}", f"(?:{left[1]}){operator}", 2
    right = random_regex(rng, depth + 1)
    if choice < 0.85:
        return f"{operand(left, 1)} {operand(right, 1)}", f"(?:{left[1]})(?:{right[1]})", 1
    return f"{left[0]} | {right[0]}", f"(?:{left[1]})|(?:{right[1]})", 0

def test_search_agrees_with_python_re():
    import re
    rng = random.Random(0)
    for _ in range(300):
        regex, python_regex, _ = random_regex(rng)
        pattern = re.compile(python_regex)
        texts = ["".join(rng.choice("abc") for _ in range(rng.randint(1, 6))) for _ in range(8)]
        expected = [float(any(pattern.fullmatch(text, i, j) for i in range(len(text)) for j in range(i + 1, len(text) + 1)))
                    for text in texts]
        assert search(regex, [" ".join(text) for text in texts]) == expected, regex
