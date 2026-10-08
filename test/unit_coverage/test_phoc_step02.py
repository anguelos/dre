import pytest
import torch

from phoc import PHOC


def histogram(phoc, word):
    return phoc(word)[:phoc.n_regions * phoc.n_symbols].view(phoc.n_regions, phoc.n_symbols)


def column(phoc, character):
    return int(phoc.encoder.str_to_intlabel_seq(character)[0]) - 1


def test_counting_shares():
    phoc = PHOC(levels=[1, 2], max_len=0)
    h = histogram(phoc, "abc")
    assert h[:, column(phoc, "a")].tolist() == [1, 1, 0]
    assert h[:, column(phoc, "b")].tolist() == [1, 0.5, 0.5]
    assert h[:, column(phoc, "c")].tolist() == [1, 0, 1]
    assert histogram(phoc, "aabb")[:, column(phoc, "a")].tolist() == [2, 2, 0]


def test_thirds_are_exact():
    phoc = PHOC(levels=[3], max_len=0)
    h = histogram(phoc, "ab")
    assert torch.allclose(h[:, column(phoc, "a")], torch.tensor([2 / 3, 1 / 3, 0]))
    assert torch.allclose(histogram(phoc, "abcd")[:, column(phoc, "b")], torch.tensor([1 / 3, 2 / 3, 0]))


def test_presence():
    phoc = PHOC(levels=[1, 2], occurrence="presence", max_len=0)
    assert histogram(phoc, "abc")[:, column(phoc, "b")].tolist() == [1, 1, 1]
    assert histogram(phoc, "aabb")[:, column(phoc, "a")].tolist() == [1, 1, 0]


def test_unknown_characters_and_length():
    phoc = PHOC(levels=[1], max_len=4)
    v = phoc("a#b")
    assert v[:phoc.n_symbols].sum() == 2
    assert v[phoc.n_symbols:].tolist() == [1, 1, 1, 0]
    with_unknown = PHOC(levels=[1], max_len=0, count_unknown=True)
    assert with_unknown("a#b")[0] == 1
    assert torch.equal(phoc("abcdefg")[-4:], torch.ones(4))


def test_batch_matches_single_words():
    phoc = PHOC()
    words = ["spinners", "", "Bellevue", "a", "aeronautics"]
    batch = phoc(words)
    for n, word in enumerate(words):
        assert torch.allclose(batch[n], phoc(word))


def test_distance_orders_edits():
    phoc = PHOC()
    e = phoc(["register", "registers", "regulate", "banana"])
    d = PHOC.distance(e, e)
    assert d[0, 1] < d[0, 2] < d[0, 3]
    assert d[0, 0] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_cuda_matches_cpu():
    words = ["spinners", "bellevue", "of"]
    assert torch.allclose(PHOC()(words), PHOC().cuda()(words).cpu())
