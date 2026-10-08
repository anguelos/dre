import pytest
import torch

from phoc.step01_interface import PHOC


class OnesPHOC(PHOC):
    def embed_codes(self, codes, offsets):
        return torch.ones(offsets.size(0) - 1, self.dim, device=codes.device)


def test_abstract():
    with pytest.raises(TypeError):
        PHOC()


def test_default_layout():
    phoc = OnesPHOC()
    assert phoc.levels == (1, 2, 3, 5)
    assert phoc.n_symbols == 36
    assert phoc.n_regions == 11
    assert phoc.dim == 11 * 36 + 20
    assert OnesPHOC(count_unknown=True).n_symbols == 37
    assert OnesPHOC(levels=[2], max_len=0).dim == 2 * 36


def test_encode_strings():
    phoc = OnesPHOC()
    codes, offsets = phoc.encode_strings(["Hello", "", "ab#"])
    assert offsets.tolist() == [0, 5, 5, 8]
    assert codes.dtype == torch.long and codes.size(0) == 8
    assert codes[7].item() == 0
    assert torch.equal(phoc.encode_strings(["AbC"])[0], phoc.encode_strings(["abc"])[0])
    empty_codes, empty_offsets = phoc.encode_strings([])
    assert empty_codes.numel() == 0 and empty_offsets.tolist() == [0]


def test_column_weights():
    phoc = OnesPHOC(levels=(1, 2), level_weights=(2.0, 3.0), max_len=4, len_weight=5.0)
    expected = torch.cat([torch.full((36,), 2.0), torch.full((72,), 3.0), torch.full((4,), 5.0)])
    assert torch.equal(phoc.column_weights, expected)


def test_forward_inputs():
    phoc = OnesPHOC(levels=(1, 2), level_weights=(2.0, 3.0), max_len=4, len_weight=5.0)
    single = phoc("hello")
    assert single.shape == (phoc.dim,)
    assert torch.equal(single, phoc.column_weights)
    batch = phoc(["hello", "world", ""])
    assert batch.shape == (3, phoc.dim)
    assert torch.equal(phoc(phoc.encode_strings(["hello", "world", ""])), batch)
    assert torch.equal(phoc(("hello", "world")), batch[:2])
    assert phoc([]).shape == (0, phoc.dim)
    with pytest.raises(TypeError):
        phoc(42)


@pytest.mark.parametrize("kwargs", [dict(occurrence="max"), dict(levels=[]), dict(levels=[0, 2]), dict(level_weights=[1.0]), dict(max_len=-1)])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        OnesPHOC(**kwargs)


def test_distance():
    a = torch.tensor([[0.0, 1.0], [2.0, 2.0]])
    b = torch.tensor([[1.0, 1.0]])
    assert torch.equal(PHOC.distance(a, b), torch.tensor([[1.0], [2.0]]))


def test_repr():
    assert "dim=416" in repr(OnesPHOC())
