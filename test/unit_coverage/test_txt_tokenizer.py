import pytest
import torch

from dre.txt_tokenizer import Glove, clean_gutenberg_txt

GLOVE_PATH = "test/data/glove_10samples_50d.txt"


@pytest.fixture(scope="module")
def glove():
    return Glove(GLOVE_PATH, tokenizer="naive")


def test_clean_gutenberg_txt():
    txt = "header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nline one\nline two\n\n\nnext para\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nfooter"
    assert clean_gutenberg_txt(txt) == "line one line two\nnext para "


def test_load_glove():
    embedding_dict, embedding_list = Glove.load_glove(GLOVE_PATH)
    assert len(embedding_list) == 25
    assert embedding_dict["hello"].shape == (50,)


def test_glove_average_and_lookup(glove):
    all_embeddings = torch.stack(list(glove.embbeding_dict.values()))
    assert torch.allclose(glove.avg, all_embeddings.mean(dim=0))
    assert glove.embedding_length() == 50
    assert torch.equal(glove["hello"], glove.embbeding_dict["hello"])
    assert torch.equal(glove["notaword"], glove.avg)


def test_embed_sentence(glove):
    embedded = glove.embed_sentence("hello world.")
    assert embedded.shape == (50, 3)
    assert torch.equal(embedded[:, 0], glove["hello"])
    assert torch.equal(embedded[:, 2], glove["."])


def test_get_observation_probabillities(glove):
    row_wise = torch.stack([glove["hello"], glove["world"]])
    probs = glove.get_observation_probabillities("hello great world", row_wise)
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2, dtype=probs.dtype))
