import torch

from dre.default_mapping import MAPPING, default_mapper
from phoc import GloveToPHOC, PHOCToSemantic, PHOCToString


def test_glove_to_phoc_predicts_non_negative_phocs():
    model = GloveToPHOC(16, hidden=[32], mapping=MAPPING).eval()
    predicted = model(torch.randn(5, 16))
    assert predicted.shape == (5, model.phoc.dim) and bool((predicted >= 0).all())
    assert torch.equal(model.targets(["king", "queen"]), model.phoc(["king", "queen"]))
    assert GloveToPHOC(**model.config()).phoc.dim == model.phoc.dim


def test_models_accept_phocs_as_well_as_strings():
    words = ["king", "queen", "man"]
    semantic = PHOCToSemantic(16, hidden=[32], mapping=MAPPING).eval()
    decoder = PHOCToString(default_mapper().dst_alphabet_str, hidden=32, mapping=MAPPING).eval()
    phocs = semantic.phoc(words)
    assert torch.allclose(semantic(phocs), semantic(words))
    assert decoder.spell(phocs) == decoder.spell(words)
    targets = decoder.encode_targets(words)
    assert torch.allclose(decoder(phocs, targets), decoder(words, targets))


def test_gradients_flow_through_a_predicted_phoc():
    to_phoc, semantic = GloveToPHOC(16, hidden=[32], mapping=MAPPING), PHOCToSemantic(16, hidden=[32], mapping=MAPPING)
    vectors = torch.randn(4, 16)
    semantic(to_phoc(vectors)).sum().backward()
    assert all(p.grad is not None for p in to_phoc.parameters())
