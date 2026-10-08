import pytest
import sys
import numpy as np
import torch
sys.path.append("./")
import dre
from itertools import product


devices = ["cpu"] + [f"cuda{n}" for n in range(torch.cuda.device_count())]
tokens = [["Varius", "people",  "told", "me"]]
regexes = ["various"]


@pytest.mark.parametrize(("tokens", "regex", "device"), product(tokens, regexes, devices))
def test_observer(tokens, regex, device):
    embedder = dre.Glove50Embedder("./test/data/glove_10samples_50d.txt", device=device)
    word_parser = dre.WordParser()
    observer = dre.SequenceObserver(embedder=embedder, regex_str=regex, regex_parser=word_parser, device=device)
    print("TOKEN:",observer._observe_token(tokens[0]))
    print("TOKENS:",observer.observe_sequence(tokens))
    probs, locations = observer.match_transitions(tokens)
    print("PROBS:",repr(probs))


