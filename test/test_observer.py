import pytest
import sys
import numpy as np
import torch
sys.path.append("./")
import dre
from itertools import product

print_and_plot_debug = False


devices = ["cpu"] + [f"cuda{n}" for n in range(torch.cuda.device_count())]
tokens = [["Varius", "people",  "told", "me"]]
regexes = [("various|the", 3), ("great", 2)]
@pytest.mark.parametrize(("tokens", "regex_sz", "device"), product(tokens, regexes, devices))
def test_observer(tokens, regex_sz, device):
    regex, sz = regex_sz
    embedder = dre.Glove50Embedder("./test/data/glove_10samples_50d.txt", device=device)
    word_parser = dre.WordParser()
    observer = dre.SequenceObserver(embedder=embedder, regex_str=regex,
                                    regex_parser=word_parser, device=device)
    assert observer.observation_size == sz
    assert dre.AbstractObserver.observe_sequence(observer, tokens).size(1) == len(tokens)
    if print_and_plot_debug:
        print("TOKEN:", observer._observe_token(tokens[0]))
        print("TOKENS:", observer.observe_sequence(tokens))
    probs, locations = observer.match_transitions(tokens)
    if print_and_plot_debug:
        print("PROBS:", repr(probs))
