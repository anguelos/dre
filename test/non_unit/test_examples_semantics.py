import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "semantics"))

from dre import compile_regex  # noqa: E402
from re2rnn_rules import CLASSES, aggregate, read_rules, to_dre  # noqa: E402


def test_read_rules():
    text = "[a]\n$ * x $ * // example\n// $ * commented\n\n[b+c]\n$ * y $ *\n"
    assert read_rules(text) == [("a", "$ * x $ *"), ("b+c", "$ * y $ *")]


@pytest.mark.parametrize(("rule", "expected"), [
    ("$ * dating & * $ * call $*", ". * dating <punct> * . * call . *"),
    ("$ * call % $*", ". * call <number> . *"),
    ("( which | what ) ${0,2} ( do | does )", "( which | what ) ( . ( . ) ? ) ? ( do | does )"),
    ("a{2} b ( c | d ){1,2}", "a a b ( c | d ) ( ( c | d ) ) ?"),
    ("$ * what 's $ EOS", ". * what 's . EOS"),
])
def test_to_dre(rule, expected):
    assert to_dre(rule) == expected
    compile_regex(expected)


def test_to_dre_rejects_unknown_syntax():
    with pytest.raises(ValueError):
        to_dre("( .* er )")


def test_classes():
    assert CLASSES["<number>"]("1996") and CLASSES["<number>"]("3.5") and not CLASSES["<number>"]("1.2.3")
    assert CLASSES["<punct>"]("?") and not CLASSES["<punct>"]("'s")


def test_aggregate_priorities_and_default():
    labels = ["ham", "spam"]
    # rule 0 is ham's catch-all, rule 1 a spam rule; spam dominates ham
    assert aggregate([[1, 0], [1, 1], [0, 0]], ["ham", "spam"], labels, "SMS") == [0, 1, 0]
