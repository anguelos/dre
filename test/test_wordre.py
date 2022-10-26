import pytest
import sys
sys.path.append("./")
import dre


def create_or_segment():
    left_symbol = dre.NFASegment.create_transition_segment("left","left")
    right_symbol = dre.NFASegment.create_transition_segment("right","right")
    result = dre.NFASegment.create_or_segment(left_segment=left_symbol, right_segment=right_symbol).label_states()
    return result.reduce_states().label_states()


@pytest.mark.parametrize(("creator", "regex"), [(create_or_segment,"left|right")])
def test_wordre(creator, regex):
    parser = dre.WordParser()
    print(parser.parse_str(regex).nfa_segment)
    print(str(creator()))
    assert str(parser.parse_str(regex).nfa_segment) == str(creator())