import pytest
import sys
import numpy as np 
sys.path.append("./")
from dre import FAGraph, NFASegment

def create_transition_graph():
    return NFASegment.create_transition_segment("symbol1","caption")

def create_concatenation_graph():
    tr1 = NFASegment.create_transition_segment("symbol1","caption1")
    tr2 = NFASegment.create_transition_segment("symbol2","caption2")
    concat = NFASegment.create_concatenate_segment(tr1,tr2)
    return concat.label_states()


def create_asterisk_graph():
    symbol_segment1 = NFASegment.create_transition_segment("symbol1","caption")
    symbol_segment2 = NFASegment.create_transition_segment("symbol2","caption")
    symbol_segment3 = NFASegment.create_transition_segment("symbol3","caption")
    asterisk = NFASegment.create_asterisk_segment(symbol_segment2)
    seq12 = NFASegment.create_concatenate_segment(symbol_segment1, asterisk)
    seq123 = NFASegment.create_concatenate_segment(seq12, symbol_segment3)
    return seq123.label_states()

def create_question_graph():
    symbol_segment1 = NFASegment.create_transition_segment("symbol1","caption")
    symbol_segment2 = NFASegment.create_transition_segment("symbol2","caption")
    symbol_segment3 = NFASegment.create_transition_segment("symbol3","caption")
    question = NFASegment.create_question_segment(symbol_segment2)
    seq12 = NFASegment.create_concatenate_segment(symbol_segment1, question)
    seq123 = NFASegment.create_concatenate_segment(seq12, symbol_segment3)
    return seq123.label_states()

def create_plus_graph():
    symbol_segment1 = NFASegment.create_transition_segment("symbol1","caption")
    symbol_segment2 = NFASegment.create_transition_segment("symbol2","caption")
    symbol_segment3 = NFASegment.create_transition_segment("symbol3","caption")
    plus = NFASegment.create_plus_segment(symbol_segment2)
    seq12 = NFASegment.create_concatenate_segment(symbol_segment1, plus)
    seq123 = NFASegment.create_concatenate_segment(seq12, symbol_segment3)
    return seq123.label_states()


def test_empty_segment():
    segment = NFASegment.create_transition_segment("","")
    assert not bool(segment.sink._incoming[0])
    assert segment.sink._incoming[0].source is segment.source
    assert segment.source._outgoing[0].destination is segment.sink

# TODO (anguelos) double check cardinalities
all_creators_sizes_reducedsizes = [(create_transition_graph, (1,2), (1,2)), 
                                   (create_concatenation_graph, (5,6), (3,4)), 
                                   (create_asterisk_graph, (12,12), (8,8)),
                                   (create_question_graph, (12,12), (6,6)), 
                                   (create_plus_graph, (15,14), (7,7))]

@pytest.mark.parametrize(("creator", "tr_st_sz"), [t[:2] for t in all_creators_sizes_reducedsizes])
def test_clone(creator, tr_st_sz, _):
    segment = creator()
    num_transitions, num_states = tr_st_sz
    transitions, states = segment.get_all_states_and_transitions()
    assert len(transitions) == num_transitions
    assert len(states) == num_states

    cloned_segment = segment.reduce_states()
    cloned_transitions, cloned_states = cloned_segment.get_all_states_and_transitions()

    assert len(cloned_transitions) == num_transitions
    assert len(cloned_states) == num_states


@pytest.mark.parametrize(("creator","tr_st_sz", "tr_st_reduced_sz"), all_creators_sizes_reducedsizes)
def test_reduce_size(creator, tr_st_sz, tr_st_reduced_sz):
    segment = creator()
    num_transitions, num_states = tr_st_sz
    reduced_num_transitions, reduced_num_states = tr_st_reduced_sz
    transitions, states = segment.get_all_states_and_transitions()
    assert len(transitions) == num_transitions
    assert len(states) == num_states

    reduced_segment = segment.reduce_states()
    transitions, states = reduced_segment.get_all_states_and_transitions()

    assert len(transitions) == reduced_num_transitions
    assert len(states) == reduced_num_states


all_creators = [t[0] for t in all_creators_sizes_reducedsizes]
@pytest.mark.parametrize("segment_creator", all_creators)
def test_clone(segment_creator):
    original_s = segment_creator()
    copy_s = original_s.clone()
    orig_tr, orig_st = original_s.get_all_states_and_transitions()
    copy_tr, copy_st = copy_s.get_all_states_and_transitions()
    orig_tr = [str(tr) for tr in orig_tr]
    orig_st = [str(st) for st in orig_st]
    copy_tr = [str(tr) for tr in copy_tr]
    copy_st = [str(st) for st in copy_st]
    assert sorted(orig_tr) == sorted(copy_tr)
    assert sorted(orig_st) == sorted(copy_st)