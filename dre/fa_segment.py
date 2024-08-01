from collections import defaultdict
#from lib2to3.pgen2.pgen import NFAState
import numpy as np
from typing import List, Set, Tuple, Union


from .fa_state import FAState
from .fa_transition import FATransition


class NFASegment(object):
    def __init__(self, _=None, source=None, sink=None):
        assert source is not None
        assert sink is not None
        self.source=source
        self.sink=sink

    def is_dfa(self):
        transitions, states = self.get_all_states_and_transitions()
        non_e_transitions = all([bool(tr) for tr in transitions])
        no_multiple_outgoing = []
        for state in states:
            outgoing_symbols = [tr.symbol_str for tr in state._outgoing]
            no_multiple_outgoing.append(len(outgoing_symbols) == len(set(outgoing_symbols)))
        return non_e_transitions and all(no_multiple_outgoing)

    def reduce_states(self):
        tmp_segment = self.clone()
        _, states = tmp_segment.get_all_states_and_transitions()
        removable = [st for st in states if tmp_segment.can_remove(st)]
        i = 1
        while len(removable):
            tmp_segment = tmp_segment.clone_without(removable[0])
            _, states = tmp_segment.get_all_states_and_transitions()
            removable = [st for st in states if tmp_segment.can_remove(st)]
            i += 1
        return tmp_segment

    def get_all_states_and_transitions(self) -> Tuple[Tuple[FAState], Tuple[FAState]]:
        transitions, states = NFASegment._get_all_states_and_transitions(self.source, set(), set())
        return tuple(sorted(transitions)), tuple(sorted(states))

    @staticmethod
    def _get_all_states_and_transitions(state: FAState, visited_states: Set[FAState], visited_transitions: Set[FATransition]) -> Tuple[Set[FAState], Set[FAState]]:
        result_states = visited_states.union({state})
        result_transitions = visited_transitions.copy()
        for incoming in state.incoming_transitions:
            result_transitions.add(incoming)
            if incoming.source not in result_states:
                new_transitions, new_states = NFASegment._get_all_states_and_transitions(state=incoming.source,visited_states=result_states, visited_transitions=result_transitions)
                result_states.update(new_states)
                result_transitions.update(new_transitions)

        for outgoing in state.outgoing_transitions:
            result_transitions.add(outgoing)
            if outgoing.destination not in result_states:
                new_transitions, new_states = NFASegment._get_all_states_and_transitions(state=outgoing.destination,visited_states=result_states, visited_transitions=result_transitions)
                result_states.update(new_states)
                result_transitions.update(new_transitions)
        return result_transitions, result_states

    def clone(self):
        transitions, states = self.get_all_states_and_transitions()
        assert self.sink in states
        old2new_states = {s: s.clone_disconnected() for s in states}
        #old2new_transitions = {t: t.clone() for t in transitions}
        old2new_transitions = {}
        for t in transitions:
            old2new_transitions[t] = FATransition(source=old2new_states[t._source], destination=old2new_states[t._destination], symbol_str=t.symbol_str, caption=t.caption)
        new_source = old2new_states[self.source]  # source of the segment
        new_sink = old2new_states[self.sink]
        return NFASegment(source=new_source, sink=new_sink)

    def can_remove(self, s: FAState):
        if s is self.source or s is self.sink:
            return False
        if len(s._incoming) != 1 or len(s._outgoing) != 1:
            return False
        symbols = s._incoming[0].symbol_str, s._outgoing[0].symbol_str
        if "" not in symbols:  # A chain of symbols is not removable
            return False
        if len(s._incoming) != 1 or len(s._outgoing) != 1:
            return False
        return True

    def clone_without(self, rm_state):
        assert self.can_remove(rm_state)
        transitions, states = self.get_all_states_and_transitions()
        old2new_states = {s: s.clone_disconnected() for s in states if s is not rm_state}
        new_source = old2new_states[self.source]
        new_sink = old2new_states[self.sink]
        keep_transitions = [tr for tr in transitions if not (rm_state is tr._source) and not (rm_state is tr._destination)]

        new_before = old2new_states[rm_state._incoming[0]._source]
        new_after = old2new_states[rm_state._outgoing[0]._destination]
        new_symbols = rm_state._incoming[0].symbol_str, rm_state._outgoing[0].symbol_str

        old2new_transitions = {0: FATransition(source = new_before, destination=new_after, symbol_str=new_symbols[0]+new_symbols[1])}
        for t in keep_transitions:
            source=old2new_states[t._source]
            destination=old2new_states[t._destination]
            old2new_transitions[t] = FATransition(source = source, destination=destination, symbol_str=t.symbol_str,caption=t.caption)
        result = NFASegment(source=new_source, sink=new_sink)
        return result


    def __str__(self):
        transitions, states = self.get_all_states_and_transitions()
        states = [st[1] for st in sorted([(s.id,s) for s in states])]
        res = "\n".join(["%3d:%s"%(k,v) for k,v in enumerate(states)])
        res = "States:\n"+res+"\nTransitions:\n"
        for transition in transitions:
            res+=str(transition)+"\n"
        return res+"\n"

    @classmethod
    def create_transition_segment(cls,symbol_str,caption):
        source=FAState()
        sink=FAState()
        transition=FATransition(source=source, destination=sink, symbol_str=symbol_str,caption=caption)
        return cls(sink=sink, source=source)

    @classmethod
    def create_concatenate_segment(cls,left_segment,right_segment):
        source=FAState()
        sink = FAState()
        left_transition = FATransition(source=source, destination=left_segment.source)
        midle_transition = FATransition(source=left_segment.sink, destination=right_segment.source)
        right_transition = FATransition(source=right_segment.source, destination=sink)
        return cls(sink=sink,source=source)

    @classmethod
    def create_asterisk_segment(cls,input_segment):
        source = FAState()
        sink = FAState()
        source_transition = FATransition(source=source, destination=input_segment.source)
        destination_transition = FATransition(source=input_segment.sink,destination=sink)
        loop_transition = FATransition(source=input_segment.sink, destination=input_segment.source)
        return cls(sink=sink,source=source)

    @classmethod
    def create_question_segment(cls,input_segment):

        source=FAState()
        sink=FAState()
        source_transition=FATransition(source=source,destination=input_segment.source)
        destination_transition=FATransition(source=input_segment.sink, destination=sink)
        shortcut_transition=FATransition(source=source, destination=sink)
        return cls(sink=sink,source=source)

    @classmethod
    def create_plus_segment(cls,input_segment):
        input_segment_copy = input_segment.clone()
        asterisk = cls.create_asterisk_segment(input_segment_copy)
        return cls.create_concatenate_segment(asterisk,input_segment_copy)


    @classmethod
    def create_or_segment(cls, left_segment, right_segment):
        source = FAState()
        sink = FAState()
        left_source_transition=FATransition(source=source, destination=left_segment.source)
        left_destination_transition=FATransition(source=left_segment.sink, destination=sink)
        right_source_transition=FATransition(source=source, destination=right_segment.source)
        right_destination_transition=FATransition(source=right_segment.sink, destination=sink)
        return cls(sink=sink,source=source)

    def label_states(self):
        _, states = self.get_all_states_and_transitions()
        lengths_and_states=[]
        for state in states:
            lengths_and_states.append([len(state._incoming)>0+len(state._outgoing)==0,-state.get_descendant_count(),id(state),state])
        states=[s[3] for s in sorted(lengths_and_states)]
        for k,state in enumerate(states):
            state.id=k
            if state==self.source:
                state.is_starting = True
            else:
                state.is_starting=False
            if state == self.sink:
                state.is_ending = True
            else:
                state.is_ending = False
        return self

    def substitute_e_transitions(self):
        new_segment=self.clone()
        old_transitions, states = new_segment.get_all_states_and_transitions()
        for source in states:
            children_dict = source.get_non_null_children_and_their_null_descendants()
            for symbol in children_dict:
                for destination in children_dict[symbol]:
                    transition = FATransition(symbol_str=symbol,caption="R"+symbol)
                    transition.set_destination(destination)
                    transition.set_source(source)
        for transition in old_transitions:
            if not transition:
                transition.remove()
        return new_segment


    def get_dfa_start_state(self):
        """Takes a NFA segment and returns the start node of a DFA graph

        based on https://www.youtube.com/watch?v=taClnxU-nao
        self is an e-NFA segment


        Assumptions:
        There is a single source and a single sink state.
        And others...

        :return:
        """
        # nfas short Nondeterninistic Finite Automaton State
        # dfas short Deterninistic Finite Automaton State
        transitions,states=self.get_all_states_and_transitions()
        #symbol2state_list
        nfas_symbold2nfas=defaultdict(lambda:[]) # NFA transition table
        for state in states:
            symbol2state_list=state.get_children_and_null_descendants()
            for symbol,destination in list(symbol2state_list.items()):
                nfas_symbold2nfas[state.id,symbol].append(destination.id)

        alphabet=[k[1] for k in list(nfas_symbold2nfas.keys()) if k[1] is not None]

        unvisited_dfas=[(self.source.id)]
        dfas_symbol2dfas=defaultdict(lambda:[]) # DFA transition table
        visited_dfas=[] # holds which dfa states have been used as inputs
        while len(unvisited_dfas):
            source_dfas=unvisited_dfas.pop()
            for symbol in alphabet:
                dest_state = []
                for old_src_state in source_dfas:
                    for old_dst_state in nfas_symbold2nfas[old_src_state,symbol]:
                        dest_state.append(old_dst_state)
                        # transition to null from destination
                        dest_state.append(nfas_symbold2nfas[old_dst_state,None])
                dest_state=tuple(sorted(set(dest_state)))
                dfas_symbol2dfas[source_dfas,symbol]=dest_state
                unvisited_dfas.push(dest_state)
            visited_dfas.append(source_dfas)
            unvisited_dfas=list(set(dfas_symbol2dfas.values())-set(visited_dfas))
        all_dfa_state_tuples=[k[1] for k in list(dfas_symbol2dfas.keys())]+list(dfas_symbol2dfas.values())
        all_dfa_states={}
        for dfa_state_tuple in all_dfa_state_tuples:
            dfa_state = FAState(visual_caption=repr(dfa_state_tuple))
            all_dfa_states[dfa_state_tuple]=dfa_state
            dfa_state.is_starting=all([states[id].is_starting for id in dfa_state_tuple])
            dfa_state.is_ending=any([states[id].is_ending for id in dfa_state_tuple])

        for source_dfa_tuple,symbol in list(dfas_symbol2dfas.keys()):
            source_dfas=all_dfa_states[source_dfa_tuple]
            destination_dfas=all_dfa_states[dfas_symbol2dfas[(source_dfa_tuple,symbol)]]
            transition = FATransition(symbol_str=symbol,caption="DFA{%s}"%symbol)
            transition.set_source(source_dfas)
            transition.set_destination(destination_dfas)

        start_states = [s for s in list(all_dfa_states.values()) if s.is_starting]
        assert len(start_states)==1
        return start_states[0]

    def create_transition_matrices(self, start_regex=False, end_regex=False):
        """Creates eNFA transition matrices.

        Each symbol has a its own transition matrix

        :return: A defaultdict mapping symbols to transition matrices
        """
        all_transitions, states = self.get_all_states_and_transitions()
        ids=sorted(set([s.id for s in states]))
        assert min(ids) == 0
        state_sz=max(ids)+1
        transition_matrices=defaultdict(lambda:np.zeros([state_sz,state_sz]))
        transition_matrices[""]=np.eye(state_sz) # e transitions are reflective
        for transition in all_transitions:
            src = transition.source.id
            dst = transition.destination.id
            transition_matrices[transition.symbol_str][src,dst]=1
        # saturating e transitions
        e_transitions=transition_matrices[""]
        e_transitions=np.linalg.matrix_power(e_transitions,state_sz-1)
        del transition_matrices[""] # removing null transitions
        for symbol, matrix in list(transition_matrices.items()):
            matrix=np.dot(matrix,e_transitions)
            matrix[matrix>1]=1
            transition_matrices[symbol]=matrix

        starting_states=np.zeros(state_sz)
        accepting_states=np.zeros(state_sz)
        
        for state in states:
            if state.is_starting:
                starting_states[state.id]=1
            if state.is_ending:
                accepting_states[state.id]=1
        
        starting_states = np.dot(starting_states,e_transitions)
        starting_states[starting_states>0]=1
        starting_states = starting_states.reshape(-1)

        start_state_indexes = starting_states==1
        accept_state_indexes = accepting_states==1

        for symbol, matrix in list(transition_matrices.items()):
            if not start_regex: # if no ^ was set at the begining of the regex
                matrix[start_state_indexes, start_state_indexes] = 1
            if not end_regex: # if no ^ was set at the begining of the regex
                matrix[accept_state_indexes, accept_state_indexes] = 1
        return transition_matrices, starting_states, accepting_states
