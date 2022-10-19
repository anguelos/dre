from collections import defaultdict
import numpy as np


from .fa_state import FAState
from .fa_transition import FATransition

class NFASegment(object):
    def __init__(self, _=None, source=None, sink=None):
        self.source=source
        self.sink=sink
    
    @property
    def reduced_source(self):
        src = self.source
        while isinstance(src,FAState) and len(src._outgoing) == 1 and bool(src._outgoing[0]) and isinstance(src._outgoing[0].destination, FAState):
            src = src._outgoing[0].destination
        return src

    @property
    def reduced_sink(self):
        dst = self.sink
        while isinstance(dst,FAState) and len(dst._incoming) == 1 and bool(dst._incoming[0]) and isinstance(dst._incoming[0].source, FAState):
            dst = dst._incoming[0].source
        return dst

    # def get_nfa_matrices(self,embedding_dict):
    #     transitions, states = self.get_all_states_and_transitions()
    #     null_transitions=[tr for tr in transitions if not tr]
    #     symbol_transitions = [tr for tr in transitions if tr]
    #     symbols={symbol:n for n,symbol in enumerate(sorted(set([tr.symbol_str for tr in valid_transitions])))}
    #     symbols_mat=np.array([embedding_dict[symbol] for symbol in list(symbols.keys())])
    #     symbol_transition_mat=np.zeros([len(states),len(states),len(symbols)])
    #     null_transition_mat=np.eye(len(states))
    #     starting_states=np.zeros(len(states))
    #     ending_states = np.zeros(len(states))
    #     for state in states:
    #         if state.is_starting:
    #             starting_states[state.id]=1
    #         if state.is_ending:
    #             ending_states[state.id]=1
    #     #for tr in symbol_transitions:
    #     #    symbol_transition_mat[tr.destination.id,tr.source.id,symbols[tr]]=1
    #     #for tr in null_transitions:
    #     #    null_transition_mat[tr.destination.id,tr.source.id]=1
    #     # null states transit infinetly
    #     #null_transition_mat=np.linalg.matrix_power(null_transition_mat,len(states))
    #     #null_transition_mat[null_transition_mat>1]=1
    #     return starting_states,ending_states,symbol_transition_mat,null_transition_mat,symbols_mat

    def is_dfa(self):
        transitions, states = self.get_all_states_and_transitions()
        non_e_transitions=all([bool(tr) for tr in transitions])
        no_multiple_outgoing=[]
        for state in states:
            outgoing_symbols=[tr.symbol_str for tr in state._outgoing]
            no_multiple_outgoing.append(len(outgoing_symbols)==len(set(outgoing_symbols)))
        return non_e_transitions and all(no_multiple_outgoing)

    def reduce_states(self):
        new_segment=self.clone()
        _, states=new_segment.get_all_states_and_transitions()

        while any([st.is_removable for st in sorted(sorted(states))]):
            new_states = []
            for state in states:
                if state.is_removable:
                    print(("Removing:",str(state)))
                    state.remove()
                else:
                    new_states.append(state)
            states=new_states
        starting_states=[s for s in states if s.is_starting]
        assert len(starting_states)==1 #TODO remove this check
        ending_states=[s for s in states if s.is_ending]
        assert len(ending_states) == 1 #TODO remove this check
        return NFASegment(source=starting_states[0],sink=ending_states[0])


    def get_all_states_and_transitions(self):
        transitions, states = self._get_all_states_and_transitions(self.source)
        return transitions, states

    def _get_all_states_and_transitions(self,state,visited_states=()):
        if state in visited_states:
            return (),()
        transitions=state.incoming_transitions+state.outgoing_transitions
        states=(state,)+state.incoming_states+state.outgoing_states
        for new_state in set(states)-set(visited_states):
            new_transitions,new_states=self._get_all_states_and_transitions(new_state,visited_states+(state,))
            transitions+=new_transitions
            states+=new_states
        return tuple(set(transitions)),tuple(set(states))


    def clone(self):
        transitions,states=self.get_all_states_and_transitions()
        old2new_states={s:s.clone() for s in states}
        old2new_transitions = {t: t.clone() for t in transitions}
        for old_transition,new_transition in list(old2new_transitions.items()):
            new_destination=old2new_states[old_transition.destination]
            new_source = old2new_states[old_transition.source]
            new_transition.set_destination(new_destination)
            new_transition.set_source(new_source)
        new_source=old2new_states[self.source] # source of the segment
        new_sink = old2new_states[self.sink]
        return NFASegment(source=new_source, sink=new_sink)


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
        transition=FATransition(symbol_str=symbol_str,caption=caption)
        source=FAState()
        sink=FAState()
        transition.set_destination(sink)
        transition.set_source(source)
        return cls(sink=sink,source=source)

    @classmethod
    def create_concatenate_segment(cls,left_segment,right_segment):
        transition=FATransition()
        transition.set_destination(right_segment.source)
        transition.set_source(left_segment.sink)
        return cls(sink=right_segment.sink,source=left_segment.source)

    @classmethod
    def create_asterisk_segment(cls,input_segment):
        source=FAState()
        sink=FAState()
        source_transition=FATransition()
        source_transition.set_source(source)
        source_transition.set_destination(input_segment.source)

        destination_transition=FATransition()
        destination_transition.set_destination(sink)
        destination_transition.set_source(input_segment.sink)

        loop_transition=FATransition()
        loop_transition.set_source(input_segment.sink)
        loop_transition.set_destination(input_segment.source)

        return cls(sink=sink,source=source)

    @classmethod
    def create_question_segment(cls,input_segment):
        # source=FAState()
        # sink=FAState()

        # source_transition=FATransition()
        # source_transition.set_source(source)
        # source_transition.set_destination(input_segment.reduced_source)

        # destination_transition=FATransition()
        # destination_transition.set_destination(sink)
        # destination_transition.set_source(input_segment.reduced_sink)

        # shortcut_transition=FATransition()
        # shortcut_transition.set_source(source)
        # shortcut_transition.set_destination(sink)

        # return cls(sink=sink,source=source)
        source=FAState()
        sink=FAState()

        source_transition=FATransition()
        source_transition.set_source(source)
        source_transition.set_destination(input_segment.source)

        destination_transition=FATransition()
        destination_transition.set_destination(sink)
        destination_transition.set_source(input_segment.sink)

        shortcut_transition=FATransition()
        shortcut_transition.set_source(source)
        shortcut_transition.set_destination(sink)

        return cls(sink=sink,source=source)

    @classmethod
    def create_plus_segment(cls,input_segment):
        input_segment_copy=input_segment.clone()
        asterisk=cls.create_asterisk_segment(input_segment_copy)
        return cls.create_concatenate_segment(asterisk,input_segment_copy)


    @classmethod
    def create_or_segment(cls,left_segment,right_segment):
        source=FAState()
        sink=FAState()

        left_source_transition=FATransition()
        left_source_transition.set_source(source)
        left_source_transition.set_destination(left_segment.source)

        left_destination_transition=FATransition()
        left_destination_transition.set_destination(sink)
        left_destination_transition.set_source(left_segment.sink)

        right_source_transition=FATransition()
        right_source_transition.set_source(source)
        right_source_transition.set_destination(right_segment.source)

        right_destination_transition=FATransition()
        right_destination_transition.set_destination(sink)
        right_destination_transition.set_source(right_segment.sink)
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
