#!/usr/bin/env python3
from matplotlib import pyplot as plt
import matplotlib
import numpy as np


class FAGraph(object):
    class FATransition(object):
        def __init__(self, symbol_str="", caption="\\0"):
            self.symbol_str=symbol_str
            self.caption=caption
            self._source = None
            self._destination = None

        @property
        def src_id(self):
            if self._source:
                return self._source.id
            else:
                return -2

        @property
        def dst_id(self):
            if self._source:
                return self._destination.id
            else:
                return -2

        @property
        def source(self):
            return self._source

        @property
        def destination(self):
            return self._destination

        def set_source(self,state):
            self._source=state
            if self not in state._outgoing:
                state._outgoing.append(self)

        def set_destination(self,state):
            self._destination=state
            if self not in state._incoming:
                state._incoming.append(self)

        def remove(self):
            dst=self._destination
            dst._incoming=[tr for tr in dst._incoming if tr is not self]
            src=self._source
            src._outgoing=[tr for tr in src._outgoing if tr is not self]
            self._destination=None
            self._source=None

        def __str__(self):
            return "%2d->T{%3s}->%2d"%(self.src_id,self.caption,self.dst_id)

        def __bool__(self):
            return len(self.symbol_str) > 0
        def __bool__ (self):
            return len(self.symbol_str) > 0

        def clone(self,source=None,destination=None):
            res=FAGraph.FATransition(symbol_str=self.symbol_str, caption=self.caption)
            res._source=source
            res._destination=destination
            return res

        @classmethod
        def create_transition(cls, symbol_str=".",caption="\\u2205"):
            return cls(symbol_str=symbol_str,caption=caption)

    class FAState(object):
        def __init__(self,visual_caption="NA"):
            self.is_accepting=False
            self.visual_caption=visual_caption
            self.id=None
            self._incoming = []
            self._outgoing = []
            self.is_starting = None
            self.is_ending = None

        def add_incoming(self,transition):
            self._incoming.append(transition)

        def add_outgoing(self,transition):
            self._outgoing.append(transition)

        def remove_incoming(self,state_or_transition,tolerate_non_existing=False):
            if isinstance(state_or_transition,FAGraph.FAState):
                state=state_or_transition
                new_incoming=[i for i in self._incoming if i.source is not state]
                if not tolerate_non_existing and len(new_incoming)==len(self._incoming):
                    raise ValueError
                self._incoming=new_incoming
            elif isinstance(state_or_transition,FAGraph.FATransition):
                transition=state_or_transition
                new_incoming=[i for i in self._incoming if i is not transition]
                if not tolerate_non_existing and len(new_incoming)==len(self._incoming):
                    raise ValueError
                self._incoming=new_incoming

        def remove_outgoing(self,state_or_transition,tolerate_non_existing=True):
            if isinstance(state_or_transition,FAGraph.FAState):
                state = state_or_transition
                new_outgoing=[i for i in self._outgoing if i.destination is not state]
                if not tolerate_non_existing or len(new_outgoing)==len(self._outgoing):
                    raise ValueError
                self._outgoing=new_outgoing
            elif isinstance(state_or_transition,FAGraph.FATransition):
                transition = state_or_transition
                new_outgoing=[i for i in self._outgoing if i is not transition]
                if not tolerate_non_existing and len(new_outgoing)==len(self._outgoing):
                    raise ValueError
                self._outgoing=new_outgoing

        @property
        def incoming_states(self):
            return tuple([transition.source for transition in self._incoming])

        @property
        def incoming_transitions(self):
            return tuple(self._incoming)

        @property
        def outgoing_states(self):
            return tuple([transition.destination for transition in self._outgoing])

        @property
        def outgoing_transitions(self):
            return tuple(self._outgoing)

        def clone(self):
            res=FAGraph.FAState(visual_caption=self.visual_caption)
            res.is_ending=self.is_ending
            res.is_starting=self.is_starting
            res.id=self.id
            return res

        def get_descendant_count(self):
            return len(self.get_descendants())

        def get_descendants(self):
            visited=set()
            self._get_descendants(visited)
            return list(visited)

        def _get_descendants(self,visited):
            children=[tr.destination for tr in self._outgoing]
            for child in children:
                if child not in visited:
                    visited.add(child)
                    child._get_descendants(visited)

        def get_children_and_null_descendants(self):
            children_dict={}
            for transition in self.outgoing_transitions:
                if bool(transition):
                    children_dict[transition.symbol_str]=transition.destination
            children_dict[None]=[self]+self._get_null_descendants([])
            return children_dict

        def get_non_null_children_and_their_null_descendants(self):
            """Returns all nodes that are a single symbol away from self.

            e*s union s*e
            :return:
            """
            children_dict=defaultdict(lambda:[])
            for transition in self.outgoing_transitions:
                if bool(transition): # symbol following Null
                    children=transition.destination._get_null_descendants([])
                    if transition.destination not in children:
                        children.append(transition.destination)
                    children_dict[transition.symbol_str]+=(children)

            null_descendants=self._get_null_descendants([])
            for null_child in null_descendants:
                for transition in null_child._outgoing:
                    if transition: # A non-null child of a null children
                        children_dict[transition.symbol_str].append(transition.destination)
            children_dict={k:list(set(v)) for k,v in list(children_dict.items())}
            return children_dict

        def _get_null_descendants(self,visited):
            result=[]
            for transition in self.outgoing_transitions:
                if (transition.destination not in visited) and not bool(transition):
                    result.append(transition.destination)
                    transition.destination._get_null_descendants(visited+[self])
            return result

        @property
        def is_removable(self):
            all_nodes=self.incoming_states + self.outgoing_states+(self,)
            if len(all_nodes)!=len(set(all_nodes)):
                return False # Can't be allowed to create
            if len(self._outgoing)==0 or len(self._incoming)==0:
                return False
            if len(self._outgoing)==1 and not self._outgoing[0]:
                return True
            if len(self._incoming) == 1 and not self._incoming[0]:
                return True
            return False

        def remove(self):
            if len(self._outgoing)==1 and not(self._outgoing[0]):
                replacing_state=self._outgoing[0].destination
                for tr in self._incoming:
                    tr.set_destination(replacing_state)
                replacing_state.remove_incoming(self)
            elif len(self._incoming)==1 and not(self._incoming[0]):
                replacing_state=self._incoming[0].source
                for tr in self._outgoing:
                    tr.set_source(replacing_state)
                replacing_state.remove_outgoing(self)
            else:
                raise ValueError

        def __str__(self):
            id=self.id
            start_str=">" if self.is_starting else " "
            end_str = "<" if self.is_ending else " "
            if id is None:
                id=-1
            return "%sS{%d|%d>%d}%s"%(start_str,id,len(self.incoming_transitions),len(self.outgoing_transitions),end_str)

        def __lt__(self,other):
            python_id_cmp=.5*(id(self)<id(other))
            return self.id+python_id_cmp<other.id



    class NFASegment(object):
        def __init__(self, _=None, source=None, sink=None):
            self.source=source
            self.sink=sink

        def get_nfa_matrices(self,embedding_dict):
            transitions, states = self.get_all_states_and_transitions()
            null_transitions=[tr for tr in transitions if not tr]
            symbol_transitions = [tr for tr in transitions if tr]
            symbols={symbol:n for n,symbol in enumerate(sorted(set([tr.symbol_str for tr in valid_transitions])))}
            symbols_mat=np.array([embedding_dict[symbol] for symbol in list(symbols.keys())])
            symbol_transition_mat=np.zeros([len(states),len(states),len(symbols)])
            null_transition_mat=np.eye(len(states))
            starting_states=np.zeros(len(states))
            ending_states = np.zeros(len(states))
            for state in states:
                if state.is_starting:
                    starting_states[state.id]=1
                if state.is_ending:
                    ending_states[state.id]=1
            for tr in symbol_transitions:
                symbol_transition_mat[tr.destination.id,tr.source.id,symbols[tr]]=1
            for tr in null_transitions:
                null_transition_mat[tr.destination.id,tr.source.id]=1
            # null states transit infinetly
            null_transition_mat=np.linalg.matrix_power(null_transition_mat,len(states))
            null_transition_mat[null_transition_mat>1]=1
            return starting_states,ending_states,symbol_transition_mat,null_transition_mat,symbols_mat

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
            transitions,states=new_segment.get_all_states_and_transitions()
            print("copied transitions\n\t")
            print(("\n\t".join([str(tr) for tr in transitions])))
            print("\n\t}\n")
            while any([st.is_removable for st in states]):
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
            ending_states=[s for s in states if s.is_starting]
            assert len(ending_states) == 1 #TODO remove this check
            return FAGraph.NFASegment(source=starting_states[0],sink=ending_states[0])


        def get_all_states_and_transitions(self):
            transitions,states= self._get_all_states_and_transitions(self.source)
            return transitions,states


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
            return FAGraph.NFASegment(source=new_source, sink=new_sink)


        def __str__(self):
            transitions,states=self.get_all_states_and_transitions()
            states=[st[1] for st in sorted([(s.id,s) for s in states])]
            res="\n".join(["%3d:%s"%(k,v) for k,v in enumerate(states)])
            res="States:\n"+res+"\nTransitions:\n"
            for transition in transitions:
                res+=str(transition)+"\n"
            return res+"\n"

        @classmethod
        def create_transition_segment(cls,symbol_str,caption):
            transition=FAGraph.FATransition(symbol_str=symbol_str,caption=caption)
            source=FAGraph.FAState()
            sink=FAGraph.FAState()
            transition.set_destination(sink)
            transition.set_source(source)
            return cls(sink=sink,source=source)

        @classmethod
        def create_concatenate_segment(cls,left_segment,right_segment):
            transition=FAGraph.FATransition()
            transition.set_destination(right_segment.source)
            transition.set_source(left_segment.sink)
            return cls(sink=right_segment.sink,source=left_segment.source)

        @classmethod
        def create_asterisk_segment(cls,input_segment):
            source=FAGraph.FAState()
            sink=FAGraph.FAState()
            source_transition=FAGraph.FATransition()
            source_transition.set_source(source)
            source_transition.set_destination(input_segment.source)

            destination_transition=FAGraph.FATransition()
            destination_transition.set_destination(sink)
            destination_transition.set_source(input_segment.sink)

            loop_transition=FAGraph.FATransition()
            loop_transition.set_source(input_segment.sink)
            loop_transition.set_destination(input_segment.source)

            return cls(sink=sink,source=source)

        @classmethod
        def create_question_segment(cls,input_segment):
            source=FAGraph.FAState()
            sink=FAGraph.FAState()

            source_transition=FAGraph.FATransition()
            source_transition.set_source(source)
            source_transition.set_destination(input_segment.source)

            destination_transition=FAGraph.FATransition()
            destination_transition.set_destination(sink)
            destination_transition.set_source(input_segment.sink)

            shortcut_transition=FAGraph.FATransition()
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
            source=FAGraph.FAState()
            sink=FAGraph.FAState()

            left_source_transition=FAGraph.FATransition()
            left_source_transition.set_source(source)
            left_source_transition.set_destination(left_segment.source)

            left_destination_transition=FAGraph.FATransition()
            left_destination_transition.set_destination(sink)
            left_destination_transition.set_source(left_segment.sink)

            right_source_transition=FAGraph.FATransition()
            right_source_transition.set_source(source)
            right_source_transition.set_destination(right_segment.source)

            right_destination_transition=FAGraph.FATransition()
            right_destination_transition.set_destination(sink)
            right_destination_transition.set_source(right_segment.sink)
            return cls(sink=sink,source=source)

        def label_states(self):
            _,states=self.get_all_states_and_transitions()
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
            old_transitions,states=new_segment.get_all_states_and_transitions()
            for source in states:
                children_dict=source.get_non_null_children_and_their_null_descendants()
                for symbol in children_dict:
                    for destination in children_dict[symbol]:
                        transition=FAGraph.FATransition(symbol_str=symbol,caption="R"+symbol)
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
                dfa_state=FAGraph.FAState(visual_caption=repr(dfa_state_tuple))
                all_dfa_states[dfa_state_tuple]=dfa_state
                dfa_state.is_starting=all([states[id].is_starting for id in dfa_state_tuple])
                dfa_state.is_ending=any([states[id].is_ending for id in dfa_state_tuple])

            for source_dfa_tuple,symbol in list(dfas_symbol2dfas.keys()):
                source_dfas=all_dfa_states[source_dfa_tuple]
                destination_dfas=all_dfa_states[dfas_symbol2dfas[(source_dfa_tuple,symbol)]]
                transition=FAGraph.FATransition(symbol_str=symbol,caption="DFA{%s}"%symbol)
                transition.set_source(source_dfas)
                transition.set_destination(destination_dfas)

            start_states = [s for s in list(all_dfa_states.values()) if s.is_starting]
            assert len(start_states)==1
            return start_states[0]

        def create_transition_matrices(self):
            """Creates eNFA transition matrices.

            Each symbol has a its own transition matrix

            :return: A defaultdict mapping symbols to transition matrices
            """
            all_transitions,states=self.get_all_states_and_transitions()
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
            starting_states=np.dot(starting_states,e_transitions)
            return transition_matrices,starting_states,accepting_states


    @staticmethod
    def create_matrices(state_list,symbol2idx,max_idx=None):
        if max_idx is None:
            max_idx=max(symbol2idx.values())

        output_states=np.zeros(self.nb_states)
        output_states[[k for k in range(len(self)) if self[k].is_ending]]=1

        state2state=np.zeros([len(self),len(self)])
        for state_id in range(len(self)):
            in_state=self[state_id]
            state2state[state_id,[s.id for s in  in_state.outgoing_states]]=1

        state2symbol=np.zeros([len(self),max_idx+1])
        symbol2state = np.zeros([max_idx + 1, len(self)])
        for transition in self.all_transitions:
            state2symbol[transition.source.id,symbol2idx[transition.symbol_str]]=1
            symbol2state[symbol2idx[transition.symbol_str],transition.destination.id,]=1
        return output_states,state2state,state2symbol,symbol2state


    @staticmethod
    def draw_graph(nfa_segment,re_str="NA"):
        plt.figure()
        all_transitions,states=nfa_segment.get_all_states_and_transitions()
        null_transitions=[tr for tr in all_transitions if not tr]
        valid_transitions = [tr for tr in all_transitions if tr]
        x_node_step=30
        state_sz=5
        text_dx = 2
        text_dy = 2
        valid_arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=8"
        null_arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=8"
        plt.axes().set_xlim(-2*state_sz, (max([1+s.id for s in states]) - 1) * x_node_step + state_sz*2)
        plt.axes().set_ylim(-40, 40)
        plt.axes().set_aspect(1)

        patches=[]
        captions_and_coords=[]

        for state in states:
            x=state.id * x_node_step
            y=0
            patches.append(
                matplotlib.patches.Circle([x, y], radius=state_sz,
                                          color=[0, 1, 0]))
            captions_and_coords.append([x-text_dx,y-text_dy,str(state.id)])

        for null_transition in null_transitions:
            src_id=null_transition.src_id
            dst_id = null_transition.dst_id
            y=(src_id<dst_id)*state_sz-(src_id>=dst_id)*state_sz
            patches.append(matplotlib.patches.FancyArrowPatch((src_id*x_node_step, y), (dst_id*x_node_step,y),
                                        connectionstyle="arc3,rad=-.2",arrowstyle=null_arrowstyle,color=[.5,.5,.5]))

        for valid_transition in valid_transitions:
            src_id=valid_transition.src_id
            dst_id = valid_transition.dst_id
            y=(src_id<dst_id)*state_sz-(src_id>=dst_id)*state_sz
            patches.append(matplotlib.patches.FancyArrowPatch((src_id*x_node_step, y), (dst_id*x_node_step,y),
                                        connectionstyle="arc3,rad=-.2",arrowstyle=valid_arrowstyle))
            x = (src_id + dst_id)*x_node_step / 2.0
            y=.15*(dst_id*x_node_step - src_id*x_node_step)
            captions_and_coords.append([x-text_dx,y-text_dy,valid_transition.caption])

        for p in patches:
            plt.gca().add_patch(p)
        for x,y,caption in captions_and_coords:
            plt.text(x,y,caption)
        plt.title(re_str)

    def parse_re(self,re_str,alphabet_map):
        import ply.lex as lex
        tokens = (
            'START','END','LETTER', 'DIGIT', 'SYMBOL','ANY',
            'ASTERISK', 'PLUS', 'QUESTION', 'OR', 'LPAR', 'RPAR',
        )

        # Tokens
        t_START = r'\^'
        t_END = r'\$'
        t_LETTER = r'[a-zA-Z]'
        t_DIGIT = r'[0-9]'
        t_SYMBOL = r'[!@$%\^:;,]'
        t_ANY = r'\.'
        t_ASTERISK = r'\*'
        t_PLUS = r'\+'
        t_QUESTION = r'\?'
        t_OR = r'\|'
        t_LPAR = r'\('
        t_RPAR = r'\)'

        def t_error(t):
            raise TypeError("Unknown re token '%s'" % (t.value,))

        lex.lex()

        import ply.yacc as yacc
        lex.input(re_str)

        precedence = (
            ('right', 'QUESTION','PLUS','ASTERISK'),
            ('left', 'OR'),
        )

        def p_expression_symbol(p):
            """expression : LETTER
                          | DIGIT
                          | SYMBOL
                          | ANY
                          | START
                          | END"""
            p[0]=FAGraph.NFASegment.create_transition_segment(symbol_str=p[1], caption=p[1])

        def p_expression_concatenate(p):
            """expression : expression expression"""
            p[0]=FAGraph.NFASegment.create_concatenate_segment(p[1], p[2])

        def p_expression_postfix(p):
            """expression : expression ASTERISK
                          | expression QUESTION
                          | expression PLUS"""
            if p[2]=="?":
                p[0]=FAGraph.NFASegment.create_question_segment(p[1])
            elif p[2]=="+":
                p[0] = FAGraph.NFASegment.create_plus_segment(p[1])
            elif p[2]=="*":
                p[0]=FAGraph.NFASegment.create_asterisk_segment(p[1])
            else:
                raise ValueError

        def p_expression_or(p):
            'expression : expression OR expression'
            p[0] = FAGraph.NFASegment.create_or_segment(p[1], p[3])

        def p_expression_group(p):
            'expression : LPAR expression RPAR'
            p[0] = p[2]

        def p_error(t):
            print(("Syntax error at '%s'" % t.value))

        import ply.yacc as yacc
        parser = yacc.yacc()
        self.re_str=re_str
        self.nfa_segment=parser.parse(re_str)
        self.nfa_segment.label_states()


    def __str__(self):
        res= "NFA Graph:\n"+str(self.nfa_segment)
        return res

#import lm_util






if __name__=="__main__":
    p={"regex":"a?","string":"aaa","alphabet":"dabc"}
    p,_=fargv.fargv(p)
    re_str=p.string
    alphabet={v:k for k,v in enumerate(p.alphabet)}
    g=FAGraph()
    g.parse_re(p.regex, alphabet)
    print((str(g)))
    FAGraph.draw_graph(g.nfa_segment,re_str=p.regex)
    reduced_g=g.nfa_segment.reduce_states()
    FAGraph.draw_graph(reduced_g,re_str=p.regex)
    plt.show()
    non_e=reduced_g.substitute_e_transitions()
    FAGraph.draw_graph(non_e, re_str=p.regex)
    plt.show()
    print((str(non_e)))
