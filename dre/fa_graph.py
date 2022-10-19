from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
from .fa_segment import NFASegment


class FAGraph(object):
    # @staticmethod
    # def create_matrices(state_list,symbol2idx,max_idx=None):
    #     if max_idx is None:
    #         max_idx=max(symbol2idx.values())

    #     output_states=np.zeros(self.nb_states)
    #     output_states[[k for k in range(len(self)) if self[k].is_ending]]=1

    #     state2state=np.zeros([len(self),len(self)])
    #     for state_id in range(len(self)):
    #         in_state=self[state_id]
    #         state2state[state_id,[s.id for s in  in_state.outgoing_states]]=1

    #     state2symbol=np.zeros([len(self),max_idx+1])
    #     symbol2state = np.zeros([max_idx + 1, len(self)])
    #     for transition in self.all_transitions:
    #         state2symbol[transition.source.id,symbol2idx[transition.symbol_str]]=1
    #         symbol2state[symbol2idx[transition.symbol_str],transition.destination.id,]=1
    #     return output_states,state2state,state2symbol,symbol2state


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
    
    def plot(self, filename="/tmp/nfa.gv", view=False):
        import graphviz
        transitions, states = self.nfa_segment.get_all_states_and_transitions()

        f = graphviz.Digraph('finite_state_machine', filename=filename)
        f.attr(rankdir='LR', size='8,5')

        f.attr('node', shape='doublecircle')
        for state in sorted(self._accepting_states):
            f.node(f'S{state}')

        f.attr('node', shape='circle')
        for state in sorted(set(range(len(states)))-set(self._accepting_states)):
            f.node(f'S{state}')

        f.attr('node', style='invis')
        f.node(f'')
        for n in sorted(self._starting_states):
            f.edge('', f'S{n}', label='')

        from collections import defaultdict
        d = defaultdict(lambda:[])
        for tr in transitions:
            print(str(tr))
            d[str(tr)].append((object.__hash__(tr),tr))
        print("PLOT:",d)
        for transition in transitions:
            src_node = f"S{transition.src_id}"
            dst_node = f"S{transition.dst_id}"
            label = f"{transition.caption}"
            f.edge(src_node, dst_node, label=label)
        if view:
            f.view()
        else:
            f.render(filename=filename)



    def __str__(self):
        res= "NFA Graph:\n"+str(self.nfa_segment)
        return res


    def __init__(self, segment:NFASegment, start:bool, end:bool, re_str:str="N/A") -> None:
        self.start = start
        self.end = end
        self.nfa_segment = segment
        self.re_str = re_str
        
        self.nfa_segment = self.nfa_segment.label_states()
        self.nfa_segment = self.nfa_segment.reduce_states()
        #self.nfa_segment = self.nfa_segment.substitute_e_transitions()
        
        self.nfa_segment = self.nfa_segment.label_states()

        _, starting_states, accepting_states = self.nfa_segment.create_transition_matrices()
        self._starting_states = np.nonzero(starting_states)[0].tolist()
        self._accepting_states = np.nonzero(accepting_states)[0].tolist()

#import lm_util

