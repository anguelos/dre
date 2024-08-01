from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
from .fa_segment import NFASegment


class FAGraph(object):
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
        f.node('')
        for n in sorted(self._starting_states):
            f.edge('', f'S{n}', label='')

        d = defaultdict(lambda: [])
        for tr in transitions:
            print(str(tr))
            d[str(tr)].append((object.__hash__(tr), tr))
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
        res = "NFA Graph:\n"+str(self.nfa_segment)
        return res

    def __init__(self, segment: NFASegment, start: bool, end: bool, re_str: str="N/A") -> None:
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
