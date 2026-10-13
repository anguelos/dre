import numpy as np
import torch
from torch import Tensor
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from .fa_segment import NFASegment
from .optional import optional_import

graphviz = optional_import("graphviz")


ANY = "."  # the symbol of transitions reading any token


def is_class(symbol: str) -> bool:
    # token classes such as <number>; which tokens belong to a class is decided by the caller
    return symbol.startswith("<") and symbol.endswith(">")


@dataclass
class Automaton:
    # What the engine (emissions.py, forward.py) consumes: one transition matrix per query symbol, ε-transitions folded in.
    regex: str
    symbols: list[str]  # query symbols, ANY included if used
    transitions: Tensor  # (K, S, S): transitions[k, i, j] = 1 iff reading symbols[k] in state i can lead to state j
    start: Tensor  # (S,) states active before reading anything
    accept: Tensor  # (S,)

    @property
    def n_states(self) -> int:
        return self.start.numel()

    @property
    def any_mask(self) -> Tensor:
        # (K,) True for the symbol reading any token: what a matcher needs to know about symbols, without their strings
        return torch.tensor([symbol == ANY for symbol in self.symbols], dtype=torch.bool)

    @property
    def word_mask(self) -> Tensor:
        # (K,) True for query words: their emissions are probabilities, while ANY and token classes are 0/1 tests
        return torch.tensor([symbol != ANY and not is_class(symbol) for symbol in self.symbols], dtype=torch.bool)

    def reversed(self) -> "Automaton":
        # Reads token sequences backwards: a path start -> accept becomes accept -> start (ε-closures are folded into
        # the symbol matrices, so transposing them is enough).
        return Automaton(self.regex, self.symbols, self.transitions.transpose(1, 2).contiguous(), self.accept.clone(), self.start.clone())

    def determinize(self, letters: Tensor, search: bool = False, max_states: int = 1 << 16) -> "DFA":
        # Subset construction. letters: (L, K) bool, the symbols each letter satisfies (a token can be a query word, in
        # a token class and ANY at once, so a letter is a set of symbols). A DFA state is a set of automaton states.
        # search: the start states are re-entered before every letter, as if the regex were ". * regex".
        steps = (torch.einsum("lk,kij->lij", letters.float(), self.transitions) > 0).float()            # (L, S, S)
        start = self.start > 0
        initial = torch.zeros_like(start) if search else start
        index, states, frontier, delta = {initial.numpy().tobytes(): 0}, [initial], [0], {}
        while frontier:
            sources = torch.stack([states[d] for d in frontier])
            if search:
                sources = sources | start
            reached = torch.einsum("fi,lij->flj", sources.float(), steps) > 0                          # (F, L, S)
            next_frontier = []
            for row, d in enumerate(frontier):
                for letter in range(steps.size(0)):
                    key = reached[row, letter].numpy().tobytes()
                    if key not in index:
                        if len(states) == max_states:
                            raise RuntimeError(f"The DFA of {self.regex!r} exceeds max_states={max_states}")
                        index[key] = len(states)
                        states.append(reached[row, letter])
                        next_frontier.append(index[key])
                    delta[d, letter] = index[key]
            frontier = next_frontier
        transitions = torch.zeros(len(states), steps.size(0), dtype=torch.long)
        for (d, letter), destination in delta.items():
            transitions[d, letter] = destination
        accept = torch.stack(states).float() @ (self.accept > 0).float() > 0
        return DFA(self.regex, letters, transitions, accept, 0)


@dataclass
class DFA:
    # A deterministic automaton over letters (sets of an Automaton's symbols, see Automaton.determinize): in every state
    # each letter leads to exactly one state.
    regex: str
    letters: Tensor      # (L, K) bool: letters[l, k] iff letter l satisfies symbol k
    transitions: Tensor  # (D, L) long: the state reached from state d by reading letter l
    accept: Tensor       # (D,) bool
    initial: int

    @property
    def n_states(self) -> int:
        return self.accept.numel()

    def minimized(self) -> "DFA":
        # Moore's partition refinement: states are split by acceptance, then by the blocks their letters lead to, until
        # nothing splits; each block becomes one state. (All states are reachable: determinize only creates those.)
        block = self.accept.long()
        n_blocks = int(block.unique().numel())
        while True:
            signature = torch.cat([block[:, None], block[self.transitions]], dim=1)                     # (D, 1 + L)
            _, refined = torch.unique(signature, dim=0, return_inverse=True)
            n_refined = int(refined.max()) + 1
            block = refined
            if n_refined == n_blocks:
                break
            n_blocks = n_refined
        representative = torch.zeros(n_blocks, dtype=torch.long).scatter_(0, block, torch.arange(block.numel()))
        return DFA(self.regex, self.letters, block[self.transitions[representative]], self.accept[representative], int(block[self.initial]))


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


    # @staticmethod
    # def draw_graph(nfa_segment,re_str="NA"):
    #     plt.figure()
    #     all_transitions,states=nfa_segment.get_all_states_and_transitions()
    #     null_transitions=[tr for tr in all_transitions if not tr]
    #     valid_transitions = [tr for tr in all_transitions if tr]
    #     x_node_step=30
    #     state_sz=5
    #     text_dx = 2
    #     text_dy = 2
    #     valid_arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=8"
    #     null_arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=8"
    #     plt.axes().set_xlim(-2*state_sz, (max([1+s.id for s in states]) - 1) * x_node_step + state_sz*2)
    #     plt.axes().set_ylim(-40, 40)
    #     plt.axes().set_aspect(1)

    #     patches=[]
    #     captions_and_coords=[]

    #     for state in states:
    #         x=state.id * x_node_step
    #         y=0
    #         patches.append(
    #             matplotlib.patches.Circle([x, y], radius=state_sz,
    #                                       color=[0, 1, 0]))
    #         captions_and_coords.append([x-text_dx,y-text_dy,str(state.id)])

    #     for null_transition in null_transitions:
    #         src_id=null_transition.src_id
    #         dst_id = null_transition.dst_id
    #         y=(src_id<dst_id)*state_sz-(src_id>=dst_id)*state_sz
    #         patches.append(matplotlib.patches.FancyArrowPatch((src_id*x_node_step, y), (dst_id*x_node_step,y),
    #                                     connectionstyle="arc3,rad=-.2",arrowstyle=null_arrowstyle,color=[.5,.5,.5]))

    #     for valid_transition in valid_transitions:
    #         src_id=valid_transition.src_id
    #         dst_id = valid_transition.dst_id
    #         y=(src_id<dst_id)*state_sz-(src_id>=dst_id)*state_sz
    #         patches.append(matplotlib.patches.FancyArrowPatch((src_id*x_node_step, y), (dst_id*x_node_step,y),
    #                                     connectionstyle="arc3,rad=-.2",arrowstyle=valid_arrowstyle))
    #         x = (src_id + dst_id)*x_node_step / 2.0
    #         y=.15*(dst_id*x_node_step - src_id*x_node_step)
    #         captions_and_coords.append([x-text_dx,y-text_dy,valid_transition.caption])

    #     for p in patches:
    #         plt.gca().add_patch(p)
    #     for x,y,caption in captions_and_coords:
    #         plt.text(x,y,caption)
    #     plt.title(re_str)
    
    def plot(self, filename="/tmp/nfa.gv", view=False):
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

    def to_automaton(self, normalize: Callable[[str], str] | None = None) -> Automaton:
        # No search self-loops (start_regex=end_regex=True): the engine implements search and anchored matching itself.
        # normalize: applied to query words (not to ANY or token classes); words with the same normalized form become one
        # symbol, its transitions the union of theirs.
        transitions, starting_states, accepting_states = self.nfa_segment.create_transition_matrices(start_regex=True, end_regex=True)
        if normalize is not None:
            merged: dict[str, np.ndarray] = {}
            for symbol, matrix in transitions.items():
                key = symbol if symbol == ANY or is_class(symbol) else normalize(symbol)
                merged[key] = np.maximum(merged[key], matrix) if key in merged else matrix
            transitions = merged
        symbols = sorted(transitions.keys())
        matrices = torch.tensor(np.stack([transitions[symbol] for symbol in symbols]), dtype=torch.float32)
        start = torch.tensor(starting_states, dtype=torch.float32)
        accept = torch.tensor(accepting_states, dtype=torch.float32)
        return Automaton(self.re_str, symbols, matrices, start, accept)


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

