import numpy as np
import finit_automata
from matplotlib import pyplot as plt
import torch
"""
Active alphabet the part of an alphabet that participates in the Reg-Expr.
The set of all possible non-null transitions in an eNFA.
"""

def simple_regex_match(symbol_seq,transition_matrix_dict,starting_states,accepting_states,substring=False):
    current_state = starting_states.copy()
    for symbol in symbol_seq:
        if substring:
            current_state=np.maximum(current_state,starting_states)
        current_state=np.dot(current_state,transition_matrix_dict[symbol])
        if substring and np.any(current_state*accepting_states):
            return True
    return np.any(current_state*accepting_states)

def trace_simple_regex_match(symbol_seq,transition_matrix_dict,starting_states,accepting_states):
    all_states=[starting_states]
    current_state = starting_states.copy()
    for symbol in symbol_seq:
        current_state=np.dot(current_state,transition_matrix_dict[symbol])
        all_states.append(current_state)
    return np.array(all_states)


# def torch_onehot(seq_str_list,symbol2idx):
#     num_seq=len(seq_str_list)
#     seq_len=max([len(s) for s in seq_str_list])
#     onehot_sz=max(symbol2idx.values())+1
#     onehot=np.zeros([num_seq,seq_len,onehot_sz])
#     for b in range(num_seq):
#         for t in range(len(seq_str_list[b])):
#             onehot[b,t,symbol2idx[seq_str_list[b][t]]]=1
#     tensor=torch.Tensor(onehot)
#     return tensor
#     return torch.autograd.Variable(tensor,requires_grad=False)
#

def torch_embed_inputs(seq_list,symbol2idx,onehot_sz=None):
    batch_sz=len(seq_list)
    max_seq=max([len(s) for s in seq_list])
    if onehot_sz is None:
        onehot_sz=max(symbol2idx.values())+1
    time_onehot_batch=np.zeros([max_seq,onehot_sz,batch_sz])
    for b in range(len(seq_list)):
        for t in range(len(seq_list[b])):
            time_onehot_batch[t,symbol2idx[seq_list[b][t]],b]=1
    return torch.Tensor(time_onehot_batch)


class Regexer(torch.nn.Module):
    def __init__(self,transition_matrix_dict,starting_states,accepting_states):
        super(Regexer, self).__init__()
        self.symbol2num={n:s for n,s in enumerate(sorted(transition_matrix_dict.keys()))}
        self.transitions=[]
        all_transitions=[]
        for n in sorted(self.symbol2num.keys()):
            symbol=self.symbol2num[n]
            self.transitions.append(torch.Tensor(transition_matrix_dict[symbol]))
            all_transitions.append(transition_matrix_dict[symbol])
        self.all_transitions=torch.Tensor(all_transitions)
        self.starting_states=torch.Tensor(starting_states)
        self.accepting_states=torch.Tensor(accepting_states)

    def forward(self,time_onehot_batch):
        seq_len,one_hot_sz,batch_sz=time_onehot_batch.shape
        current_states=self.starting_states.repeat(batch_sz).view(batch_sz,-1)
        goal_states = self.accepting_states.repeat(batch_sz).view(batch_sz, -1)
        #time_oh_batch=batched_time_onehot.transpose(0,1).transpose(1,2).contiguous()
        for t in range(seq_len):
            current_states=torch.matmul(current_states,self.all_transitions)
            print((current_states.size()))
            print((time_onehot_batch[t,:,:,None].size()))
            outgoing_states=current_states*time_onehot_batch[t,:,:,None]
            outgoing_states=outgoing_states.max(dim=0)[0]
            current_states=outgoing_states
        return (goal_states*current_states).sum(dim=1)


def plot_transition_matrices(symbol_seq,transition_matrix_dict,starting_states,accepting_states):
    plt.figure()
    trace=trace_simple_regex_match(symbol_seq,transition_matrix_dict,starting_states,accepting_states)
    ax = plt.subplot(1, len(transition_matrix_dict) + 1, 1)
    ax.imshow(trace, cmap="gray", interpolation='none', vmin=0, vmax=1)
    ax.set_title("Trace:")
    # Major ticks
    ax.set_xticks(np.arange(0, trace.shape[1], 1))
    ax.set_yticks(np.arange(0, trace.shape[0], 1))
    ax.set_xticks(np.arange(-.5, trace.shape[1], 1), minor=True);
    ax.set_yticks(np.arange(-.5, trace.shape[0], 1), minor=True);
    ax.grid(which='minor', color='g', linestyle='-', linewidth=1)

    idx=2
    for symbol,matrix in list(transition_matrix_dict.items()):
        ax = plt.subplot(1,len(transition_matrix_dict)+1,idx)
        ax.imshow(matrix,cmap="gray",interpolation='none', vmin=0, vmax=1)
        ax.set_title(symbol)
        # Major ticks
        ax.set_xticks(np.arange(0,  matrix.shape[1],1))
        ax.set_yticks(np.arange(0,  matrix.shape[0],1))
        ax.set_xticks(np.arange(-.5, matrix.shape[1], 1),minor=True);
        ax.set_yticks(np.arange(-.5, matrix.shape[0], 1),minor=True);
        ax.grid(which='minor', color='r', linestyle='-', linewidth=1)
        idx+=1


def enfa_mat_to_nfa_mat(symbol_transition_tensor,null_transition_mat):
    """Applies all null transitions to all symbol transition outputs.
    :param symbol_transition_tensor:
    :param null_transition_mat:
    :return: A tensor of the size of the symbol transition state
    """
    return np.dot(symbol_transition_tensor,null_transition_mat)

def apply_enfa_np(current_state,p_transition,symbol_transition_mat,null_transition_mat):
    """Applies a single iteration of an enfa and return the new state.

    :param current_state: A 1xStateCount Matrix containing the state at timestep t.
    :param p_transition: A probability over the active alphabet for observed transitions.
    :param symbol_transition_mat: A tensor of StateCount x StateCount x |Active Alphabet|
    :param null_transition_mat: A Matrix of StateCount x StateCount
    :return: A matrix of the same dimensionality as current_state with t+1.
    """


def run_torch(regex,strlist,symbol2idx={'a':0,'b':1,'c':2}):
    g=finit_automata.FAGraph()
    g.parse_re(regex,"abc")
    oh_seq=torch_embed_inputs(strlist,symbol2idx)
    transition_dict,starting_states,accepting_states= g.nfa_segment.create_transition_matrices()
    #print transition_dict
    reg=Regexer(transition_dict,starting_states,accepting_states)
    print((reg(oh_seq)))


if __name__=="__main__":
    import sys
    sys.argv+=["acab","a((ca)|(aa))b"]
    seq = sys.argv[1]
    regex=sys.argv[2]
    run_torch(regex,["a","aabb","bababa"])
    sys.exit()
    g=finit_automata.FAGraph()
    g.parse_re(regex,"abc")
    transition_dict,starting_states,accepting_states= g.nfa_segment.create_transition_matrices()
    import time
    import re
    t=time.time()
    for _ in range(1000):
        status= simple_regex_match(seq*100,transition_dict,starting_states,accepting_states)
    print(("R:{} | {} | Accepted:{} \nRun  100000 times in  {} msec".format(regex,seq,status,int((time.time()-t)*1000))))
    t=time.time()
    for _ in range(1000):
        status= re.match(regex,seq*100)
    print(("R:{} | {} | Accepted:{} \nRun  100000 times in  {} msec".format(regex,seq,status,int((time.time()-t)*1000))))
    #finit_automata.FAGraph.draw_graph(g.nfa_segment, re_str=regex)
    #plot_transition_matrices(seq,transition_dict,starting_states,accepting_states)
    #plt.show()
