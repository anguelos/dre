import numpy as np
import sys

re_str="(a?a)|b";
starting_states=np.array([[1,0,0]], dtype="float64")
ending_states=np.array([[0,0,1]], dtype="float64")
#current_states=np.array([[1.,.1,.05]]);
null_trans=np.array([[1,0,0],[0,1,1],[0,0,1]]);
symbol_trans=np.array([[[0,1,0],[0,0,1],[0,0,0]],[[0,0,1],[0,0,0],[0,0,0]]]);
emb={"a":[0,1],"b":[1,0]};
#emb_str=lambda x:np.array([emb[ch] for ch in x],dtype="float32")*.9+.1*np.random.rand(len(x),2)
emb_str=lambda x:np.array([emb[ch] for ch in x],dtype="float32")+((0.01*np.arange(len(x)))[:,None])
print(emb_str("aaaa").shape)
emb2symbolidx=np.array([[0.,1.],[1.,0.]])
get_p_transition=lambda x:np.dot(emb_str(x),emb2symbolidx)

def iterate_state(current_state,p_transition,symbol_transition_mat,null_transition_mat):
    #print (p_transition[:,:,None]).shape,(np.dot(current_state, symbol_transition_mat)).shape
    combined=np.dot(symbol_transition_mat,null_transition_mat)
    all_outgoing_states=np.dot(current_state, symbol_transition_mat)
    null_transited_states =np.dot(all_outgoing_states,
                                   null_transition_mat)#.max(axis=1)
    print(np.dot(current_state,combined)==null_transited_states)
    print(combined)
    symbol_transited_states=(null_transited_states * p_transition[:,:,None]).max(axis=1)

    print()
    print('Tr :\t',str(p_transition[:,:,None]).replace("\n","\n\t"))
    print("Cur:\t",str(current_state).replace("\n","\n\t"))
    print("Out:\t",str(all_outgoing_states).replace("\n","\n\t"))
    print("E  :\t",str(null_transited_states).replace("\n","\n\t"))
    print("F  :\t",str((null_transited_states * p_transition[:, :, None])).replace("\n","\n\t"))
    print("Res:\t",str(symbol_transited_states).replace("\n","\n\t"))
    return symbol_transited_states

def estimate_seq(symbol_seq):
    current_state=starting_states.copy()
    p_transitions = get_p_transition(symbol_seq)
    for n in range(len(symbol_seq)):
        p_transition=p_transitions[n:n+1,:]
        #symbol=symbol_seq[n]
        current_state=iterate_state(current_state,p_transition,symbol_trans,null_trans)
        print('|'.join(symbol_seq[:n+1]),(current_state*ending_states).max())

sys.argv.append("baaabbbaaabb")

estimate_seq(sys.argv[1])

sys.exit()

def run_re(token_list,embeding_dict,starting_states,ending_states,symbol_transition_mat,null_transition_mat,symbols_mat):
    current_state=np.dot(starting_states,null_transition_mat)
    for token_str in token_list:
        p_transitions=np.dot(embeding_dict[token_str],symbols_mat)
        symbol_transited_states=np.tensordot(current_state, symbol_transition_mat, (1, 0))[0,:,:].T
        null_transited_states=np.tensordot(symbol_transited_states, null_transition_mat, (1, 0))
    return np.any(ending_states*current_state)

matching_list1=["A","B","B","C","D","D","D","A"]
matching_list2=["D"]
matching_list3=["B","C","A"]
failing_list1=[""]
failing_list1=["A","B","B","C","C","D","D","A"]
failing_list2=["D","A"]

embedding_dict={"A":np.array([1,0,0,0,0]),
                "B":np.array([0,1,0,0,0]),
                "C":np.array([0,0,1,0,0]),
                "D":np.array([0,0,0,1,0]),
                "E":np.array([0,0,0,0,1])}

starting_states=[1,0,0,0,0]
ending_states=[0,0,0,0,1]

symbol_transition=[
    # A
                   [[0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]],
    # B
                   [[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],
    # C
                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],
    # D
                   [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]]

null_transited_states=[[0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]
