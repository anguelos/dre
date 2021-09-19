import torch
from numpy import argmax


def collate_fn(batch):
    """
       data: is a list of tuples with (index, label, tensors, length)
             where 'tensor' is a tensor of arbitrary shape
             and label/length are scalars
    """
    # print(batch)
    tensors = [k["encoded"] for k in batch]
    lengths = [k["length"] for k in batch]

    # add 0 padding at the end of tensors
    for i in range(len(batch)):

        zeros_tensor = torch.zeros((max(lengths) - lengths[i]), 37)

        # print(zeros_tensor.size())
        tensors[i] = torch.cat((tensors[i], zeros_tensor), 0)
        print(tensors[i].size())

    return tensors, lengths


def one_hot(data):
    # define universe of possible input values
    alphabet = 'abcdefghijklmnopqrstuvwxyz 0123456789'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]
    # print(integer_encoded)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    # invert encoding
    # inverted = int_to_char[argmax(onehot_encoded[0])]
    # print(inverted)
    return onehot_encoded
