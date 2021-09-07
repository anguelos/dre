import torch


def collate_fn(batch):
    """
       data: is a list of tuples with (index, label, tensors, length)
             where 'tensor' is a tensor of arbitrary shape
             and label/length are scalars
    """

    tensors = [k["encoded"] for k in batch]
    lengths = [k["length"] for k in batch]

    max_len = max(lengths)

    # add 0 padding at the end of tensors
    for i in range(len(batch)):
        length_tensor = torch.tensor(lengths[i])
        height_tensor = torch.tensor(1)
        tensors[i] = torch.cat([tensors[i], torch.zeros((max_len - lengths[i]))])
        tensors[i] = tensors[i].unsqueeze(0)  # add length to tensor
        tensors[i] = tensors[i].unsqueeze(0)  # add height to tensor
        # tensors[i] = torch.cat(length_tensor, tensors[i])
        print(tensors[i].size())

    return tensors, lengths
