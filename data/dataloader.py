# Should output TorchTensors with 3 dimensions (Width x Channel x Batch) (Might add height as 4th dimension based on
# Anguelos’s other code - Currently height is just 1 line)
# Another tensor should be necessary for encrypting batch information
# a collation of matching a regular expression.

# Proper regex matcher with a string. The transfer function should correctly predict the string.

from torch.utils.data import DataLoader
from util import collate_fn

class MultilineStringDataLoader:
    """ Args:
            dataset (string): pytorch dataset tuple.
            collate_fn: data collation.
            batch_size: batch size.
            shuffle: random shuffling.

    """
    def __init__(self, dataset, batch_size, shuffle):
        # noinspection PyTypeChecker
        dl = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)
        for x, y in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn):
            print('batch creation complete')

        print(dl.collate_fn)



