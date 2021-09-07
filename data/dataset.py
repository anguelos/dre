import torch


class MultilineStringDataset:
    """ Args:
            filename (string): Path to the txt file.
    """

    def __init__(self, filename):
        self.samples = []

        # Should take a multi-line string with many text lines
        f = open(filename, "r")
        # Separate each new lines (\n) and generate a sequence of text
        lowered = f.read().lower()
        # Get 2D pytorch tensors for each lines
        stripped = "".join(c for c in lowered if c not in ('!', '.', ':', '-', ','))
        samples_to_process = stripped.splitlines()
        f.close()

        for idx, sample in enumerate(samples_to_process):
            encoded = list(sample.encode('ascii'))
            tensor = torch.tensor(encoded)
            sample = {'index': idx, 'label': sample, 'encoded': tensor, 'length': len(encoded)}
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
