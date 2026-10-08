import torch


class AbstractDistance(object):
    """Distances between embedded tokens, and each token's own scale ("mass") in the same units.

    The mass sets how far a token may be from a query word and still be likely that word (see JointSoftmaxObserver).
    The base implementation is the L1 distance and the L1 norm.
    """
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (N, D), b: (M, D) -> (N, M)
        return torch.cdist(a.float(), b.float(), p=1)

    def mass(self, a: torch.Tensor) -> torch.Tensor:
        # a: (N, D) -> (N,)
        return a.float().abs().sum(dim=1)


class L1Distance(AbstractDistance):
    # PHOC: the L1 distance between embeddings, the L1 norm of a token's embedding as its mass.
    pass


class CosineDistance(AbstractDistance):
    # Semantic embeddings: 1 - cosine similarity; zero vectors (tokens without an embedding) are at distance 1 from
    # everything. The mass is 1, so a background threshold is a plain cosine distance.
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.float() / a.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
        b = b.float() / b.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
        return 1 - a @ b.T

    def mass(self, a: torch.Tensor) -> torch.Tensor:
        return torch.ones(a.size(0), device=a.device)


class LevenshteinDistance(AbstractDistance):
    # Edit distance between tokens embedded as zero-padded character codes (CharCodeEmbedder); the mass is a token's
    # length in characters. Computed for all (N, M) pairs at once, one DP row at a time; a's rows are chunked so that a
    # chunk's DP row holds at most chunk_elements cells.
    def __init__(self, chunk_elements: int = 1 << 24):
        self.chunk_elements = chunk_elements

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_lengths, b_lengths = (a > 0).sum(dim=1), (b > 0).sum(dim=1)
        columns = torch.arange(b.size(1) + 1, device=b.device, dtype=torch.float)          # (L+1,) DP column index j
        chunk = max(1, self.chunk_elements // max(1, b.size(0) * (b.size(1) + 1)))
        result = torch.empty(a.size(0), b.size(0), device=a.device)
        for begin in range(0, a.size(0), chunk):
            codes, lengths = a[begin:begin + chunk], a_lengths[begin:begin + chunk]
            # row i of the DP table for every pair: distance between a's first i characters and b's first j ones
            row = columns.expand(codes.size(0), b.size(0), -1).clone()                        # (n, M, L+1), i = 0
            distances = b_lengths[None, :].expand(codes.size(0), -1).float().clone()          # pairs with an empty a
            for i in range(1, int(lengths.max()) + 1 if lengths.numel() else 1):
                substitution = (codes[:, i - 1, None, None] != b[None, :, :]).float()         # (n, M, L)
                # deletion from a and substitution only depend on the previous row ...
                candidate = torch.empty_like(row)
                candidate[:, :, 0] = i
                candidate[:, :, 1:] = torch.minimum(row[:, :, 1:] + 1, row[:, :, :-1] + substitution)
                # ... insertions chain along the row: D[j] = min over k <= j of candidate[k] + (j - k)
                row = torch.cummin(candidate - columns, dim=2).values + columns
                done = lengths == i
                if done.any():
                    distances[done] = row[done].gather(2, b_lengths[None, :, None].expand(int(done.sum()), -1, 1)).squeeze(2)
            result[begin:begin + chunk] = distances
        return result

    def mass(self, a: torch.Tensor) -> torch.Tensor:
        return (a > 0).sum(dim=1).float()
