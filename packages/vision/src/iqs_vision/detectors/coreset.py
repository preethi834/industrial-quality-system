"""Greedy k-center coreset selection (PatchCore §3.2), memory-bandwidth friendly.

Same algorithm and random stream as anomalib's KCenterGreedy (sparse random projection
with eps = 0.9, then a random start, then repeatedly add the point farthest from the
selected set). The difference is the distance update: anomalib computes
‖f − c‖ with a fresh N×d temporary every iteration; here
‖f − c‖² = ‖f‖² − 2 f·c + ‖c‖² with precomputed ‖f‖², i.e. one matrix–vector product
and no large allocation per step. Arg-max of squared distances equals arg-max of
distances, so the selected points are the same up to floating-point ties.
"""

from __future__ import annotations

import torch


def kcenter_greedy_indices(embedding: torch.Tensor, sampling_ratio: float) -> torch.Tensor:
    """Indices of the coreset (uses the global torch RNG, like anomalib)."""
    from anomalib.models.components.dimensionality_reduction import SparseRandomProjection

    n = embedding.shape[0]
    k = int(n * sampling_ratio)
    if k <= 0:
        raise ValueError(f"coreset size {k} for n={n}, ratio={sampling_ratio}")
    projection = SparseRandomProjection(eps=0.9)
    projection.fit(embedding)
    feats = projection.transform(embedding).contiguous()
    sq_norms = (feats * feats).sum(dim=1)
    min_dist = torch.full((n,), float("inf"), dtype=feats.dtype)
    idx = int(torch.randint(high=n, size=(1,)).item())
    selected = torch.empty(k, dtype=torch.long)
    selected[0] = idx
    dist = torch.empty(n, dtype=feats.dtype)
    for i in range(1, k):
        torch.mv(feats, feats[idx], out=dist)
        dist.mul_(-2).add_(sq_norms).add_(sq_norms[idx])
        torch.minimum(min_dist, dist, out=min_dist)
        min_dist[idx] = 0.0
        idx = int(torch.argmax(min_dist).item())
        selected[i] = idx
    return selected


def kcenter_greedy(embedding: torch.Tensor, sampling_ratio: float) -> torch.Tensor:
    return embedding[kcenter_greedy_indices(embedding, sampling_ratio)]
