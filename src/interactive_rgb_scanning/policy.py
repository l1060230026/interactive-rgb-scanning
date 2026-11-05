"""Score-independent image regions and a shared randomized-score selector."""

import numpy as np


def image_regions(shape, grid=3):
    h, w = map(int, shape)
    regions = []
    for row in range(grid):
        for col in range(grid):
            y0, y1 = row * h // grid, (row + 1) * h // grid
            x0, x1 = col * w // grid, (col + 1) * w // grid
            regions.append(
                dict(
                    id=row * grid + col,
                    row=row,
                    col=col,
                    bounds=[x0, y0, x1, y1],
                    uv=[(x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2],
                )
            )
    return regions


def select_region(scores, tie_seed, permutation=None):
    """The caller may change scores, but never candidate geometry or execution."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Finite nonempty regional scores required")
    permutation = (
        np.arange(len(scores))
        if permutation is None
        else np.asarray(permutation, dtype=int)
    )
    if sorted(permutation.tolist()) != list(range(len(scores))):
        raise ValueError("Expected a permutation of all candidate indices")
    assigned = scores[permutation]
    ties = np.flatnonzero(assigned == assigned.max())
    chosen = int(np.random.default_rng(tie_seed).choice(ties))
    return dict(
        selected=chosen,
        raw_scores=scores.tolist(),
        assigned_scores=assigned.tolist(),
        permutation=permutation.tolist(),
        tie_seed=int(tie_seed),
    )


def score_assignment(scores, variant, scene_index, cycle, permutation_seed, tie_seed):
    permutation = np.arange(len(scores))
    entropy = [int(permutation_seed), int(scene_index), int(cycle)]
    if variant == "spatially_shuffled":
        permutation = np.random.default_rng(
            np.random.SeedSequence(entropy)
        ).permutation(len(scores))
    elif variant != "true_blue":
        raise ValueError(variant)
    result = select_region(scores, tie_seed + 1000 * scene_index + cycle, permutation)
    result.update(
        variant=variant, permutation_seed=int(permutation_seed), entropy=entropy
    )
    return result
