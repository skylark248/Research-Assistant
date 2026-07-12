"""Bootstrap confidence intervals — stdlib only, no numpy/scipy.

Percentile method: resample the per-row values with replacement, take each
resample's mean, and read the interval straight off the sorted means. Good
enough for eval-report error bars; not a substitute for a real power
analysis.
"""

import random
from statistics import mean


def bootstrap_ci(values: list[float], n_resamples: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """(1 - alpha) CI of the mean. Deterministic for a given seed.

    len(values) < 2 carries no spread information → zero-width (mean, mean).
    An empty list has no mean at all → ValueError.
    """
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    if len(values) < 2:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values)))
                   for _ in range(n_resamples))
    lo_i = int((alpha / 2) * n_resamples)
    hi_i = int((1 - alpha / 2) * n_resamples) - 1
    return (means[lo_i], means[hi_i])


def weighted_kappa(a: list[int], b: list[int], n_categories: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal scores 1..n_categories.

    Weights disagreements by squared distance, so a 4-vs-5 near-miss costs
    far less than a 1-vs-5 blunder — the standard choice for rubric scales.
    Zero expected disagreement (both raters constant and equal) → 1.0 by
    convention.
    """
    if len(a) != len(b):
        raise ValueError("rating lists must have the same length")
    if len(a) < 2:
        raise ValueError("weighted_kappa needs at least two rating pairs")
    n = n_categories
    observed = [[0.0] * n for _ in range(n)]
    for x, y in zip(a, b):
        observed[x - 1][y - 1] += 1
    total = len(a)
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(n)) for j in range(n)]
    disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) / (n - 1)) ** 2
            disagreement += weight * observed[i][j]
            expected_disagreement += weight * hist_a[i] * hist_b[j] / total
    if expected_disagreement == 0:
        return 1.0
    return 1.0 - disagreement / expected_disagreement
