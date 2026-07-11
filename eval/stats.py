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
