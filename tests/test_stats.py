import pytest


def test_ci_brackets_the_mean():
    from statistics import mean

    from eval.stats import bootstrap_ci

    values = [0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.5, 0.7]
    lo, hi = bootstrap_ci(values, seed=0)
    assert lo <= mean(values) <= hi
    assert lo < hi  # spread data → non-degenerate interval


def test_ci_deterministic_for_fixed_seed():
    from eval.stats import bootstrap_ci

    values = [0.1, 0.9, 0.4, 0.6, 0.5]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    # a different seed is allowed to differ (not asserted — could collide)


def test_single_value_zero_width():
    from eval.stats import bootstrap_ci

    assert bootstrap_ci([0.5]) == (0.5, 0.5)


def test_identical_values_zero_width():
    from eval.stats import bootstrap_ci

    lo, hi = bootstrap_ci([0.7, 0.7, 0.7, 0.7], seed=0)
    assert lo == hi == 0.7


def test_empty_values_raise():
    from eval.stats import bootstrap_ci

    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_narrower_alpha_widens_interval():
    from eval.stats import bootstrap_ci

    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.7, 0.5, 0.9]
    lo95, hi95 = bootstrap_ci(values, alpha=0.05, seed=0)
    lo50, hi50 = bootstrap_ci(values, alpha=0.50, seed=0)
    # 95% interval must contain the 50% interval
    assert lo95 <= lo50 and hi50 <= hi95
