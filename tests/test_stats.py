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


def test_kappa_perfect_agreement_is_one():
    from eval.stats import weighted_kappa

    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_kappa_observed_equals_expected_is_zero():
    from eval.stats import weighted_kappa

    # observed disagreement exactly matches chance for these marginals
    assert weighted_kappa([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0)


def test_kappa_quadratic_weighting_distinguishes_near_from_far_miss():
    from eval.stats import weighted_kappa

    a = [1, 5, 1, 5]
    near = weighted_kappa(a, [2, 4, 2, 4])  # off by one each time
    far = weighted_kappa(a, [5, 1, 5, 1])   # maximally wrong each time
    assert near == pytest.approx(0.8)
    assert far == pytest.approx(-1.0)
    # unweighted agreement would score both identically (zero exact matches);
    # the quadratic weights are what separate them
    assert near > far


def test_kappa_constant_equal_raters_is_one_by_convention():
    from eval.stats import weighted_kappa

    # zero expected disagreement → denominator 0 → 1.0 by convention
    assert weighted_kappa([3, 3, 3], [3, 3, 3]) == pytest.approx(1.0)


def test_kappa_length_mismatch_raises():
    from eval.stats import weighted_kappa

    with pytest.raises(ValueError):
        weighted_kappa([1, 2], [1, 2, 3])


def test_kappa_too_few_pairs_raises():
    from eval.stats import weighted_kappa

    with pytest.raises(ValueError):
        weighted_kappa([3], [3])
