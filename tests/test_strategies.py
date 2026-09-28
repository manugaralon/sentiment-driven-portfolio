import numpy as np
import pandas as pd
import pytest

from sentiment_portfolio.stats import permute_across_tickers, sharpe_diff_bootstrap
from sentiment_portfolio.strategies import tilt

TICKERS = list("ABCDEFGHI")  # n = 9, like the universe


def random_z(rows: int = 200, scale: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(scale=scale, size=(rows, len(TICKERS))), columns=TICKERS)


@pytest.mark.parametrize("lam", [0.25, 0.5, 1.0, -1.0, 5.0])
def test_weights_sum_to_one_and_respect_bounds(lam):
    n = len(TICKERS)
    w = tilt(random_z(scale=3.0), lam)  # large z and lam: bounds bind often
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)
    assert (w >= 0.5 / n - 1e-12).all().all() and (w <= 2 / n + 1e-12).all().all()


def test_lambda_zero_is_exactly_equal_weight():
    w = tilt(random_z(), 0.0)
    np.testing.assert_allclose(w.to_numpy(), 1 / len(TICKERS), atol=1e-12)


def test_small_tilts_are_untouched_by_the_projection():
    z = pd.DataFrame([[1.0, -1.0] + [0.0] * 7], columns=TICKERS)  # sums to 1 and inside bounds
    w = tilt(z, 0.5)
    np.testing.assert_allclose(w.iloc[0, :3], [1.5 / 9, 0.5 / 9, 1 / 9], atol=1e-12)


def test_higher_signal_never_gets_less_weight():
    z, w = random_z(scale=2.0), tilt(random_z(scale=2.0), 1.0)
    for i in range(len(z)):
        order = np.argsort(z.iloc[i].to_numpy())
        assert np.all(np.diff(w.iloc[i].to_numpy()[order]) >= -1e-12)


def test_placebo_permutes_values_within_each_date():
    z = random_z(rows=5)
    placebo = permute_across_tickers(z, np.random.default_rng(1))
    np.testing.assert_allclose(np.sort(placebo.to_numpy(), axis=1), np.sort(z.to_numpy(), axis=1))
    assert not placebo.equals(z)


def test_bootstrap_is_centered_on_the_true_sharpe_difference():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2016-01-01", periods=1500)
    common = rng.normal(0.0004, 0.01, len(days))
    a = pd.Series(common + rng.normal(0.0002, 0.002, len(days)), index=days)  # b plus a small edge
    b = pd.Series(common, index=days)
    rf = pd.Series(0.0, index=days)
    diffs = sharpe_diff_bootstrap(a, b, rf, reps=500)
    point = (a.mean() / a.std() - b.mean() / b.std()) * np.sqrt(252)
    assert abs(np.median(diffs) - point) < 0.1
    assert np.percentile(diffs, 2.5) > 0  # a paired test sees a small but steady edge
