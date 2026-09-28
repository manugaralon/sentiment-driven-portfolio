import numpy as np
import pandas as pd
import pytest

from sentiment_portfolio.alignment import session_closes
from sentiment_portfolio.signals import build_signal, cross_sectional_z, daily_panel, raw_sentiment
from sentiment_portfolio.stats import cross_sectional_ic, forward_returns, newey_west_mean, past_returns

TICKERS = ["A", "B", "C", "D"]
CLOSES = session_closes("2019-01-01", "2019-12-31")


def random_rows(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2019-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "updated_at": start + pd.to_timedelta(rng.integers(0, 364 * 24 * 3600, n), unit="s"),
            "ticker": rng.choice(TICKERS, n),
            "score": rng.uniform(-1, 1, n),
        }
    )


@pytest.mark.parametrize("n, surprise", [(5, False), (20, False), (5, True), (20, True)])
def test_signal_up_to_t_ignores_news_after_t(n, surprise):
    """Truncation invariance: dropping every article available after the close of T must not
    change the signal on any session up to T. The generic look-ahead test."""
    rows = random_rows()
    kwargs = {"n": n, "surprise": surprise, "min_history": 20}
    for cut in ["2019-03-15", "2019-07-03", "2019-11-29"]:  # normal day, pre-holiday, half day
        full = build_signal(rows, CLOSES, TICKERS, **kwargs)
        truncated = build_signal(rows[rows["updated_at"] < CLOSES[cut]], CLOSES, TICKERS, **kwargs)
        pd.testing.assert_frame_equal(full.loc[:cut], truncated.loc[:cut])
        assert not full.loc[cut:].iloc[1:].equals(truncated.loc[cut:].iloc[1:])  # the cut did bite


def test_panel_has_zeros_not_gaps_on_sessions_without_news():
    # A ticker without news on a session where others have some must count 0, not NaN:
    # a NaN would break the cumulative baseline of the surprise variant.
    rows = pd.DataFrame(
        {
            "updated_at": pd.to_datetime(["2019-03-04 15:00", "2019-03-05 15:00"], utc=True),
            "ticker": ["A", "B"],
            "score": [0.5, -0.5],
        }
    )
    score_sum, count = daily_panel(rows, CLOSES, TICKERS)
    assert count.notna().all().all() and score_sum.notna().all().all()
    assert count.loc["2019-03-04":"2019-03-05"].to_dict("list") == {
        "A": [1, 0],
        "B": [0, 1],
        "C": [0, 0],
        "D": [0, 0],
    }


def test_window_mean_weights_articles_not_days():
    sessions = pd.date_range("2020-01-01", periods=3)
    score_sum = pd.DataFrame({"A": [3.0, 0.0, -1.0]}, index=sessions)  # 3 articles of +1, then 1 of -1
    count = pd.DataFrame({"A": [3, 0, 1]}, index=sessions)
    raw = raw_sentiment(score_sum, count, n=3)
    assert raw["A"].tolist() == pytest.approx([1.0, 1.0, 0.5])


def test_surprise_subtracts_the_mean_before_the_window():
    sessions = pd.date_range("2020-01-01", periods=4)
    score_sum = pd.DataFrame({"A": [1.0, 1.0, 1.0, -0.5]}, index=sessions)  # always +1, then -0.5
    count = pd.DataFrame({"A": [1, 1, 1, 1]}, index=sessions)
    surprise = raw_sentiment(score_sum, count, n=1, surprise=True, min_history=2)
    # day 3: history {1, 1}, window {1} -> 0. day 4: history {1, 1, 1}, window {-0.5} -> -1.5.
    assert surprise["A"].isna().tolist() == [True, True, False, False]
    assert surprise["A"].iloc[2:].tolist() == pytest.approx([0.0, -1.5])


def test_z_is_zero_for_tickers_without_news_and_on_thin_days():
    raw = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [2.0, np.nan], "C": [3.0, np.nan], "D": [np.nan, 5.0]},
        index=pd.date_range("2020-01-01", periods=2),
    )
    z = cross_sectional_z(raw, min_names=3)
    assert z.iloc[0].tolist() == pytest.approx([-1.0, 0.0, 1.0, 0.0])
    assert z.iloc[1].tolist() == [0.0, 0.0, 0.0, 0.0]  # only 2 tickers with news


def test_forward_returns_start_at_execution():
    dates = pd.date_range("2020-01-01", periods=5)
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.20, 0.30, 0.40]}, index=dates)
    fwd = forward_returns(returns, h=2, lag=1)
    # decided at day 0, executed at close of day 1, held over days 2 and 3.
    assert fwd["A"].iloc[0] == pytest.approx(1.2 * 1.3 - 1)
    assert fwd["A"].iloc[2:].isna().all()
    assert past_returns(returns, h=2)["A"].iloc[2] == pytest.approx(1.1 * 1.2 - 1)


def test_ic_is_rank_correlation_and_skips_constant_days():
    dates = pd.date_range("2020-01-01", periods=2)
    signal = pd.DataFrame({"A": [1, 0], "B": [2, 0], "C": [3, 0]}, index=dates, dtype=float)
    target = pd.DataFrame({"A": [0.1, 1], "B": [0.5, 2], "C": [0.2, 3]}, index=dates)
    ic = cross_sectional_ic(signal, target)
    assert ic.index.tolist() == [dates[0]]
    assert ic.iloc[0] == pytest.approx(0.5)


def test_newey_west_matches_iid_se_without_lags_and_grows_with_autocorrelation():
    x = pd.Series(np.random.default_rng(1).normal(size=500))
    assert newey_west_mean(x, lags=0)["se"] == pytest.approx(x.std(ddof=0) / np.sqrt(500))
    smooth = x.rolling(10).mean()  # strongly autocorrelated
    assert newey_west_mean(smooth, lags=10)["se"] > 2 * newey_west_mean(smooth, lags=0)["se"]
