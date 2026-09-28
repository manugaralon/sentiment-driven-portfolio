import numpy as np
import pandas as pd


def forward_returns(returns: pd.DataFrame, h: int, lag: int = 1) -> pd.DataFrame:
    """Return from the close of t+lag (when a decision taken at t executes) to the close of
    t+lag+h, indexed by t. NaN where that window runs past the data."""
    growth = (1 + returns).cumprod()
    return growth.shift(-(lag + h)) / growth.shift(-lag) - 1


def past_returns(returns: pd.DataFrame, h: int) -> pd.DataFrame:
    """Return from the close of t-h to the close of t, indexed by t."""
    growth = (1 + returns).cumprod()
    return growth / growth.shift(h) - 1


def cross_sectional_ic(signal: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Spearman correlation across tickers on each date (ties get average ranks). NaN on dates
    where either side is constant or the target is missing."""
    signal, target = signal.align(target, join="inner")
    ok = target.notna().all(axis=1)
    a = signal[ok].rank(axis=1)
    b = target[ok].rank(axis=1)
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    ic = (a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    return ic.replace([np.inf, -np.inf], np.nan).dropna().rename("ic")


def newey_west_mean(x: pd.Series, lags: int) -> dict[str, float]:
    """Mean of a series with a standard error robust to autocorrelation up to `lags` (Bartlett
    weights). Needed because overlapping windows make consecutive daily ICs correlated."""
    x = x.dropna().to_numpy()
    n, mean = len(x), x.mean()
    e = x - mean
    variance = e @ e / n
    for k in range(1, lags + 1):
        variance += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(variance / n)
    return {
        "mean": mean,
        "se": se,
        "t": mean / se,
        "lo95": mean - 1.96 * se,
        "hi95": mean + 1.96 * se,
        "n": n,
    }


def sharpe_diff_bootstrap(
    a: pd.Series, b: pd.Series, rf: pd.Series, block: int = 20, reps: int = 2000, seed: int = 0
) -> np.ndarray:
    """Moving-block bootstrap of the annualized Sharpe difference a - b. Days are resampled in
    blocks and in pairs (same days for a and b), which keeps both the autocorrelation and the
    ~0.9+ correlation between the two strategies; that pairing is what makes the test sharp."""
    excess = np.column_stack([a - rf.loc[a.index], b - rf.loc[a.index]])
    n = len(excess)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(reps, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    sample = excess[idx]  # reps x n x 2
    sharpe = sample.mean(axis=1) / sample.std(axis=1, ddof=1) * np.sqrt(252)
    return sharpe[:, 0] - sharpe[:, 1]


def permute_across_tickers(signal: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Placebo: on each date, the same signal values handed to tickers at random. Keeps the
    signal's distribution and timing, destroys which company it belongs to."""
    values = rng.permuted(signal.to_numpy(), axis=1)
    return pd.DataFrame(values, index=signal.index, columns=signal.columns)
