import numpy as np
import pandas as pd


def equal_weight(decision_dates: pd.DatetimeIndex, assets: list[str]) -> pd.DataFrame:
    """Target weights 1/n on every decision date."""
    return pd.DataFrame(1 / len(assets), index=decision_dates, columns=assets)


def project_to_bounds(x: pd.DataFrame, lo: float, hi: float, iterations: int = 100) -> pd.DataFrame:
    """Closest weights (Euclidean) to each row of `x` that sum to 1 and stay in [lo, hi]:
    clip(x + c, lo, hi), with the shift c found by bisection. Plain clip-and-renormalize can push
    weights back out of the bounds; this cannot."""
    values = x.to_numpy()
    c_lo = lo - values.max(axis=1, keepdims=True)  # every weight at lo: sum = n*lo <= 1
    c_hi = hi - values.min(axis=1, keepdims=True)  # every weight at hi: sum = n*hi >= 1
    for _ in range(iterations):
        c = (c_lo + c_hi) / 2
        too_big = np.clip(values + c, lo, hi).sum(axis=1, keepdims=True) > 1
        c_hi = np.where(too_big, c, c_hi)
        c_lo = np.where(too_big, c_lo, c)
    return pd.DataFrame(np.clip(values + (c_lo + c_hi) / 2, lo, hi), index=x.index, columns=x.columns)


def tilt(z: pd.DataFrame, lam: float, z_cap: float = 2.0, lo: float = 0.5, hi: float = 2.0) -> pd.DataFrame:
    """Rule (b): w_i = (1 + lam * z_i) / n with z capped at +-z_cap, projected onto weights in
    [lo/n, hi/n] that sum to 1. Long-only, fully invested; lam = 0 is equal weight."""
    n = z.shape[1]
    return project_to_bounds((1 + lam * z.clip(-z_cap, z_cap)) / n, lo / n, hi / n)
