import numpy as np
import pandas as pd
import pytest

from sentiment_portfolio.metrics import (
    annual_turnover,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)

INDEX = pd.bdate_range("2020-01-01", periods=252)


def series(values):
    return pd.Series(values, index=INDEX[: len(values)], dtype=float)


def test_constant_returns():
    r = series([0.001] * 252)
    assert annualized_return(r) == pytest.approx(1.001**252 - 1)
    assert annualized_volatility(r) == pytest.approx(0.0, abs=1e-12)
    assert max_drawdown(r) == 0.0


def test_sharpe_known_value():
    # Alternating 2% / 0%: mean 1%, sample std 1% * sqrt(252/251) -> Sharpe = sqrt(251)
    r = series([0.02, 0.0] * 126)
    rf = series([0.0] * 252)
    assert sharpe_ratio(r, rf) == pytest.approx(np.sqrt(251))


def test_sharpe_subtracts_risk_free():
    r = series([0.02, 0.0] * 126)
    rf = series([0.01] * 252)  # excess returns alternate +1% / -1% -> mean 0
    assert sharpe_ratio(r, rf) == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown():
    assert max_drawdown(series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert max_drawdown(series([-0.1, 0.05])) == pytest.approx(-0.1)  # peak is the initial 1.0


def test_annual_turnover_one_way():
    t = series([0.0] * 252)
    t.iloc[[50, 100, 150, 200]] = 0.2
    assert annual_turnover(t) == pytest.approx(0.5 * 0.8)
