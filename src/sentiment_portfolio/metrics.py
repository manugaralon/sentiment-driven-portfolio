import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    """Geometric (compound) annual growth rate."""
    return float((1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: pd.Series) -> float:
    """Annualized Sharpe of daily returns in excess of the daily risk-free rate."""
    excess = returns - rf.loc[returns.index]
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss of the equity curve (a negative number). Starts at 1.0."""
    equity = np.concatenate([[1.0], (1 + returns).cumprod().to_numpy()])
    return float((equity / np.maximum.accumulate(equity) - 1).min())


def annual_turnover(turnover: pd.Series) -> float:
    """One-way turnover per year: half the dollars traded (buys + sells), annualized."""
    years = len(turnover) / TRADING_DAYS
    return float(0.5 * turnover.sum() / years)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    active = returns - benchmark.loc[returns.index]
    return float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))


def summary(returns: pd.Series, turnover: pd.Series, rf: pd.Series) -> dict[str, float]:
    return {
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, rf),
        "max_drawdown": max_drawdown(returns),
        "ann_turnover": annual_turnover(turnover),
    }
