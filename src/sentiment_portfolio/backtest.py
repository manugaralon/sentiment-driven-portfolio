from dataclasses import dataclass

import numpy as np
import pandas as pd


def weekly_decision_dates(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading session of each calendar week (Friday, or Thursday if Friday is a holiday)."""
    last = pd.Series(sessions, index=sessions).groupby(sessions.to_period("W")).max()
    return pd.DatetimeIndex(last.to_numpy())


def schedule_execution(decisions: pd.DataFrame, sessions: pd.DatetimeIndex, lag: int = 1) -> pd.DataFrame:
    """Move weights decided at the close of session t to execution at the close of t + lag.

    This is the only place where decision time becomes execution time. Decisions whose
    execution would fall after the last session are dropped.
    """
    pos = sessions.get_indexer(decisions.index)
    if (pos < 0).any():
        raise ValueError("every decision date must be a trading session")
    exec_pos = pos + lag
    keep = exec_pos < len(sessions)
    executions = decisions.iloc[keep].copy()
    executions.index = sessions[exec_pos[keep]]
    return executions


@dataclass
class BacktestResult:
    returns: pd.Series  # net daily portfolio return
    weights: pd.DataFrame  # holdings at each close, after any rebalancing
    turnover: pd.Series  # sum |w_target - w_drifted| on execution days (buys + sells)
    costs: pd.Series  # cost as a fraction of portfolio value


def run_backtest(returns: pd.DataFrame, targets: pd.DataFrame, cost_bps: float) -> BacktestResult:
    """Simulate a long-only, fully invested portfolio that trades to `targets` at the close
    of each execution date.

    `targets` is indexed by execution sessions and holds the weights to own from that close on.
    The portfolio is formed at the close of the first execution date without cost (the same for
    every strategy), so results start the next session. The return of day d is earned on the
    weights held after the close of d-1; between rebalances weights drift with prices.
    """
    unknown = targets.index.difference(returns.index)
    if len(unknown):
        raise ValueError(f"execution dates not in returns index: {list(unknown[:3])}")
    targets = targets.reindex(columns=returns.columns, fill_value=0.0)
    if not np.allclose(targets.sum(axis=1), 1.0):
        raise ValueError("target weights must sum to 1 on every execution date")
    days = returns.index[returns.index > targets.index[0]]
    r = returns.loc[days].to_numpy()
    target_rows = {day: targets.loc[day].to_numpy() for day in targets.index[1:]}
    c = cost_bps / 1e4

    w = targets.iloc[0].to_numpy()
    net = np.zeros(len(days))
    held = np.zeros((len(days), returns.shape[1]))
    turnover = np.zeros(len(days))
    for i, day in enumerate(days):
        gross = w @ r[i]
        grown = w * (1 + r[i])
        w = grown / grown.sum()  # drift with prices
        cost = 0.0
        if day in target_rows:
            turnover[i] = np.abs(target_rows[day] - w).sum()
            cost = c * turnover[i]
            w = target_rows[day]
        net[i] = (1 + gross) * (1 - cost) - 1
        held[i] = w

    return BacktestResult(
        returns=pd.Series(net, index=days, name="return"),
        weights=pd.DataFrame(held, index=days, columns=returns.columns),
        turnover=pd.Series(turnover, index=days, name="turnover"),
        costs=pd.Series(turnover * c, index=days, name="cost"),
    )
