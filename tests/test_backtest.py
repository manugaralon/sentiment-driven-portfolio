import numpy as np
import pandas as pd
import pytest

from sentiment_portfolio.backtest import run_backtest, schedule_execution, weekly_decision_dates
from sentiment_portfolio.strategies import equal_weight

DAYS = pd.bdate_range("2020-01-06", periods=4)  # Mon..Thu


def frame(rows, index=DAYS, columns=("A", "B")):
    return pd.DataFrame(rows, index=index[: len(rows)], columns=list(columns), dtype=float)


def test_equal_weight_two_assets_matches_hand_calculation():
    returns = frame([[0.0, 0.0], [0.1, 0.0], [0.0, 0.2], [0.1, -0.1]])
    targets = equal_weight(DAYS[[0, 2]], ["A", "B"])
    result = run_backtest(returns, targets, cost_bps=0)
    # formed at d0 close; d1 +10% on half; d2 +20% on B's drifted weight 0.5/1.05; d3 +10%/-10% cancel
    expected = [0.05, 0.2 * 0.5 / 1.05, 0.0]
    np.testing.assert_allclose(result.returns.to_numpy(), expected, atol=1e-12)


def test_weights_drift_between_rebalances():
    returns = frame([[0.0, 0.0], [0.1, 0.0]])
    result = run_backtest(returns, equal_weight(DAYS[[0]], ["A", "B"]), cost_bps=0)
    np.testing.assert_allclose(result.weights.loc[DAYS[1]].to_numpy(), [0.55 / 1.05, 0.5 / 1.05])


def test_cost_equals_c_times_turnover():
    returns = frame([[0.0, 0.0]] * 3)
    targets = frame([[0.5, 0.5], [0.8, 0.2]], index=DAYS[[0, 2]])
    result = run_backtest(returns, targets, cost_bps=10)
    assert result.turnover.loc[DAYS[1]] == 0.0  # formed at d0 without cost, no trade on d1
    assert result.turnover.loc[DAYS[2]] == pytest.approx(0.6)
    assert result.costs.loc[DAYS[2]] == pytest.approx(0.0006)
    assert result.returns.loc[DAYS[2]] == pytest.approx(-0.0006)


def test_trade_at_close_does_not_earn_that_days_return():
    # Hold B, switch to A at the close of d1: A's +50% on d1 must not be earned.
    returns = frame([[0.0, 0.0], [0.5, 0.0], [0.1, 0.0]])
    targets = frame([[0.0, 1.0], [1.0, 0.0]], index=DAYS[[0, 1]])
    result = run_backtest(returns, targets, cost_bps=0)
    np.testing.assert_allclose(result.returns.to_numpy(), [0.0, 0.1], atol=1e-12)


def test_targets_must_sum_to_one():
    returns = frame([[0.0, 0.0]] * 2)
    targets = frame([[0.5, 0.4]], index=DAYS[[0]])
    with pytest.raises(ValueError):
        run_backtest(returns, targets, cost_bps=0)


def test_unknown_execution_date_raises():
    returns = frame([[0.0, 0.0]])
    targets = frame([[0.5, 0.5]], index=pd.DatetimeIndex(["2020-01-11"]))  # a Saturday
    with pytest.raises(ValueError):
        run_backtest(returns, targets, cost_bps=0)


def test_schedule_execution_moves_to_next_session_and_drops_last():
    sessions = pd.bdate_range("2020-01-06", periods=5)  # Mon..Fri
    decisions = frame([[0.5, 0.5], [0.5, 0.5]], index=sessions[[2, 4]])  # Wed, Fri
    executions = schedule_execution(decisions, sessions, lag=1)
    assert list(executions.index) == [sessions[3]]  # Thu; Fri has no next session


def test_schedule_execution_rejects_non_session_dates():
    sessions = pd.bdate_range("2020-01-06", periods=5)
    decisions = frame([[0.5, 0.5]], index=pd.DatetimeIndex(["2020-01-11"]))
    with pytest.raises(ValueError):
        schedule_execution(decisions, sessions)


def test_weekly_decision_dates_handle_friday_holiday():
    # Good Friday 2023-04-07: the NYSE was closed, so that week's decision is on Thursday.
    sessions = pd.DatetimeIndex(
        list(pd.bdate_range("2023-04-03", "2023-04-06")) + list(pd.bdate_range("2023-04-10", "2023-04-14"))
    )
    assert list(weekly_decision_dates(sessions)) == [pd.Timestamp("2023-04-06"), pd.Timestamp("2023-04-14")]
