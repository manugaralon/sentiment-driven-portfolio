"""Baselines on the development period (train + validation): weekly equal-weight vs SPY buy-and-hold.

The test period is cut off before anything is computed.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from sentiment_portfolio.backtest import run_backtest, schedule_execution, weekly_decision_dates
from sentiment_portfolio.config import load_config, project_path
from sentiment_portfolio.metrics import summary
from sentiment_portfolio.prices import market_data
from sentiment_portfolio.strategies import equal_weight

COLORS = {"Equal-weight": "#2a78d6", "SPY buy-and-hold": "#eb6834"}
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def metrics_table(results: dict, periods: dict, rf: pd.Series) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        for period, (start, end) in periods.items():
            rows[(name, period)] = summary(res.returns.loc[start:end], res.turnover.loc[start:end], rf)
    return pd.DataFrame(rows).T


def plot(results: dict, val_start: str, costs_bps: float, path) -> None:
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, facecolor=SURFACE
    )
    for name, res in results.items():
        equity = (1 + res.returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        ax_eq.plot(equity.index, equity, color=COLORS[name], lw=1.5, label=name)
        ax_eq.annotate(
            f"{name}  {equity.iloc[-1]:.2f}x",
            (equity.index[-1], equity.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
        ax_dd.plot(drawdown.index, drawdown * 100, color=COLORS[name], lw=1.5, label=name)

    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 1.5, 2, 3, 5, 7]))
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
    ax_eq.yaxis.set_minor_formatter(NullFormatter())
    ax_eq.set_ylabel("Growth of $1 (log scale)", color=INK_MUTED)
    ax_eq.set_title(
        f"Baselines on train + validation, net of {costs_bps:g} bps costs", loc="left", color=INK, fontsize=12
    )
    ax_eq.legend(loc="upper left", frameon=False, labelcolor=INK)
    ax_dd.set_ylabel("Drawdown (%)", color=INK_MUTED)
    for ax in (ax_eq, ax_dd):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color="#e4e3df", lw=0.6)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.axvline(pd.Timestamp(val_start), color=INK_MUTED, lw=0.8, ls="--")
    ax_eq.text(
        pd.Timestamp(val_start),
        1.01,
        " validation →",
        transform=ax_eq.get_xaxis_transform(),
        color=INK_MUTED,
        fontsize=9,
    )
    ax_eq.text(
        pd.Timestamp(val_start),
        1.01,
        "← train ",
        transform=ax_eq.get_xaxis_transform(),
        color=INK_MUTED,
        fontsize=9,
        ha="right",
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    cfg = load_config()
    assets, bench = list(cfg["universe"]), cfg["benchmark"]
    dates, bt = cfg["dates"], cfg["backtest"]

    close = pd.read_parquet(project_path(cfg["paths"]["close"]))
    returns, rf = market_data(close, assets, bench, cfg["risk_free"])
    returns = returns.loc[: dates["validation"][1]]  # never touch the test period here
    sessions = returns.index

    decisions = weekly_decision_dates(sessions[sessions >= dates["train"][0]])
    ew_targets = schedule_execution(equal_weight(decisions, assets), sessions, bt["execution_lag_sessions"])
    spy_targets = pd.DataFrame({bench: [1.0]}, index=ew_targets.index[:1])

    results = {
        "Equal-weight": run_backtest(returns[assets], ew_targets, bt["cost_bps"]),
        "SPY buy-and-hold": run_backtest(returns[[bench]], spy_targets, bt["cost_bps"]),
    }
    periods = {"train": dates["train"], "validation": dates["validation"]}
    formed = ew_targets.index[0].date()
    print(f"Metrics net of {bt['cost_bps']} bps (portfolio formed at the close of {formed}):")
    print(metrics_table(results, periods, rf).to_string(float_format=lambda x: f"{x:8.3f}"))

    print("\nEqual-weight cost sensitivity (train + validation):")
    for bps in bt["cost_sensitivity_bps"]:
        res = run_backtest(returns[assets], ew_targets, bps)
        m = summary(res.returns, res.turnover, rf)
        print(f"  {bps:>3} bps: ann_return {m['ann_return']:.4f}  sharpe {m['sharpe']:.3f}")

    out = project_path(cfg["paths"]["figures"]) / "baselines_trainval.png"
    plot(results, dates["validation"][0], bt["cost_bps"], out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
