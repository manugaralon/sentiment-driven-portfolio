"""Baselines and the sentiment tilt on the development period (train + validation).

Baselines: weekly equal-weight (EW) vs SPY buy-and-hold. Sentiment tilt: the (N, lambda) grid is
chosen on train by information ratio vs EW, then confirmed on validation next to the controls
(placebo, momentum tilt, block bootstrap). The test period is cut off before anything is computed.

`--final` (phase 7, run once, after the `pre-test-freeze` tag): the frozen configuration from
config.yaml on the test period. Nothing is selected there; it only reports.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from sentiment_portfolio.backtest import run_backtest, schedule_execution, weekly_decision_dates
from sentiment_portfolio.config import load_config, project_path
from sentiment_portfolio.metrics import information_ratio, summary
from sentiment_portfolio.prices import market_data
from sentiment_portfolio.signals import cross_sectional_z
from sentiment_portfolio.stats import (
    cross_sectional_ic,
    forward_returns,
    newey_west_mean,
    past_returns,
    permute_across_tickers,
    sharpe_diff_bootstrap,
)
from sentiment_portfolio.strategies import equal_weight, tilt

PALETTE = ["#2a78d6", "#eb6834", "#1a9e5a", "#8a5cc2"]  # by position in the results dict
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def metrics_table(results: dict, periods: dict, rf: pd.Series) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        for period, (start, end) in periods.items():
            rows[(name, period)] = summary(res.returns.loc[start:end], res.turnover.loc[start:end], rf)
    return pd.DataFrame(rows).T


def plot(returns: dict[str, pd.Series], title: str, path, split: tuple[str, str, str] | None = None) -> None:
    """Equity (log) and drawdown. `split` = (date, left label, right label) draws a period cut."""
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, facecolor=SURFACE
    )
    for (name, r), color in zip(returns.items(), PALETTE, strict=False):
        equity = (1 + r).cumprod()
        drawdown = equity / equity.cummax() - 1
        ax_eq.plot(equity.index, equity, color=color, lw=1.5, label=name)
        ax_eq.annotate(
            f"{name}  {equity.iloc[-1]:.2f}x",
            (equity.index[-1], equity.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
        ax_dd.plot(drawdown.index, drawdown * 100, color=color, lw=1.5, label=name)

    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 1.5, 2, 3, 5, 7]))
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
    ax_eq.yaxis.set_minor_formatter(NullFormatter())
    ax_eq.set_ylabel("Growth of $1 (log scale)", color=INK_MUTED)
    ax_eq.set_title(title, loc="left", color=INK, fontsize=12)
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
    if split:
        at, left, right = pd.Timestamp(split[0]), split[1], split[2]
        for ax in (ax_eq, ax_dd):
            ax.axvline(at, color=INK_MUTED, lw=0.8, ls="--")
        kwargs = {"transform": ax_eq.get_xaxis_transform(), "color": INK_MUTED, "fontsize": 9}
        ax_eq.text(at, 1.01, f" {right} →", **kwargs)
        ax_eq.text(at, 1.01, f"← {left} ", ha="right", **kwargs)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


class TiltBacktest:
    """Weekly tilt backtests sharing the EW decision dates, execution lag and costs."""

    def __init__(self, returns: pd.DataFrame, decisions: pd.DatetimeIndex, cfg: dict):
        self.returns, self.decisions = returns, decisions
        self.lag = cfg["backtest"]["execution_lag_sessions"]
        st = cfg["strategy"]
        self.z_cap, (self.lo, self.hi) = st["z_cap"], st["weight_bounds"]

    def run(self, z: pd.DataFrame, lam: float, cost_bps: float):
        weights = tilt(z.loc[self.decisions], lam, self.z_cap, self.lo, self.hi)
        targets = schedule_execution(weights, self.returns.index, self.lag)
        return run_backtest(self.returns, targets, cost_bps)


def period_stats(res, ew, rf: pd.Series, periods: dict) -> dict:
    out = {}
    for period, (start, end) in periods.items():
        r = res.returns.loc[start:end]
        m = summary(r, res.turnover.loc[start:end], rf)
        out[(period, "sharpe")] = m["sharpe"]
        out[(period, "ir_vs_ew")] = information_ratio(r, ew.returns)
        out[(period, "turnover")] = m["ann_turnover"]
    return out


def grid_table(bt: TiltBacktest, signals: dict, lambdas: list, ew, rf, periods, cost_bps) -> pd.DataFrame:
    rows = {
        (n, lam): period_stats(bt.run(z, lam, cost_bps), ew, rf, periods)
        for n, z in signals.items()
        for lam in lambdas
    }
    table = pd.DataFrame(rows).T
    table.index.names = ["N", "lambda"]
    return table


def plot_grid(table: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    for ax, metric, title in zip(axes, ["sharpe", "ir_vs_ew"], ["Sharpe", "IR vs equal-weight"], strict=True):
        grid = table[("train", metric)].unstack("lambda")
        limit = np.abs(grid.to_numpy()).max()
        cmap, vmin = ("RdBu", -limit) if metric == "ir_vs_ew" else ("Blues", grid.to_numpy().min() - 0.1)
        image = ax.imshow(grid.to_numpy(), cmap=cmap, vmin=vmin, vmax=limit, aspect="auto")
        for (i, j), value in np.ndenumerate(grid.to_numpy()):
            dark = abs(image.norm(value) - 0.5) > 0.3 if metric == "ir_vs_ew" else image.norm(value) > 0.6
            color = "white" if dark else INK
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks(range(grid.shape[1]), [f"λ={x:g}" for x in grid.columns])
        ax.set_yticks(range(grid.shape[0]), [f"N={n}" for n in grid.index])
        ax.set_title(f"{title}, train, net of costs", loc="left", color=INK, fontsize=10)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def sentiment_study(cfg, returns, rf, decisions, ew_targets, ew, spy, periods) -> None:
    assets, st, sig = list(cfg["universe"]), cfg["strategy"], cfg["signal"]
    cost = cfg["backtest"]["cost_bps"]
    stored = pd.read_parquet(project_path(cfg["paths"]["processed"]) / "signals.parquet")
    signals = {n: stored[f"{st['variant']}_{n}"].unstack()[assets] for n in sig["windows"]}
    bt = TiltBacktest(returns[assets], decisions, cfg)
    fmt = {"float_format": lambda x: f"{x:7.3f}"}

    grid = grid_table(bt, signals, st["lambdas"], ew, rf, periods, cost)
    n, lam = grid[("train", "ir_vs_ew")].idxmax()
    print(
        f"\n== Sentiment tilt grid ({st['variant']}, net of {cost} bps); "
        f"chosen on train IR: N={n}, λ={lam} =="
    )
    print(grid.to_string(**fmt))
    fig = project_path(cfg["paths"]["figures"]) / "tilt_grid_train.png"
    plot_grid(grid, fig)
    print(f"saved {fig}")

    chosen = bt.run(signals[n], lam, cost)
    momentum_z = cross_sectional_z(past_returns(returns[assets], n))
    momentum = bt.run(momentum_z, lam, cost)
    rows = {
        "Equal-weight": ew,
        "SPY buy-and-hold": spy,
        f"Sentiment tilt N={n} λ={lam}": chosen,
        f"Momentum tilt {n}d λ={lam}": momentum,
    }
    table = metrics_table(rows, periods, rf)
    table["ir_vs_ew"] = [
        information_ratio(res.returns.loc[s:e], ew.returns) if res is not ew else np.nan
        for res in rows.values()
        for s, e in periods.values()
    ]
    print("\n== Chosen tilt vs baselines and momentum control (net) ==")
    print(table.to_string(**fmt))
    for period, (s, e) in periods.items():
        active = pd.DataFrame(
            {"sentiment": chosen.returns - ew.returns, "momentum": momentum.returns - ew.returns}
        )
        print(
            f"  corr of active returns (sentiment vs momentum), {period}: "
            f"{active.loc[s:e].corr().iloc[0, 1]:+.2f}"
        )

    print("\n== Cost sensitivity of the chosen tilt ==")
    for bps in cfg["backtest"]["cost_sensitivity_bps"]:
        stats = period_stats(
            bt.run(signals[n], lam, bps), run_backtest(returns[assets], ew_targets, bps), rf, periods
        )
        print(
            f"  {bps:>3} bps: "
            + "  ".join(
                f"{p} sharpe {stats[(p, 'sharpe')]:.3f} IR {stats[(p, 'ir_vs_ew')]:+.3f}" for p in periods
            )
        )

    print(f"\n== Placebo: signal permuted across tickers, {st['placebo_draws']} draws ==")
    rng = np.random.default_rng(0)
    actual = period_stats(chosen, ew, rf, periods)
    draws = pd.DataFrame(
        [
            period_stats(
                bt.run(permute_across_tickers(signals[n].loc[decisions], rng), lam, cost), ew, rf, periods
            )
            for _ in range(st["placebo_draws"])
        ]
    )
    for period in periods:
        ir = draws[(period, "ir_vs_ew")]
        p_value = (1 + (ir >= actual[(period, "ir_vs_ew")]).sum()) / (1 + len(ir))
        print(
            f"  {period}: actual IR {actual[(period, 'ir_vs_ew')]:+.3f} | "
            f"placebo IR median {ir.median():+.3f}, "
            f"90% range [{ir.quantile(0.05):+.3f}, {ir.quantile(0.95):+.3f}] | p = {p_value:.3f}"
        )

    print(f"\n== Block bootstrap of Sharpe(tilt) - Sharpe(EW), {st['bootstrap_block']}-day blocks ==")
    for period, (s, e) in periods.items():
        a, b = chosen.returns.loc[s:e], ew.returns.loc[s:e]
        diffs = sharpe_diff_bootstrap(a, b, rf, st["bootstrap_block"], st["bootstrap_reps"])
        point = actual[(period, "sharpe")] - summary(b, ew.turnover.loc[s:e], rf)["sharpe"]
        print(
            f"  {period}: ΔSharpe {point:+.3f}, 95% CI [{np.percentile(diffs, 2.5):+.3f}, "
            f"{np.percentile(diffs, 97.5):+.3f}], P(Δ <= 0) = {(diffs <= 0).mean():.2f}"
        )

    h = st["gate_ic_horizon"]
    train_returns = returns[assets].loc[: periods["train"][1]]
    ic = cross_sectional_ic(signals[n].loc[slice(*periods["train"])], forward_returns(train_returns, h))
    ic_stats = newey_west_mean(ic, lags=h + n)
    val = period_stats(chosen, ew, rf, periods)
    val_edge = val[("validation", "ir_vs_ew")] > 0
    print(f"\n== Phase-6 gate: train IC (h={h}) t > 2 AND tilt beats EW on validation, net ==")
    ic_pass = ic_stats["t"] > 2
    print(f"  train IC {ic_stats['mean']:+.4f} (t = {ic_stats['t']:+.2f}) -> {'pass' if ic_pass else 'FAIL'}")
    print(f"  validation IR vs EW {val[('validation', 'ir_vs_ew')]:+.3f} -> {'pass' if val_edge else 'FAIL'}")
    print(f"  verdict: {'OPEN phase 6' if ic_pass and val_edge else 'CLOSED: no PPO'}")

    print("\n== EXPLORATORY (not part of the gate): contrarian tilt, sign taken from the train IC ==")
    contra = grid_table(bt, signals, st["exploratory_lambdas"], ew, rf, periods, cost)
    print(contra.to_string(**fmt))
    cn, clam = contra[("train", "ir_vs_ew")].idxmax()
    print(
        f"  chosen on train: N={cn}, λ={clam} -> validation IR vs EW "
        f"{contra.loc[(cn, clam), ('validation', 'ir_vs_ew')]:+.3f}"
    )


def final_evaluation(cfg, returns, rf, decisions, ew_targets, ew, spy) -> None:
    """Phase 7: the frozen tilt and its controls on the test period. Reports only; selects nothing."""
    assets, st = list(cfg["universe"]), cfg["strategy"]
    n, lam = st["frozen"]["window"], st["frozen"]["lambda"]
    cost = cfg["backtest"]["cost_bps"]
    test = {"test": cfg["dates"]["test"]}
    s, e = cfg["dates"]["test"]
    stored = pd.read_parquet(project_path(cfg["paths"]["processed"]) / "signals.parquet")
    z = stored[f"{st['variant']}_{n}"].unstack()[assets]
    bt = TiltBacktest(returns[assets], decisions, cfg)
    fmt = {"float_format": lambda x: f"{x:7.3f}"}

    chosen = bt.run(z, lam, cost)
    momentum = bt.run(cross_sectional_z(past_returns(returns[assets], n)), lam, cost)
    rows = {
        "Equal-weight": ew,
        "SPY buy-and-hold": spy,
        f"Sentiment tilt N={n} λ={lam}": chosen,
        f"Momentum tilt {n}d λ={lam}": momentum,
    }
    table = metrics_table(rows, test, rf)
    table["ir_vs_ew"] = [
        information_ratio(res.returns.loc[s:e], ew.returns) if res is not ew else np.nan
        for res in rows.values()
    ]
    print(f"\n== FINAL: test {s} -> {e}, frozen config ({st['variant']}, N={n}, λ={lam}, {cost} bps) ==")
    print(table.to_string(**fmt))
    active = pd.DataFrame(
        {"sentiment": chosen.returns - ew.returns, "momentum": momentum.returns - ew.returns}
    )
    print(f"  corr of active returns (sentiment vs momentum): {active.loc[s:e].corr().iloc[0, 1]:+.2f}")

    paired = newey_west_mean(active["sentiment"].loc[s:e], lags=5)
    print("\n== Paired test on daily active return (tilt - EW), Newey-West lag 5 ==")
    print(f"  mean {paired['mean'] * 252:+.4f}/year, t = {paired['t']:+.2f}")

    a, b = chosen.returns.loc[s:e], ew.returns.loc[s:e]
    diffs = sharpe_diff_bootstrap(a, b, rf, st["bootstrap_block"], st["bootstrap_reps"])
    point = (
        summary(a, chosen.turnover.loc[s:e], rf)["sharpe"] - summary(b, ew.turnover.loc[s:e], rf)["sharpe"]
    )
    print(f"\n== Block bootstrap of Sharpe(tilt) - Sharpe(EW), {st['bootstrap_block']}-day blocks ==")
    print(
        f"  ΔSharpe {point:+.3f}, 95% CI [{np.percentile(diffs, 2.5):+.3f}, "
        f"{np.percentile(diffs, 97.5):+.3f}], P(Δ <= 0) = {(diffs <= 0).mean():.2f}"
    )

    rng = np.random.default_rng(0)
    actual = period_stats(chosen, ew, rf, test)[("test", "ir_vs_ew")]
    placebo = pd.Series(
        [
            period_stats(bt.run(permute_across_tickers(z.loc[decisions], rng), lam, cost), ew, rf, test)[
                ("test", "ir_vs_ew")
            ]
            for _ in range(st["placebo_draws"])
        ]
    )
    p_value = (1 + (placebo >= actual).sum()) / (1 + len(placebo))
    print(f"\n== Placebo, {st['placebo_draws']} draws ==")
    print(
        f"  actual IR {actual:+.3f} | placebo median {placebo.median():+.3f}, "
        f"90% range [{placebo.quantile(0.05):+.3f}, {placebo.quantile(0.95):+.3f}] | p = {p_value:.3f}"
    )

    print("\n== Cost sensitivity ==")
    for bps in cfg["backtest"]["cost_sensitivity_bps"]:
        stats = period_stats(bt.run(z, lam, bps), run_backtest(returns[assets], ew_targets, bps), rf, test)
        print(f"  {bps:>3} bps: sharpe {stats[('test', 'sharpe')]:.3f} IR {stats[('test', 'ir_vs_ew')]:+.3f}")

    h = st["gate_ic_horizon"]
    ic = newey_west_mean(cross_sectional_ic(z.loc[s:e], forward_returns(returns[assets], h)), lags=h + n)
    print(f"\n== Signal IC on test (h={h}, NW lag {h + n}) ==")
    print(f"  mean {ic['mean']:+.4f}, 95% CI [{ic['lo95']:+.4f}, {ic['hi95']:+.4f}], t = {ic['t']:+.2f}")

    in_test = {name: res.returns.loc[s:e] for name, res in rows.items()}
    out = project_path(cfg["paths"]["figures"]) / "final_test.png"
    plot(in_test, f"Test {s[:7]} → {e[:7]}, frozen configuration, net of {cost:g} bps", out)
    print(f"\nsaved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="phase 7: open the test period (run once)")
    final = parser.parse_args().final
    cfg = load_config()
    assets, bench = list(cfg["universe"]), cfg["benchmark"]
    dates, bt = cfg["dates"], cfg["backtest"]

    close = pd.read_parquet(project_path(cfg["paths"]["close"]))
    returns, rf = market_data(close, assets, bench, cfg["risk_free"])
    returns = returns.loc[: dates["test" if final else "validation"][1]]  # test only with --final
    sessions = returns.index

    decisions = weekly_decision_dates(sessions[sessions >= dates["train"][0]])
    ew_targets = schedule_execution(equal_weight(decisions, assets), sessions, bt["execution_lag_sessions"])
    spy_targets = pd.DataFrame({bench: [1.0]}, index=ew_targets.index[:1])

    results = {
        "Equal-weight": run_backtest(returns[assets], ew_targets, bt["cost_bps"]),
        "SPY buy-and-hold": run_backtest(returns[[bench]], spy_targets, bt["cost_bps"]),
    }
    if final:
        final_evaluation(
            cfg, returns, rf, decisions, ew_targets, results["Equal-weight"], results["SPY buy-and-hold"]
        )
        return
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
    title = f"Baselines on train + validation, net of {bt['cost_bps']:g} bps costs"
    plot(
        {name: res.returns for name, res in results.items()},
        title,
        out,
        split=(dates["validation"][0], "train", "validation"),
    )
    print(f"\nsaved {out}")

    sentiment_study(
        cfg, returns, rf, decisions, ew_targets, results["Equal-weight"], results["SPY buy-and-hold"], periods
    )


if __name__ == "__main__":
    main()
