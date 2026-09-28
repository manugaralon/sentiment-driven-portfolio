"""Build the daily sentiment signal and check, on train only, whether it says anything about
future returns (IC) or just echoes past ones.

Signals are saved for the whole period (building them uses no returns); every statistic below
uses train prices only, so no forward window reaches into validation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentiment_portfolio.alignment import session_closes
from sentiment_portfolio.config import load_config, project_path
from sentiment_portfolio.finbert import finbert_scorer, score_headlines
from sentiment_portfolio.news import load_news, ticker_rows
from sentiment_portfolio.prices import market_data
from sentiment_portfolio.signals import build_signal, daily_panel
from sentiment_portfolio.stats import cross_sectional_ic, forward_returns, newey_west_mean, past_returns

INK, INK_MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COLORS = {"raw": "#2a78d6", "surprise": "#eb6834"}


def scored_rows(cfg: dict) -> pd.DataFrame:
    fb, dates = cfg["finbert"], cfg["dates"]
    news = load_news(
        project_path(cfg["paths"]["news"]), dates["data_start"], dates["test"][1], cfg["news"]["aliases"]
    )
    rows = ticker_rows(news, cfg["news"]["names"], cfg["news"]["max_symbols"])
    rows = rows[rows["relevant"]]
    cache_path = project_path(cfg["paths"]["finbert_cache"]) / f"finbert-{fb['revision'][:7]}.parquet"
    scorer = finbert_scorer(fb["model"], fb["revision"], fb["max_length"], fb["batch_size"])
    scores = score_headlines(rows["headline"], cache_path, scorer, fb["chunk_size"])
    return rows.merge(scores[["text", "score"]], left_on="headline", right_on="text")


def coverage_table(count: pd.DataFrame, score_sum: pd.DataFrame, signals: dict, n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "% sessions with news": (count > 0).mean() * 100,
        "articles/session": count.mean(),
        "mean score": score_sum.sum() / count.sum(),
        f"% z != 0 (raw, N={n})": (signals[("raw", n)] != 0).mean() * 100,
        f"% z != 0 (surprise, N={n})": (signals[("surprise", n)] != 0).mean() * 100,
    }).round(2)


def ic_table(signals: dict, targets: dict) -> pd.DataFrame:
    """Mean daily IC per (variant, N, target) with a Newey-West CI. The lag covers both overlaps:
    the target window (h) and the signal's own rolling window (N)."""
    rows = {}
    for (variant, n), signal in signals.items():
        for (kind, h), target in targets.items():
            ic = cross_sectional_ic(signal, target)
            rows[(variant, n, kind, h)] = newey_west_mean(ic, lags=h + n)
    table = pd.DataFrame(rows).T
    table.index.names = ["variant", "N", "target", "h"]
    return table


def plot_ic(table: pd.DataFrame, path) -> None:
    panels = table.index.droplevel(["variant", "N"]).unique()
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.6), sharey=True, facecolor=SURFACE)
    for ax, (kind, h) in zip(axes, panels, strict=True):
        sub = table.xs((kind, h), level=["target", "h"])
        windows = sorted(sub.index.get_level_values("N").unique())
        for i, variant in enumerate(COLORS):
            rows = sub.loc[variant].loc[windows]
            x = np.arange(len(windows)) + (i - 0.5) * 0.3
            ax.errorbar(x, rows["mean"], yerr=1.96 * rows["se"], fmt="o", color=COLORS[variant], capsize=3,
                        label=variant)
        ax.axhline(0, color=INK_MUTED, lw=0.8)
        ax.set_xticks(range(len(windows)), [f"N={n}" for n in windows])
        ax.set_title(f"{kind} return, h={h}", color=INK, fontsize=10, loc="left")
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Mean daily IC (Spearman) ± 95% NW", color=INK_MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sentiment signal vs returns, train 2016–2021", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    cfg = load_config()
    tickers, dates, sig = list(cfg["universe"]), cfg["dates"], cfg["signal"]
    rows = scored_rows(cfg)
    closes = session_closes(dates["data_start"], dates["test"][1])

    signals = {
        (variant, n): build_signal(rows, closes, tickers, n, variant == "surprise", sig["min_history"],
                                   sig["min_names"])
        for variant in ("raw", "surprise")
        for n in sig["windows"]
    }
    out = project_path(cfg["paths"]["processed"]) / "signals.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    long = pd.concat({f"{v}_{n}": s.stack() for (v, n), s in signals.items()}, axis=1)
    long.rename_axis(["session", "ticker"]).to_parquet(out)
    print(f"saved {out} ({len(long)} session-ticker rows, {long.shape[1]} signals)")

    # ---- Everything below is train only ----
    first, last = dates["train"]
    close = pd.read_parquet(project_path(cfg["paths"]["close"]))
    returns, _ = market_data(close, tickers, cfg["benchmark"], cfg["risk_free"])
    returns = returns.loc[:last, tickers]  # forward windows stop at the end of train
    train = {key: s.loc[first:last] for key, s in signals.items()}

    score_sum, count = daily_panel(rows, closes, tickers)
    mid = sig["windows"][len(sig["windows"]) // 2]
    print("\n== Coverage by ticker (train sessions) ==")
    print(coverage_table(count.loc[first:last], score_sum.loc[first:last], train, mid).to_string())

    targets = {("forward", h): forward_returns(returns, h, cfg["backtest"]["execution_lag_sessions"])
               for h in sig["ic_horizons"]}
    targets |= {("past", h): past_returns(returns, h) for h in (5, 20)}
    table = ic_table(train, targets)
    print("\n== Mean daily IC, train (Newey-West, lag = h + N) ==")
    print(table.to_string(float_format=lambda x: f"{x:7.4f}"))

    fig_path = project_path(cfg["paths"]["figures"]) / "signal_ic_train.png"
    plot_ic(table, fig_path)
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
