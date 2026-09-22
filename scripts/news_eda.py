"""EDA of the downloaded news, on train + validation only (the test period stays closed).

Each section feeds a phase-2 decision: coverage (window N, start date), relevance filter,
timestamp sanity, headline edits and boilerplate. Session dates here are approximate (ET
calendar date); the exact news-to-session alignment is phase 3.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentiment_portfolio.config import load_config, project_path
from sentiment_portfolio.news import load_news, mentions_company

INK, INK_MUTED, SURFACE, GRID, BAR = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df", "#2a78d6"
CATEGORIES = {
    "analyst rating": r"\b(?:maintains|upgrades?|downgrades?|initiates|reiterates|price target)\b",
    "options activity": r"options activity|unusual options|whale",
    "movers lists": r"stocks moving|biggest movers|mid-day|pre-market session|after-hours session|52-week",
    "earnings": r"\b(?:earnings|eps|q[1-4]|quarter(?:ly)?)\b",
}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")


def ticker_rows(news: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """One row per (article, universe ticker it is tagged with), with the relevance flags."""
    rows = news.assign(ticker=news["symbols"].map(lambda s: [t for t in s if t in names]))
    rows = rows.explode("ticker").dropna(subset=["ticker"])
    rows["n_symbols"] = rows["symbols"].map(len)
    rows["named"] = False
    for ticker, pattern in names.items():
        mask = rows["ticker"] == ticker
        rows.loc[mask, "named"] = mentions_company(rows.loc[mask, "headline"], pattern)
    rows["date_et"] = rows["created_at"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return rows


def coverage(rows: pd.DataFrame, sessions: pd.DatetimeIndex, figures) -> None:
    named = rows[rows["named"]]
    print("\n== 1. Coverage: relevant (named) articles per year ==")
    print(
        named.pivot_table(index="ticker", columns=named["date_et"].dt.year, values="id", aggfunc="count")
        .fillna(0)
        .astype(int)
        .to_string()
    )

    days = named[named["date_et"].isin(sessions)].drop_duplicates(["ticker", "date_et"])
    sessions_per_month = pd.Series(1, index=sessions).groupby(sessions.to_period("M")).count()
    share = (
        days.groupby(["ticker", days["date_et"].dt.to_period("M")])
        .size()
        .unstack(0)
        .reindex(sessions_per_month.index)
        .fillna(0)
        .div(sessions_per_month, axis=0)
    )
    print("\n% of sessions with >= 1 relevant article (whole period):")
    print((share.mean() * 100).round(1).sort_values(ascending=False).to_string())

    fig, axes = plt.subplots(3, 3, figsize=(11, 7), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, ticker in zip(axes.flat, share.mean().sort_values(ascending=False).index, strict=True):
        ax.plot(share.index.to_timestamp(), share[ticker] * 100, color=BAR, lw=1.2)
        ax.set_title(ticker, loc="left", color=INK, fontsize=10)
        ax.set_ylim(0, 100)
        style(ax)
    fig.suptitle(
        "Share of sessions with at least one article naming the company (monthly, %)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(figures / "news_coverage.png", dpi=150, facecolor=SURFACE)


def relevance(rows: pd.DataFrame) -> None:
    k3 = rows["n_symbols"] <= 3
    table = (
        pd.DataFrame(
            {
                "all": rows.groupby("ticker").size(),
                "k<=3": rows[k3].groupby("ticker").size(),
                "named": rows[rows["named"]].groupby("ticker").size(),
                "named & k<=3": rows[rows["named"] & k3].groupby("ticker").size(),
            }
        )
        .fillna(0)
        .astype(int)
        .sort_values("all", ascending=False)
    )
    print("\n== 2. Relevance filters: articles kept per ticker ==")
    print(table.to_string())
    rng = np.random.default_rng(0)
    for label, mask in [
        ("named but > 3 symbols", rows["named"] & ~k3),
        ("<= 3 symbols but not named", ~rows["named"] & k3),
    ]:
        sample = rows[mask]
        sample = sample.iloc[rng.choice(len(sample), size=min(10, len(sample)), replace=False)]
        print(f"\nSample: {label} ({mask.mean():.0%} of rows)")
        for _, r in sample.iterrows():
            print(f"  [{r['ticker']:5s}] {r['headline'][:110]}")


def timestamps(news: pd.DataFrame, figures) -> None:
    et = news["created_at"].dt.tz_convert("America/New_York")
    minutes = et.dt.hour * 60 + et.dt.minute
    weekend = et.dt.dayofweek >= 5
    buckets = pd.Series(
        np.select(
            [weekend, minutes < 570, minutes < 960],
            ["weekend", "pre-market (<9:30)", "session (9:30-16:00)"],
            default="after close (>=16:00)",
        )
    )
    print("\n== 3. Timestamps (ET) ==")
    print((buckets.value_counts(normalize=True) * 100).round(1).to_string())
    premarket = news["headline"].str.contains("pre-market", case=False)
    print(
        f"Sanity: median ET hour of 'pre-market' headlines = {et[premarket].dt.hour.median():.0f}h "
        f"(n={premarket.sum()}; expected roughly 4-9h)"
    )

    counts = et[~weekend].dt.hour.value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    ax.axvspan(9.5, 16, color="#eeedea", zorder=0)
    ax.bar(counts.index, counts.to_numpy(), color=BAR, width=0.8, align="edge", zorder=2)  # bar h = [h, h+1)
    ax.text(12.25, counts.max() * 1.02, "NYSE session 9:30-16:00", ha="center", color=INK_MUTED, fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of publication (ET, weekdays)", color=INK_MUTED)
    ax.set_title("When articles are published", loc="left", color=INK, fontsize=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(figures / "news_hour_et.png", dpi=150, facecolor=SURFACE)


def edits(news: pd.DataFrame) -> None:
    delta = news["updated_at"] - news["created_at"]
    by_year = pd.DataFrame(
        {
            "> 1 h": (delta > pd.Timedelta("1h")).groupby(news["created_at"].dt.year).mean() * 100,
            "> 1 day": (delta > pd.Timedelta("1D")).groupby(news["created_at"].dt.year).mean() * 100,
        }
    ).round(1)
    print("\n== 4. Edits: % of articles updated long after creation ==")
    print(by_year.to_string())


def boilerplate(news: pd.DataFrame, rows: pd.DataFrame) -> None:
    print("\n== 5. Boilerplate: % of articles per headline category ==")
    for name, pattern in CATEGORIES.items():
        print(f"  {name:17s} {news['headline'].str.contains(pattern, case=False, regex=True).mean():6.1%}")
    dup = rows.assign(h=rows["headline"].str.lower()).duplicated(["ticker", "date_et", "h"]).mean()
    print(f"Duplicate headlines (same ticker, same ET date): {dup:.1%}")


def main() -> None:
    cfg = load_config()
    names = cfg["news"]["names"]
    start, end = cfg["dates"]["train"][0], cfg["dates"]["validation"][1]
    news = load_news(project_path(cfg["paths"]["news"]), start, end, cfg["news"]["aliases"])
    sessions = pd.read_parquet(project_path(cfg["paths"]["close"]))[cfg["benchmark"]].dropna().index
    sessions = sessions[(sessions >= start) & (sessions <= end)]
    figures = project_path(cfg["paths"]["figures"])
    print(f"{len(news)} unique articles created {start} -> {end}")

    rows = ticker_rows(news, names)
    coverage(rows, sessions, figures)
    relevance(rows)
    timestamps(news, figures)
    edits(news)
    boilerplate(news, rows)


if __name__ == "__main__":
    main()
