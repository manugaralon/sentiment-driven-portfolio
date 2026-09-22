"""Score every relevant headline with FinBERT (pinned revision), cached on disk.

The whole period is scored, test included: scoring is not evaluating. The quality report at the
end only looks at train headlines. Re-running only scores headlines missing from the cache.
"""

import numpy as np
import pandas as pd

from sentiment_portfolio.config import load_config, project_path
from sentiment_portfolio.finbert import finbert_scorer, score_headlines
from sentiment_portfolio.news import HEADLINE_CATEGORIES, load_news, ticker_rows


def quality_report(rows: pd.DataFrame) -> None:
    print("\n== Score distribution (train headlines) ==")
    bins = pd.cut(rows["score"], [-1.01, -0.5, -0.1, 0.1, 0.5, 1.0])
    print((bins.value_counts(normalize=True).sort_index() * 100).round(1).to_string())
    print(f"mean {rows['score'].mean():+.3f} | per ticker:")
    print(rows.groupby("ticker")["score"].agg(["mean", "count"]).round(3).to_string())

    print("\n== Mean score by headline category ==")
    for name, pattern in HEADLINE_CATEGORIES.items():
        in_cat = rows["headline"].str.contains(pattern, case=False, regex=True)
        print(f"  {name:17s} {rows.loc[in_cat, 'score'].mean():+.3f}  (n={in_cat.sum()})")

    print("\n== 50 random train headlines, sorted by score (manual review) ==")
    sample = rows.iloc[np.random.default_rng(0).choice(len(rows), 50, replace=False)]
    for _, r in sample.sort_values("score").iterrows():
        print(f"  {r['score']:+.2f}  [{r['ticker']:5s}] {r['headline'][:100]}")


def main() -> None:
    cfg = load_config()
    fb, dates = cfg["finbert"], cfg["dates"]
    news = load_news(
        project_path(cfg["paths"]["news"]), dates["data_start"], dates["test"][1], cfg["news"]["aliases"]
    )
    rows = ticker_rows(news, cfg["news"]["names"], cfg["news"]["max_symbols"])
    rows = rows[rows["relevant"]]

    cache_path = project_path(cfg["paths"]["finbert_cache"]) / f"finbert-{fb['revision'][:7]}.parquet"
    scorer = finbert_scorer(fb["model"], fb["revision"], fb["max_length"], fb["batch_size"])
    scores = score_headlines(rows["headline"], cache_path, scorer, fb["chunk_size"])

    first, last = pd.Timestamp(dates["train"][0], tz="UTC"), pd.Timestamp(dates["train"][1], tz="UTC")
    train = rows[(rows["created_at"] >= first) & (rows["created_at"] < last + pd.Timedelta(days=1))]
    train = train.merge(scores[["text", "score"]], left_on="headline", right_on="text")
    quality_report(train)


if __name__ == "__main__":
    main()
