"""Download Alpaca news (Benzinga) for the universe: one parquet per month of UPDATE time.

Alpaca's start/end filter applies to updated_at, so we download up to the current month (an
article created before the end of the sample may have been edited later) and filter on
created_at when loading. Finished months already on disk are skipped, so the script can be
stopped and resumed. Pilot: `python scripts/download_news.py --start 2021-03 --end 2021-03`.
"""

import argparse
import time

import pandas as pd

from sentiment_portfolio.config import load_config, project_path
from sentiment_portfolio.news import alpaca_headers, fetch_news, to_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="first month, YYYY-MM (default: data_start)")
    parser.add_argument("--end", help="last month, YYYY-MM (default: current month)")
    args = parser.parse_args()

    cfg = load_config()
    symbols = list(cfg["universe"]) + list(cfg["news"]["aliases"])
    out_dir = project_path(cfg["paths"]["news"])
    out_dir.mkdir(parents=True, exist_ok=True)

    current = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
    months = pd.period_range(args.start or cfg["dates"]["data_start"][:7], args.end or str(current), freq="M")
    headers = alpaca_headers()

    t0, total = time.time(), 0
    for month in months:
        path = out_dir / f"{month}.parquet"
        if path.exists() and month < current:  # past months are closed: nothing new can land there
            continue
        start = month.start_time.tz_localize("UTC")
        end = (month + 1).start_time.tz_localize("UTC")
        articles = fetch_news(symbols, start, end, headers)
        to_frame(articles).to_parquet(path)
        total += len(articles)
        print(f"{month}: {len(articles):5d} articles  ({time.time() - t0:5.0f} s elapsed)")
    print(f"done: {total} articles downloaded in this run -> {out_dir}")


if __name__ == "__main__":
    main()
