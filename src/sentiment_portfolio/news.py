import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from sentiment_portfolio.config import ROOT

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
FIELDS = ["id", "created_at", "updated_at", "headline", "summary", "symbols", "source", "url"]


def alpaca_headers() -> dict[str, str]:
    load_dotenv(ROOT / ".env")
    try:
        return {
            "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"],
        }
    except KeyError as missing:
        raise RuntimeError(f"{missing.args[0]} not set: copy .env.example to .env and fill it in") from None


def fetch_news(
    symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, headers: dict, pause: float = 0.3
) -> list[dict]:
    """All articles tagged with any of `symbols` whose UPDATE time falls in [start, end].

    Alpaca filters on updated_at, not created_at (verified 2026-09-22), so callers must filter
    on created_at themselves. `pause` keeps us under the free plan's 200 calls/min.
    """
    params = {
        "symbols": ",".join(symbols),
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 50,
        "sort": "asc",
        "include_content": "false",
    }
    articles = []
    while True:
        page = _get(params, headers)
        articles += page["news"]
        token = page.get("next_page_token")
        if not token:
            return articles
        params["page_token"] = token
        time.sleep(pause)


def _get(params: dict, headers: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        response = requests.get(NEWS_URL, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** (attempt + 1))  # rate limited or server error: back off and retry
            continue
        response.raise_for_status()
    raise RuntimeError(f"Alpaca news: giving up after {retries} tries (last HTTP {response.status_code})")


def to_frame(articles: list[dict]) -> pd.DataFrame:
    """Keep only the fields we use; timestamps as UTC datetimes."""
    df = pd.DataFrame(articles, columns=FIELDS)
    for col in ("created_at", "updated_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def load_news(news_dir: Path, start: str, end: str, aliases: dict[str, str]) -> pd.DataFrame:
    """Concatenate the monthly chunks into one table of unique articles CREATED in [start, end].

    An article can appear twice (chunk boundaries, re-downloaded current month): the latest
    update wins. Ticker aliases (GOOG -> GOOGL) are applied to each article's symbol list.
    """
    df = pd.concat([pd.read_parquet(f) for f in sorted(Path(news_dir).glob("*.parquet"))], ignore_index=True)
    df = df.sort_values("updated_at").drop_duplicates("id", keep="last")
    df["symbols"] = df["symbols"].map(lambda syms: sorted({aliases.get(s, s) for s in syms}))
    first = pd.Timestamp(start, tz="UTC")
    after_last = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["created_at"] >= first) & (df["created_at"] < after_last)]
    return df.sort_values("created_at").reset_index(drop=True)
