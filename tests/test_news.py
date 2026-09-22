import pandas as pd

from sentiment_portfolio import news
from sentiment_portfolio.news import fetch_news, load_news, mentions_company, ticker_rows, to_frame


def article(id_, created, updated=None, headline="h", symbols=("AAPL",)):
    return {
        "id": id_,
        "created_at": created,
        "updated_at": updated or created,
        "headline": headline,
        "summary": "",
        "symbols": list(symbols),
        "source": "benzinga",
        "url": "",
        "content": "<p>body</p>",  # extra fields must be dropped
        "author": "someone",
    }


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise AssertionError("unexpected HTTP error")


def test_fetch_news_follows_pagination_and_retries_on_429(monkeypatch):
    responses = [
        FakeResponse(200, {"news": [article(1, "2020-01-02T10:00:00Z")], "next_page_token": "abc"}),
        FakeResponse(429),
        FakeResponse(200, {"news": [article(2, "2020-01-03T10:00:00Z")], "next_page_token": None}),
    ]
    calls = []

    def fake_get(url, headers, params, timeout):
        calls.append(dict(params))
        return responses.pop(0)

    monkeypatch.setattr(news.requests, "get", fake_get)
    monkeypatch.setattr(news.time, "sleep", lambda s: None)
    start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-02-01", tz="UTC")
    articles = fetch_news(["AAPL"], start, end, headers={})

    assert [a["id"] for a in articles] == [1, 2]
    assert "page_token" not in calls[0] and calls[1]["page_token"] == "abc" == calls[2]["page_token"]
    assert calls[0]["start"] == "2020-01-01T00:00:00Z"


def test_to_frame_keeps_only_used_fields_as_utc():
    df = to_frame([article(1, "2020-01-02T10:00:00Z")])
    assert list(df.columns) == news.FIELDS
    assert str(df["created_at"].dt.tz) == "UTC"


def test_to_frame_handles_empty_month():
    assert to_frame([]).empty


def test_load_news_dedupes_maps_aliases_and_filters_on_created_at(tmp_path):
    chunk_jan = [
        article(1, "2016-01-05T15:00:00Z", headline="original"),
        article(2, "2015-12-31T23:00:00Z", updated="2016-01-02T09:00:00Z"),  # created before start
    ]
    chunk_feb = [
        article(1, "2016-01-05T15:00:00Z", updated="2016-02-01T08:00:00Z", headline="edited"),
        article(3, "2016-02-10T12:00:00Z", symbols=("GOOG", "GOOGL", "AAPL")),
        article(4, "2016-03-01T00:00:01Z"),  # created after end
    ]
    to_frame(chunk_jan).to_parquet(tmp_path / "2016-01.parquet")
    to_frame(chunk_feb).to_parquet(tmp_path / "2016-02.parquet")

    df = load_news(tmp_path, "2016-01-01", "2016-02-29", aliases={"GOOG": "GOOGL"})

    assert df["id"].tolist() == [1, 3]
    assert df.loc[df["id"] == 1, "headline"].item() == "edited"  # latest update wins
    assert df.loc[df["id"] == 3, "symbols"].item() == ["AAPL", "GOOGL"]


def test_mentions_company_uses_word_boundaries():
    headlines = pd.Series(
        ["Apple's iPhone sales", "Pineapple futures", "PG&E files for bankruptcy", "P&G raises guidance"]
    )
    assert mentions_company(headlines, "apple|aapl").tolist() == [True, False, False, False]
    assert mentions_company(headlines, "procter|p&g").tolist() == [False, False, False, True]


def test_ticker_rows_flags_relevant_articles():
    news = to_frame(
        [
            article(
                1,
                "2020-01-02T10:00:00Z",
                headline="Apple and Amazon sign a deal",
                symbols=("AAPL", "AMZN", "XYZ"),
            ),
            article(
                2,
                "2020-01-02T11:00:00Z",
                headline="Top stocks: Apple, Tesla",
                symbols=list("ABCDEF") + ["AAPL"],
            ),
            article(3, "2020-01-02T12:00:00Z", headline="Markets fall on oil", symbols=("AAPL",)),
        ]
    )
    rows = ticker_rows(news, {"AAPL": "apple", "AMZN": "amazon"}, max_symbols=5)
    relevant = rows.set_index(["id", "ticker"])["relevant"]
    assert relevant.to_dict() == {
        (1, "AAPL"): True,
        (1, "AMZN"): True,
        (2, "AAPL"): False,
        (3, "AAPL"): False,
    }
