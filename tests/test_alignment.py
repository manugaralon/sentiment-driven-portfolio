import pandas as pd
import pytest

from sentiment_portfolio.alignment import assign_session, session_closes

CLOSES = session_closes("2019-11-20", "2021-03-31")


def session_of(timestamp: pd.Timestamp) -> pd.Timestamp:
    return assign_session(pd.Series([timestamp]), CLOSES).iloc[0]


@pytest.mark.parametrize(
    "published_et, expected",
    [
        ("2019-12-03 15:59", "2019-12-03"),  # before the close: same session
        ("2019-12-03 16:00", "2019-12-04"),  # exactly at the close: next session
        ("2019-12-03 16:01", "2019-12-04"),
        ("2019-12-07 11:00", "2019-12-09"),  # Saturday -> Monday
        ("2019-11-28 10:00", "2019-11-29"),  # Thanksgiving (closed) -> next session
        ("2019-11-29 12:59", "2019-11-29"),  # half day: the market closes at 13:00
        ("2019-11-29 14:00", "2019-12-02"),  # after the early close -> Monday
    ],
)
def test_assign_session_eastern_time(published_et, expected):
    published = pd.Timestamp(published_et, tz="America/New_York")
    assert session_of(published) == pd.Timestamp(expected)


@pytest.mark.parametrize(
    "published_utc, expected",
    [
        ("2021-03-12 20:30", "2021-03-12"),  # Friday, EST: the close is 21:00 UTC -> before it
        ("2021-03-15 20:30", "2021-03-16"),  # Monday, EDT since Mar 14: the close is 20:00 UTC -> after
    ],
)
def test_assign_session_across_daylight_saving_change(published_utc, expected):
    assert session_of(pd.Timestamp(published_utc, tz="UTC")) == pd.Timestamp(expected)


def test_items_after_the_last_close_get_no_session():
    assert pd.isna(session_of(pd.Timestamp("2021-04-01 12:00", tz="America/New_York")))


def test_calendar_has_the_early_close():
    assert CLOSES.loc["2019-11-29"] == pd.Timestamp("2019-11-29 13:00", tz="America/New_York")
