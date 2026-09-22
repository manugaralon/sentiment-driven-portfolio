import pandas as pd
import pandas_market_calendars as mcal


def session_closes(start: str, end: str) -> pd.Series:
    """Close time (UTC) of every NYSE session in [start, end], indexed by session date.

    Uses the real calendar: holidays, early closes (13:00 ET) and daylight-saving changes.
    """
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return schedule["market_close"].rename("close")


def assign_session(available_at: pd.Series, closes: pd.Series) -> pd.Series:
    """The session whose closing decision can use each item: the first session that closes
    strictly AFTER the item became available.

    An item available at exactly 16:00:00 goes to the next session; weekends and holidays roll
    forward. Items after the last close get NaT. `available_at` must be tz-aware.
    """
    position = pd.DatetimeIndex(closes).searchsorted(pd.DatetimeIndex(available_at), side="right")
    sessions = closes.index.append(pd.DatetimeIndex([pd.NaT]))  # position == len(closes) -> NaT
    return pd.Series(sessions[position], index=available_at.index, name="session")
