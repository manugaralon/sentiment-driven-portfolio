import pandas as pd

from sentiment_portfolio.alignment import assign_session


def daily_panel(
    rows: pd.DataFrame, closes: pd.Series, tickers: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum of scores and article count per (session, ticker). Each article goes to the first
    session that closes after its `updated_at` (the headline we have is the edited version)."""
    session = assign_session(rows["updated_at"], closes)
    grouped = rows.assign(session=session).dropna(subset=["session"]).groupby(["session", "ticker"])["score"]
    full = {"index": closes.index, "columns": tickers, "fill_value": 0}
    score_sum = grouped.sum().unstack(fill_value=0).reindex(**full)
    count = grouped.count().unstack(fill_value=0).reindex(**full)
    return score_sum, count


def raw_sentiment(
    score_sum: pd.DataFrame, count: pd.DataFrame, n: int, surprise: bool = False, min_history: int = 20
) -> pd.DataFrame:
    """Mean article score over the last `n` sessions (NaN without articles).

    With `surprise`, each ticker's own mean score over all articles BEFORE the window is
    subtracted (NaN until it has `min_history` articles), so a ticker whose news is always
    upbeat is not permanently overweighted.
    """
    window = score_sum.rolling(n, min_periods=1).sum() / count.rolling(n, min_periods=1).sum()
    if not surprise:
        return window
    base_count = count.cumsum().shift(n)
    baseline = score_sum.cumsum().shift(n) / base_count
    return (window - baseline).where(base_count >= min_history)


def cross_sectional_z(raw: pd.DataFrame, min_names: int = 3) -> pd.DataFrame:
    """z-score across tickers on each session, among those with a value. Tickers without one
    get 0 (the equal weight); so does every ticker on sessions with fewer than `min_names`."""
    z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0)
    enough = raw.notna().sum(axis=1) >= min_names
    return z.where(enough, axis=0).fillna(0.0)


def build_signal(
    rows: pd.DataFrame,
    closes: pd.Series,
    tickers: list[str],
    n: int,
    surprise: bool = False,
    min_history: int = 20,
    min_names: int = 3,
) -> pd.DataFrame:
    """Signal known at the close of each session: news -> daily panel -> window mean -> z."""
    score_sum, count = daily_panel(rows, closes, tickers)
    return cross_sectional_z(raw_sentiment(score_sum, count, n, surprise, min_history), min_names)
