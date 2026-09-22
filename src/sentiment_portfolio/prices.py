import pandas as pd
import yfinance as yf


def download_close(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily close for each symbol, adjusted for splits and dividends (total return).

    `end` is inclusive. Columns are symbols, index is dates (naive, exchange local).
    """
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(symbols, start=start, end=end_exclusive, auto_adjust=True, progress=False)
    close = data["Close"][symbols]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close


def market_data(
    close: pd.DataFrame, assets: list[str], benchmark: str, rf_symbol: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split raw closes into daily simple returns (assets + benchmark) and daily risk-free rate.

    Sessions are the days the benchmark traded. The T-bill yield is forward-filled onto them
    (bond-market holidays differ) and lagged one day: the yield known at the close of d-1 is
    what you earn over day d.
    """
    sessions = close[benchmark].dropna().index
    prices = close.loc[sessions, assets + [benchmark]]
    missing = prices.isna().sum()
    if missing.any():
        raise ValueError(f"missing prices on benchmark sessions: {missing[missing > 0].to_dict()}")
    returns = prices.pct_change().iloc[1:]
    rf = (close[rf_symbol].ffill().loc[sessions] / 100 / 252).shift(1).loc[returns.index]
    return returns, rf.rename("rf")
