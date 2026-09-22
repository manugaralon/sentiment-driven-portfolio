import pandas as pd


def equal_weight(decision_dates: pd.DatetimeIndex, assets: list[str]) -> pd.DataFrame:
    """Target weights 1/n on every decision date."""
    return pd.DataFrame(1 / len(assets), index=decision_dates, columns=assets)
