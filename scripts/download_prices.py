"""Download adjusted daily closes for the universe, the benchmark and the risk-free rate.

Everything downstream reads the cached parquet, never yfinance directly.
"""

from sentiment_portfolio.config import load_config, project_path
from sentiment_portfolio.prices import download_close


def main() -> None:
    cfg = load_config()
    symbols = list(cfg["universe"]) + [cfg["benchmark"], cfg["risk_free"]]
    close = download_close(symbols, cfg["dates"]["data_start"], cfg["dates"]["test"][1])

    out = project_path(cfg["paths"]["close"])
    out.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(out)
    first, last = close.index[0].date(), close.index[-1].date()
    print(f"{len(close)} days x {close.shape[1]} symbols, {first} -> {last}")
    print("missing values:", close.isna().sum().to_dict())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
