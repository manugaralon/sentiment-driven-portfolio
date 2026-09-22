# Sentiment-Driven Portfolio

A weekly-rebalanced portfolio of 9 large US stocks whose weights tilt with a daily news-sentiment
signal (FinBERT on headlines). The goal is to learn financial NLP and portfolio construction with
rigor, not to beat the market: no look-ahead, a temporal train/validation/test split, baselines with
transaction costs and honest statistics.

Status: work in progress. See [PLAN.md](PLAN.md) for the full plan (in Spanish).

## Setup

Linux / macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Pipeline

```bash
python scripts/download_prices.py   # adjusted closes -> data/raw/prices/close.parquet
python scripts/run_backtest.py      # baselines on train + validation -> reports/figures/
pytest
```

## Design choices so far

- **Universe chosen ex-ante**: the largest company per GICS sector at 2015-12-31 (verified with SEC
  EDGAR shares outstanding), excluding conglomerates and names with spin-offs. Picking today's
  giants would reward any signal correlated with "this company did well".
- **Timing**: weights decided at the close of the last session of each week are executed at the
  close of the next session. Between rebalances weights drift with prices; costs are charged on
  the dollars actually traded (10 bps by default).
- **Test period (2024-01 to 2026-08) is untouched** until the final evaluation.
