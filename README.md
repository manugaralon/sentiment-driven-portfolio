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
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

(On Linux the default `torch` wheel bundles CUDA, several GB; the CPU build is enough here.)

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Pipeline

Copy `.env.example` to `.env` and add free Alpaca paper-account keys (news API).

```bash
python scripts/download_prices.py   # adjusted closes -> data/raw/prices/close.parquet
python scripts/download_news.py     # Alpaca/Benzinga news, monthly chunks (~30 min, resumable)
python scripts/news_eda.py          # coverage, relevance, timestamps -> reports/figures/
python scripts/score_news.py        # FinBERT scores, cached on disk (~1 h on CPU, resumable)
python scripts/build_signal.py      # daily signal panel + IC analysis on train -> reports/figures/
python scripts/run_backtest.py      # baselines, tilt grid and controls on train + validation
pytest                              # fast tests; `pytest -m slow` also loads the real FinBERT
```

## Design choices so far

- **Universe chosen ex-ante**: the largest company per GICS sector at 2015-12-31 (verified with SEC
  EDGAR shares outstanding), excluding conglomerates and names with spin-offs. Picking today's
  giants would reward any signal correlated with "this company did well".
- **Timing**: weights decided at the close of the last session of each week are executed at the
  close of the next session. Between rebalances weights drift with prices; costs are charged on
  the dollars actually traded (10 bps by default).
- **News availability**: an article can only inform the decision of the first NYSE session that
  closes strictly after its `updated_at` time (the headline we hold is the edited version). The real
  calendar is used, including 13:00 early closes and daylight-saving changes.
- **Relevance**: an article counts for a ticker if the headline names the company and it is tagged
  with at most 5 symbols (more are lists of stocks).
- **FinBERT** (pinned revision) scores each headline as P(positive) - P(negative); labels are read
  from the model config, never hard-coded.
- **Signal**: mean headline score over the last N sessions (article-weighted), z-scored across
  tickers; tickers without news get z = 0 (equal weight). A "surprise" variant first subtracts each
  ticker's own mean score from before the window, so a company with always-upbeat coverage is not
  permanently overweighted. A truncation test checks the signal up to T ignores news after T.
- **Tilt rule**: w_i = (1 + λ·z_i)/n with z capped at ±2, projected onto weights in [0.5/n, 2/n]
  that sum to 1 (long-only; λ = 0 is exactly equal weight). N and λ are picked on train by
  information ratio vs equal weight, the whole grid is reported, and validation is judged against
  a placebo (signal shuffled across tickers), a momentum tilt and a paired block bootstrap.
- **Test period (2024-01 to 2026-08) is untouched** until the final evaluation.
