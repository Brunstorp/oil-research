# OVX-Filtered WTI Calendar Spread

IEOR 198 final project. Tests whether the WTI calendar spread (F2 − F1) mean-reverts when OVX is in a high-volatility regime.

**Result: null.** Net Sharpe 0.012 (t = 0.02), beaten by an always-on baseline. See `report.tex` for discussion.

## Setup

```bash
pip install -r requirements.txt
echo "FRED_API_KEY=..." >> .env
echo "EIA_API_KEY=..."  >> .env
```

Get free keys at:
- https://fred.stlouisfed.org/docs/api/api_key.html
- https://www.eia.gov/opendata/register.php

## Run

```bash
python data.py        # downloads + caches the panel
python backtest.py    # runs the strategy, writes results/
```

## Files

| File             | Purpose                                                   |
|------------------|-----------------------------------------------------------|
| `config.py`      | API keys, date range, series IDs, results dir             |
| `data.py`        | Pulls FRED + EIA series, builds the daily panel           |
| `backtest.py`    | Signal, walk-forward backtest, control + IS/OOS, plots    |
| `results/`       | `panel.parquet`, `backtest_*.parquet`, report, plot       |

## Strategy in one paragraph

On the third Wednesday of each month, if OVX is above its trailing 252-day 80th percentile, take a mean-reversion position on the spread: short F2 − F1 in contango, long in backwardation. One barrel per leg, 3 bps per leg per position change, hold until the next roll day. Compared against an always-on variant (same direction, no OVX filter) and a 70/30 chronological IS/OOS split.

## Sample

June 2008 – April 2024 (4,120 business days). Sample ends April 2024 because the EIA discontinued the constant-maturity series.