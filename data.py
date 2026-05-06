"""Data layer: FRED for spot + OVX, EIA for the WTI futures curve.

Both APIs are free; register at:
  - https://fred.stlouisfed.org/docs/api/api_key.html
  - https://www.eia.gov/opendata/register.php

After loading, the panel is automatically truncated to the date range
where ALL four price series (S, F1, F2, F3) have data. Currently this
means the panel ends on 2024-04-05 (when the EIA discontinued RCLC1-3),
even though FRED keeps providing spot and OVX through the present.
"""
from __future__ import annotations
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import config

UA = {"User-Agent": "Mozilla/5.0 (research script)"}


def fred_series(series_id: str,
                start: str = config.START_DATE,
                end:   str = config.END_DATE) -> pd.Series:
    if not config.FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY missing; add it to .env")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": config.FRED_API_KEY,
              "file_type": "json",
              "observation_start": start, "observation_end": end}
    r = requests.get(url, params=params, timeout=60, headers=UA)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["observations"])
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan),
                                errors="coerce")
    s = df.dropna(subset=["value"]).set_index("date")["value"].sort_index()
    s.name = series_id
    return s


def eia_series(series_id: str,
               start: str = config.START_DATE,
               end:   str = config.END_DATE) -> pd.Series:
    if not config.EIA_API_KEY:
        raise RuntimeError("EIA_API_KEY missing; add it to .env")
    base = "https://api.eia.gov/v2/petroleum/pri/fut/data/"
    params = {"api_key": config.EIA_API_KEY, "frequency": "daily",
              "data[0]": "value", "facets[series][]": series_id,
              "start": start, "end": end,
              "sort[0][column]": "period", "sort[0][direction]": "asc",
              "offset": 0, "length": 5000}
    r = requests.get(base, params=params, timeout=60, headers=UA)
    r.raise_for_status()
    rows = r.json().get("response", {}).get("data", [])
    if not rows:
        warnings.warn(f"EIA returned 0 rows for {series_id}")
        return pd.Series(dtype=float, name=series_id)
    df = pd.DataFrame(rows)
    df["date"]  = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna(subset=["value"]).set_index("date")["value"].sort_index()
    s.name = series_id
    return s


def load_all() -> tuple[pd.DataFrame, list[str]]:
    """Build the panel: S, OVX (FRED) and F1, F2, F3 (EIA).

    The returned panel is truncated to the date range where all four
    PRICE series (S, F1, F2, F3) have observations. OVX may extend
    beyond that range but is also truncated for consistency.
    """
    audit = [f"# Data audit ({datetime.utcnow():%Y-%m-%d %H:%M UTC})", ""]

    cols = {}
    for col, sid in config.FRED_SERIES.items():
        try:
            s = fred_series(sid)
            cols[col] = s
            audit.append(f"FRED OK   {sid:14s} ({col}): {len(s):5d} obs, "
                         f"{s.index.min().date()} -> {s.index.max().date()}")
        except Exception as e:
            audit.append(f"FRED FAIL {sid:14s} ({col}): {e}")
            cols[col] = pd.Series(dtype=float, name=sid)

    for col, sid in config.EIA_FUTURES.items():
        try:
            s = eia_series(sid)
            cols[col] = s
            audit.append(f"EIA  OK   {sid:14s} ({col}): {len(s):5d} obs, "
                         f"{s.index.min().date()} -> {s.index.max().date()}")
        except Exception as e:
            audit.append(f"EIA  FAIL {sid:14s} ({col}): {e}")
            cols[col] = pd.Series(dtype=float, name=sid)

    panel = pd.concat(cols, axis=1).sort_index().asfreq("B").ffill(limit=2)

    # Truncate to the window where all four price series exist
    price_cols = ["S", "F1", "F2", "F3"]
    have_all = panel[price_cols].notna().all(axis=1)
    if have_all.any():
        first = have_all[have_all].index.min()
        last  = have_all[have_all].index.max()
        before = len(panel)
        panel = panel.loc[first:last].copy()
        audit.append("")
        audit.append(f"Truncated panel to common-coverage window "
                     f"{first.date()} -> {last.date()}: "
                     f"{before} -> {len(panel)} obs")

    # Note (but do not remove) negative prices
    for col in price_cols:
        if col in panel.columns:
            mask = (panel[col] <= 0) & panel[col].notna()
            if mask.any():
                audit.append(f"  Note: {int(mask.sum())} non-positive obs "
                             f"in {col} (KEPT in dollar-level analysis): "
                             f"{list(panel.index[mask][:5])}")

    return panel, audit


if __name__ == "__main__":
    panel, audit = load_all()
    (config.RESULTS_DIR / "data_audit.txt").write_text("\n".join(audit))
    panel.to_parquet(config.RESULTS_DIR / "panel.parquet")
    print("\n".join(audit))
    print(panel.tail())