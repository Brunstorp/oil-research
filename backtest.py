"""Backtest: OVX-filtered WTI calendar spread (F1 vs F2).

Strategy
--------
1. Compute spread = F2 - F1 (constant-maturity 1m and 2m WTI futures).
2. Define a "high-vol regime" as days when OVX exceeds its trailing
   1-year (252 trading-day) 80th percentile. The percentile is built
   using only data available up to and including day t, then we lag
   the resulting flag by one day before trading -> no lookahead.
3. On the 3rd Wednesday of each month (proxy for WTI expiry / roll
   day) we re-evaluate:
       position on the spread = -sign(F2 - F1)
   i.e. mean-revert the curve: short the spread in contango, long
   the spread in backwardation. If the regime flag is False on that
   day, we stay flat.
4. Hold the position fixed until the next 3rd Wednesday, then
   re-evaluate.

P&L convention
--------------
We track dollar P&L for a 1-barrel-per-leg position. Daily P&L is
    position_held_overnight * (spread_t - spread_{t-1}).
On roll days the daily P&L is set to zero, because the constant-
maturity series has a contract-rollover gap that is NOT a tradable
return. Transaction costs are charged on each position change as
COST_BPS_PER_LEG bps applied to (|F1| + |F2|).

Comparisons reported
--------------------
- OVX-filtered strategy, full sample
- "Always-on" control (same direction trade every month, no OVX
  filter) -> isolates whether OVX is doing real work
- 70/30 chronological in-sample / out-of-sample split

Outputs
-------
- results/backtest_report.txt
- results/backtest_main.parquet, backtest_control.parquet
- results/backtest_plot.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from data import load_all


# ---------- parameters ----------
LOOKBACK_DAYS    = 252        # 1y rolling window for OVX percentile
OVX_PERCENTILE   = 0.80       # regime threshold
COST_BPS_PER_LEG = 3.0        # transaction cost per leg per position change
NOTIONAL_PER_LEG = 1.0        # 1 barrel per leg -> P&L in dollars
START_CAPITAL    = 100.0      # cosmetic, only for equity-curve scale


# ---------- signal construction ----------
def compute_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add spread, OVX percentile, regime flag, and raw direction.

    All rolling stats use only past-and-current data, so no
    lookahead is introduced here.
    """
    df = panel.copy()
    df["spread"]  = df["F2"] - df["F1"]
    df["ovx_pct"] = (df["OVX"]
                     .rolling(LOOKBACK_DAYS,
                              min_periods=LOOKBACK_DAYS // 2)
                     .quantile(OVX_PERCENTILE))
    df["regime"]  = df["OVX"] > df["ovx_pct"]
    df["dir_raw"] = -np.sign(df["spread"])    # mean-revert the spread
    return df


def roll_days(idx: pd.DatetimeIndex) -> pd.Series:
    """Mark the 3rd Wednesday of each month (proxy for WTI expiry)."""
    out = pd.Series(False, index=idx)
    df = pd.DataFrame({"d": idx})
    df["w"]  = df["d"].dt.dayofweek
    df["ym"] = df["d"].dt.to_period("M")
    weds = df[df["w"] == 2].copy()
    weds["nth"] = weds.groupby("ym").cumcount()
    third_dates = weds.loc[weds["nth"] == 2, "d"]
    out.loc[third_dates] = True
    return out


# ---------- backtest engine ----------
def run_backtest(panel: pd.DataFrame,
                 *, use_ovx_filter: bool = True
                 ) -> tuple[pd.DataFrame, dict]:
    """Walk forward, taking decisions only on roll days."""
    df = compute_signals(panel)
    df["roll"] = roll_days(df.index)

    # Lag both signal and regime by 1 day so that a decision at
    # close-of-day-t uses information available at end-of-day-(t-1).
    df["dir_lag"]    = df["dir_raw"].shift(1)
    df["regime_lag"] = df["regime"].shift(1)

    # Walk forward through the roll calendar, holding the position
    # constant between consecutive 3rd Wednesdays.
    pos = np.zeros(len(df))
    current = 0.0
    for i in range(len(df)):
        if df["roll"].iat[i]:
            d = df["dir_lag"].iat[i]
            r = df["regime_lag"].iat[i]
            ok_regime = (not use_ovx_filter) or (pd.notna(r) and bool(r))
            if pd.notna(d) and ok_regime:
                current = float(d)
            else:
                current = 0.0
        pos[i] = current
    df["position"] = pos

    # P&L on a 1-bbl-per-leg spread, masking the roll-day gap.
    df["d_spread"]  = df["spread"].diff()
    df["pos_held"]  = df["position"].shift(1).fillna(0.0)
    df["pnl_gross"] = df["pos_held"] * df["d_spread"] * NOTIONAL_PER_LEG
    df.loc[df["roll"], "pnl_gross"] = 0.0

    # Costs: charged whenever the position changes (open / close /
    # flip). Cost in dollars = bps * (|F1| + |F2|) per barrel.
    pos_change = (df["position"].diff().abs()
                  .fillna(df["position"].abs()) > 0)
    df["cost"]    = (pos_change.astype(float)
                     * (COST_BPS_PER_LEG / 1e4)
                     * (df["F1"].abs() + df["F2"].abs())
                     * NOTIONAL_PER_LEG)
    df["pnl_net"] = df["pnl_gross"] - df["cost"]

    df["equity"]       = START_CAPITAL + df["pnl_net"].cumsum()
    df["equity_gross"] = START_CAPITAL + df["pnl_gross"].cumsum()

    # Metrics computed AFTER the lookback warm-up only.
    warm = df["ovx_pct"].first_valid_index()
    sub  = df.loc[warm:]
    in_pos = sub["pos_held"] != 0
    daily  = sub["pnl_net"]

    n_active = int(in_pos.sum())
    if n_active > 1 and daily[in_pos].std() > 0:
        mu = daily[in_pos].mean()
        sd = daily[in_pos].std()
        sharpe = mu / sd * np.sqrt(252)
        tstat  = mu / (sd / np.sqrt(n_active))
    else:
        sharpe = float("nan")
        tstat  = float("nan")

    eq = sub["equity"]
    drawdown = float((eq - eq.cummax()).min())

    metrics = {
        "start":            str(sub.index.min().date()),
        "end":              str(sub.index.max().date()),
        "days_total":       len(sub),
        "days_in_position": n_active,
        "frac_in_position": float(in_pos.mean()),
        "num_trades":       int(pos_change.loc[warm:].sum()),
        "total_pnl_gross":  float(sub["pnl_gross"].sum()),
        "total_costs":      float(sub["cost"].sum()),
        "total_pnl_net":    float(sub["pnl_net"].sum()),
        "mean_daily_pnl":   float(daily[in_pos].mean()) if n_active else float("nan"),
        "std_daily_pnl":    float(daily[in_pos].std())  if n_active else float("nan"),
        "sharpe_annual":    sharpe,
        "tstat_mean_pnl":   tstat,
        "hit_rate":         float((daily[in_pos] > 0).mean()) if n_active else float("nan"),
        "max_drawdown_$":   drawdown,
    }
    return df, metrics


# ---------- helpers ----------
def fmt(metrics: dict) -> str:
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:22s} {v:>12.4f}")
        else:
            lines.append(f"  {k:22s} {v}")
    return "\n".join(lines)


def split_chrono(df: pd.DataFrame, frac: float = 0.7
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ---------- main ----------
def main() -> None:
    panel, _ = load_all()

    df_main,    m_main    = run_backtest(panel, use_ovx_filter=True)
    df_control, m_control = run_backtest(panel, use_ovx_filter=False)

    panel_is, panel_oos = split_chrono(panel, 0.7)
    _, m_is  = run_backtest(panel_is,  use_ovx_filter=True)
    _, m_oos = run_backtest(panel_oos, use_ovx_filter=True)

    report = [
        "# WTI calendar spread (F1/F2) - OVX regime filter",
        "",
        "## Strategy",
        f"- Spread        = F2 - F1 (1m, 2m constant-maturity WTI)",
        f"- Direction     = -sign(F2 - F1)  (mean-revert)",
        f"- Regime filter = OVX > trailing {LOOKBACK_DAYS}d "
            f"{int(OVX_PERCENTILE*100)}th pctile (lagged 1d)",
        f"- Re-eval on    = 3rd Wednesday each month",
        f"- Costs         = {COST_BPS_PER_LEG} bps per leg "
            f"per position change",
        f"- Notional      = {NOTIONAL_PER_LEG} bbl per leg",
        "",
        "## Full sample, OVX-filtered",
        fmt(m_main),
        "",
        "## Full sample, ALWAYS-ON control (no OVX filter)",
        fmt(m_control),
        "",
        "## In-sample (first 70%), OVX-filtered",
        fmt(m_is),
        "",
        "## Out-of-sample (last 30%), OVX-filtered",
        fmt(m_oos),
    ]
    txt = "\n".join(report)
    (config.RESULTS_DIR / "backtest_report.txt").write_text(txt)
    df_main.to_parquet(config.RESULTS_DIR / "backtest_main.parquet")
    df_control.to_parquet(config.RESULTS_DIR / "backtest_control.parquet")
    print(txt)

    # ---------- plots ----------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(df_main.index,    df_main["equity"],
                 label="OVX-filtered (net of costs)", lw=1.4)
    axes[0].plot(df_main.index,    df_main["equity_gross"],
                 label="OVX-filtered (gross)", lw=1.0, alpha=0.6)
    axes[0].plot(df_control.index, df_control["equity"],
                 label="Always-on control (net)", lw=1.0, alpha=0.75)
    axes[0].axhline(START_CAPITAL, color="k", ls=":", lw=0.8)
    axes[0].set_ylabel("Equity ($, 1 bbl/leg)")
    axes[0].set_title("WTI calendar spread cumulative P&L")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(df_main.index, df_main["OVX"],     color="C3", lw=0.8,
                 label="OVX")
    axes[1].plot(df_main.index, df_main["ovx_pct"], color="C2", lw=0.9,
                 ls="--",
                 label=f"trailing {LOOKBACK_DAYS}d "
                       f"{int(OVX_PERCENTILE*100)}th pct")
    if df_main["regime"].any():
        ymin = float(df_main["OVX"].min())
        ymax = float(df_main["OVX"].max())
        axes[1].fill_between(df_main.index, ymin, ymax,
                             where=df_main["regime"].fillna(False),
                             alpha=0.12, color="red",
                             label="in regime (raw)")
    axes[1].set_ylabel("OVX")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(df_main.index, df_main["spread"], color="C0", lw=0.6,
                 label="F2 - F1")
    axes[2].axhline(0, color="k", ls=":", lw=0.8)
    axes[2].set_ylabel("F2 - F1 ($)")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="best", fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "backtest_plot.png", dpi=140)
    print(f"\nSaved {config.RESULTS_DIR / 'backtest_plot.png'}")


if __name__ == "__main__":
    main()