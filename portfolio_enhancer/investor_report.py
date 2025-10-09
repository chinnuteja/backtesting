# investor_report.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional benchmark fetch via yfinance (falls back gracefully if not available)
try:
    import yfinance as yf
    _YF = True
except Exception:
    _YF = False


# -----------------------------
# Utilities
# -----------------------------
def _to_dt_index(obj: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    return out

def _prod_minus_one(x: pd.Series) -> float:
    return float((1.0 + x).prod() - 1.0)

def _pct_axis(ax):
    ax.yaxis.set_major_formatter(lambda y, pos: f"{100*y:.0f}%")

def _ensure_2023(series_or_df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    out = _to_dt_index(series_or_df)
    mask = (out.index >= "2023-01-01") & (out.index <= "2023-12-31")
    return out.loc[mask]

def _infer_strategy_series(daily_returns: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """
    Accepts:
      - pd.Series of daily returns
      - pd.DataFrame with a single 'strategy'/'portfolio' column or the first column as strategy
    Returns:
      - pd.Series of daily returns
    """
    if isinstance(daily_returns, pd.Series):
        s = daily_returns.copy()
    else:
        cols = list(daily_returns.columns)
        if "strategy" in cols:
            s = daily_returns["strategy"].copy()
        elif "portfolio" in cols:
            s = daily_returns["portfolio"].copy()
        else:
            s = daily_returns.iloc[:, 0].copy()
    s = _ensure_2023(pd.Series(s))
    s.name = "strategy"
    return s.fillna(0.0)

def _fetch_nifty_benchmark_2023() -> Optional[pd.Series]:
    """Fetch ^NSEI daily % returns for 2023; return None if unavailable."""
    if not _YF:
        return None
    try:
        tkr = yf.Ticker("^NSEI")
        df = tkr.history(start="2023-01-01", end="2023-12-31")
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        ret = df["Close"].pct_change().dropna()
        ret.name = "nifty"
        return ret
    except Exception:
        return None

def _grow_to_value(returns: pd.Series, initial_value: float = 10_000.0) -> pd.Series:
    """Convert a return series into a value series starting at initial_value."""
    curve = (1.0 + returns).cumprod()
    return pd.Series(initial_value * curve, index=returns.index, name="value")

def _monthly_from_daily(daily: pd.Series) -> pd.Series:
    """Aggregate daily returns to month-end returns."""
    return daily.resample("M").apply(_prod_minus_one)

def _compute_drawdown(daily: pd.Series) -> pd.Series:
    """Compute drawdown series from daily returns."""
    curve = (1.0 + daily).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    dd.name = "drawdown"
    return dd

def _weights_df_from_decisions(decisions: List[Dict]) -> pd.DataFrame:
    """
    Build a monthly weights DataFrame (index = month-end datetimes) from backtester decisions.
    Each decision item is expected to have either:
      - 'period_end' (YYYY-MM-DD) or 'date' (string) or a datetime-like
      - 'weights' = {'Equities': x, 'Gold': y, 'REITs': z, 'Bitcoin': w}
    """
    rows = []
    for d in decisions:
        # date/period parsing
        dt_raw = d.get("period_end") or d.get("date") or d.get("decision_date")
        dt = pd.to_datetime(dt_raw).tz_localize(None)
        w = d.get("weights", {}) or {}
        rows.append({
            "date": dt,
            "Equities": float(w.get("Equities", 0.0)),
            "Gold":     float(w.get("Gold", 0.0)),
            "REITs":    float(w.get("REITs", 0.0)),
            "Bitcoin":  float(w.get("Bitcoin", 0.0)),
        })
    if not rows:
        raise ValueError("No monthly decisions found to build weights chart.")
    df = pd.DataFrame(rows).set_index("date").sort_index()
    # Focus on 2023 and snap to month-end for consistency
    df = _ensure_2023(df)
    df.index = df.index.to_period("M").to_timestamp("M")  # ensure month-end stamps
    # Clip and renormalize (safety)
    df = df.clip(lower=0.0)
    s = df.sum(axis=1)
    s = s.replace(0.0, np.nan)
    df = df.div(s, axis=0).fillna(df)  # if already sums to 1, unchanged
    return df


# -----------------------------
# Chart 1: Cumulative Performance
# -----------------------------
def plot_equity_curve_2023(
    strategy_daily: pd.Series,
    output_path: str,
    benchmark_daily: Optional[pd.Series] = None,
    title: str = "2023 Strategy Performance vs. Benchmark",
):
    strategy_daily = _ensure_2023(strategy_daily)
    if benchmark_daily is None:
        benchmark_daily = _fetch_nifty_benchmark_2023()

    # Align calendar
    if benchmark_daily is not None and not benchmark_daily.empty:
        both = pd.concat([strategy_daily.rename("strategy"), benchmark_daily.rename("benchmark")], axis=1).dropna()
        s = both["strategy"]
        b = both["benchmark"]
    else:
        s = strategy_daily
        b = None

    s_val = _grow_to_value(s, initial_value=10_000.0)
    b_val = _grow_to_value(b, initial_value=10_000.0) if b is not None else None

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(s_val.index, s_val.values, linewidth=2, label="Monthly Rebalancing Strategy")
    if b_val is not None:
        ax.plot(b_val.index, b_val.values, linewidth=2, label="Nifty 50 (^NSEI)")

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Chart 2: Dynamic Asset Allocation (stacked area)
# -----------------------------
def plot_allocation_2023(
    weights_monthly: pd.DataFrame,
    output_path: str,
    title: str = "Asset Allocation Over Time (Monthly Rebalancing)",
):
    # Expect monthly index at month-end
    w = weights_monthly.copy()
    # Ensure only 2023 and correct order
    w = _ensure_2023(w)
    w = w[["Equities", "Gold", "REITs", "Bitcoin"]].fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.stackplot(w.index, w.T.values, labels=["Equities", "Gold", "REITs", "Bitcoin"], alpha=0.95)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Portfolio Weight (%)")
    ax.set_xlabel("2023")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{100*y:.0f}%")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Chart 3: Monthly Returns + Drawdown
# -----------------------------
def plot_risk_charts_2023(
    strategy_daily: pd.Series,
    output_path: str,
    title_top: str = "Monthly Returns",
    title_bottom: str = "Portfolio Drawdown",
):
    s = _ensure_2023(strategy_daily).fillna(0.0)
    monthly = _monthly_from_daily(s)

    dd = _compute_drawdown(s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.0), sharex=False, gridspec_kw={"height_ratios": [2, 1.5]})

    # Top: monthly returns bars
    ax1.bar(monthly.index, monthly.values, width=15)
    ax1.set_title(title_top, fontsize=13)
    _pct_axis(ax1)
    ax1.grid(True, alpha=0.25)

    # Bottom: drawdown area
    ax2.fill_between(dd.index, dd.values, 0.0, step=None, alpha=0.35)
    ax2.set_title(title_bottom, fontsize=13)
    _pct_axis(ax2)
    ax2.grid(True, alpha=0.25)

    # Labels
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Main entry
# -----------------------------
def generate_investor_report(
    results: Dict,
    output_prefix: str = "performance_summary",
    override_benchmark_daily: Optional[pd.Series] = None,
) -> Dict[str, str]:
    """
    Parameters
    ----------
    results : dict
        Expected keys:
          - 'decisions': List[{'period_end'/'date': ..., 'weights': {'Equities','Gold','REITs','Bitcoin'}}]
          - 'daily_returns': pd.Series or pd.DataFrame (strategy daily returns for 2023)
          - 'summary': Dict (not used by charts but accepted)
    output_prefix : str
        Base filename for saved charts (without extension)
    override_benchmark_daily : Optional[pd.Series]
        If you already have daily benchmark returns, pass them here (will skip yfinance fetch).

    Returns
    -------
    dict with file paths of the three charts.
    """
    # 1) Daily strategy returns
    daily = results.get("daily_returns")
    if daily is None:
        raise ValueError("results['daily_returns'] is required (pd.Series or pd.DataFrame).")
    daily_s = _infer_strategy_series(daily)

    # 2) Monthly weights from decisions
    decisions = results.get("decisions") or []
    weights_m = _weights_df_from_decisions(decisions)

    # 3) Benchmark daily (optional override)
    bench = None
    if isinstance(override_benchmark_daily, pd.Series):
        bench = _ensure_2023(override_benchmark_daily.dropna())

    # Make sure output directory exists
    out_files = {}
    base = output_prefix

    # Chart 1
    f1 = f"{base}_equity_curve.png"
    plot_equity_curve_2023(daily_s, f1, benchmark_daily=bench)
    out_files["equity_curve"] = os.path.abspath(f1)

    # Chart 2
    f2 = f"{base}_allocations.png"
    plot_allocation_2023(weights_m, f2)
    out_files["allocations"] = os.path.abspath(f2)

    # Chart 3
    f3 = f"{base}_risk.png"
    plot_risk_charts_2023(daily_s, f3)
    out_files["risk"] = os.path.abspath(f3)

    return out_files


# -----------------------------
# Example (comment out in production)
# -----------------------------
if __name__ == "__main__":
    # Minimal mock to illustrate usage. Replace with your real `results` object.
    rng = pd.date_range("2023-01-02", "2023-12-29", freq="B")
    mock_daily = pd.Series(np.random.normal(0.0004, 0.008, len(rng)), index=rng, name="strategy")

    mock_decisions = []
    for m in range(1, 13):
        dt = pd.Timestamp(f"2023-{m:02d}-01") + pd.offsets.MonthEnd(0)
        w = {"Equities": 0.4, "Gold": 0.3, "REITs": 0.2, "Bitcoin": 0.1}
        mock_decisions.append({"period_end": dt, "weights": w})

    mock_results = {
        "daily_returns": mock_daily,
        "decisions": mock_decisions,
        "summary": {"total_return": 0.25, "ann_vol": 0.16, "sharpe": 1.5},
    }

    files = generate_investor_report(mock_results, output_prefix="performance_summary")
    print("Saved charts:", files)
