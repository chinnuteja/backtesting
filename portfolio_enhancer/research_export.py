# portfolio_enhancer/research_export.py
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import yfinance as yf

# ========================
# Assets & Aux series
# ========================
ASSET_TICKERS = {
    "Equities": "^NSEI",   # Nifty 50
    "Gold":     "GLD",     # Gold proxy
    "REITs":    "VNQ",     # REIT proxy
    "Bitcoin":  "BTC-USD", # BTC
}
AUX_TICKERS_PRIMARY  = {"VIX": "^VIX", "TNX": "^TNX", "DXY": "DX-Y.NYB"}
AUX_TICKERS_FALLBACK = {"DXY": "DXY"}  # alternative symbol for DXY

# ========================
# Portfolio constraints / knobs (same as backtester)
# ========================
FLOORS = {"Equities": 0.00, "Gold": 0.00, "REITs": 0.00, "Bitcoin": 0.00}
CAPS   = {"Equities": 0.50, "Gold": 0.35, "REITs": 0.30, "Bitcoin": 0.20}
TURNOVER_LIMIT = 0.30
EWMA_SPAN      = 63          # daily EWMA span
TILT_ALPHA     = 0.30        # mu * (1 + alpha * KPI), KPI in [-1,1]
HORIZON_DAYS   = 21          # monthly horizon used by the backtester

# ========================
# KPI modules (same ones your backtester uses)
# ========================
from .kpis.gold.gold_kpi_composite import gold_kpi_composite
from .kpis.bitcoin.btc_sentiment import analyze_btc_sentiment
from .kpis.reit import reit_kpi_composite
from .kpis.nifty import analyze_nifty_composite


# ========================
# Utilities
# ========================
def _to_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_localize(None) if x.tzinfo is not None else x
    return pd.Timestamp(x).tz_localize(None)

def _flatten(d: Dict, parent: str = "", sep: str = ".") -> Dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out

def _last_value_on_or_before(s: Optional[pd.Series], asof: pd.Timestamp) -> Optional[float]:
    if s is None:
        return None
    part = s.loc[:asof].dropna()
    return float(part.iloc[-1]) if len(part) else None

def _yf_hist(ticker: str, start: str, end: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)), auto_adjust=True)
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    return df


# ========================
# Data loaders
# ========================
def load_assets_daily(start: str, end: str) -> pd.DataFrame:
    """Return dataframe with daily Close and daily returns for all assets."""
    close_cols = {}
    for name, tk in ASSET_TICKERS.items():
        df = _yf_hist(tk, start, end)
        if df.empty:
            raise RuntimeError(f"No data for {name} ({tk}) between {start} and {end}")
        close_cols[name] = df["Close"].astype(float)
    px = pd.DataFrame(close_cols).sort_index()
    rets = px.pct_change().dropna(how="all").rename(columns=lambda c: f"{c}_ret")
    return px.join(rets)

def load_aux_daily(start: str, end: str) -> pd.DataFrame:
    out = {}
    for name, tk in AUX_TICKERS_PRIMARY.items():
        try:
            df = _yf_hist(tk, start, end)
            if not df.empty:
                out[name] = df["Close"].astype(float)
            else:
                raise RuntimeError("empty")
        except Exception:
            fb = AUX_TICKERS_FALLBACK.get(name)
            if fb:
                try:
                    df = _yf_hist(fb, start, end)
                    if not df.empty:
                        out[name] = df["Close"].astype(float)
                except Exception:
                    pass
    aux = pd.DataFrame(out).sort_index()
    for c in aux.columns:
        aux[f"{c}_chg1"] = aux[c].pct_change()
    return aux


# ========================
# Period helpers
# ========================
def period_dates_monthly(start: str, end: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (decision_date, period_end) pairs for each month fully inside [start, end]."""
    s = _to_ts(start).normalize()
    e = _to_ts(end).normalize()
    months = pd.date_range(s, e, freq="M")
    out = []
    for pe in months:
        dd = pe - pd.offsets.MonthEnd(1)  # previous month end
        if dd >= s and pe <= e:
            out.append((_to_ts(dd), _to_ts(pe)))
    return out

def realized_period_return(series: pd.Series, start_exclusive: pd.Timestamp, end_inclusive: pd.Timestamp) -> float:
    sub = series.loc[(series.index > start_exclusive) & (series.index <= end_inclusive)]
    if sub.empty:
        return 0.0
    return float((1.0 + sub.fillna(0.0)).prod() - 1.0)


# ========================
# Signals, regimes, shocks
# ========================
def ewma_mu_daily(daily_returns: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """EWMA of daily returns (3y lookback) as of decision date."""
    end = asof - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=3)
    cols = [f"{a}_ret" for a in ASSET_TICKERS.keys()]
    sub = daily_returns.loc[(daily_returns.index >= start) & (daily_returns.index <= end), cols]
    if sub.empty:
        return pd.Series(0.0, index=list(ASSET_TICKERS.keys()))
    mu = sub.ewm(span=EWMA_SPAN, adjust=False).mean().iloc[-1]
    mu.index = [c.replace("_ret", "") for c in mu.index]
    return mu.fillna(0.0)

def trailing_vol_ann(daily_returns: pd.DataFrame, asof: pd.Timestamp, window: int = 63) -> pd.Series:
    end = asof - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=3)
    cols = [f"{a}_ret" for a in ASSET_TICKERS.keys()]
    sub = daily_returns.loc[(daily_returns.index >= start) & (daily_returns.index <= end), cols]
    if sub.empty or len(sub) < window:
        return pd.Series(0.0, index=list(ASSET_TICKERS.keys()))
    vol = sub.rolling(window).std().iloc[-1] * np.sqrt(252.0)
    vol.index = [c.replace("_ret", "") for c in vol.index]
    return vol.fillna(0.0)

def apply_kpi_tilt(mu_h: pd.Series, kpis: Dict[str, Dict]) -> pd.Series:
    out = mu_h.copy()
    for k in ASSET_TICKERS.keys():
        score = float((kpis.get(k, {}) or {}).get("composite_sentiment", 0.0) or 0.0)
        out.loc[k] = float(out.loc[k]) * (1.0 + TILT_ALPHA * float(np.clip(score, -1.0, 1.0)))
    return out

def classify_regime(vix_value: Optional[float]) -> str:
    if vix_value is None or np.isnan(vix_value):
        return "neutral"
    if vix_value >= 25.0:      # high fear
        return "risk_off"
    elif vix_value <= 16.0:    # calm
        return "risk_on"
    return "neutral"

def shock_flags(aux_df: pd.DataFrame, asof: pd.Timestamp) -> Dict[str, float]:
    end = asof
    start = end - pd.Timedelta(days=7)
    sub = aux_df.loc[(aux_df.index >= start) & (aux_df.index <= end)]
    def _chg(s: Optional[pd.Series]) -> float:
        if s is None or s.empty: return np.nan
        s = s.dropna()
        if len(s) < 2: return np.nan
        return float((s.iloc[-1] - s.iloc[0]) / s.iloc[0])
    tnx_5d = _chg(sub.get("TNX", pd.Series(dtype=float)))
    dxy_5d = _chg(sub.get("DXY", pd.Series(dtype=float)))
    return {"TNX_5d_chg": tnx_5d, "DXY_5d_chg": dxy_5d}


# ========================
# Robust, silent KPI callers (monthly cadence)
# ========================
async def _silent_call(factory: Callable[[], object], timeout: float = 60.0) -> dict:
    """
    Capture prints during factory() and while awaiting the coroutine (if any).
    Returns dict with 'composite_sentiment' on success, neutral 0.0 on failure.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = factory()
        if asyncio.iscoroutine(res):
            with contextlib.redirect_stdout(buf):
                res = await asyncio.wait_for(res, timeout=timeout)
        if isinstance(res, dict):
            return res
        return {"composite_sentiment": float(res) if res is not None else 0.0}
    except Exception:
        return {"composite_sentiment": 0.0}

async def kpi_equities(asof: str, H: str) -> Dict:
    for fac in (lambda: analyze_nifty_composite(asof, H),
                lambda: analyze_nifty_composite(asof)):
        out = await _silent_call(fac)
        if out is not None:
            return out
    return {"composite_sentiment": 0.0}

async def kpi_gold(asof: str, H: str) -> Dict:
    for fac in (lambda: gold_kpi_composite(asof, H),
                lambda: gold_kpi_composite(asof)):
        out = await _silent_call(fac)
        if out is not None:
            return out
    return {"composite_sentiment": 0.0}

async def kpi_reit(asof: str, H: str) -> Dict:
    return await _silent_call(lambda: reit_kpi_composite(asof, H))

async def kpi_btc(asof: str, H: str) -> Dict:
    return await _silent_call(lambda: analyze_btc_sentiment(backtest_date=asof, horizon=H, historical_cutoff=asof))


# ========================
# Optimizer helpers (same as backtester)
# ========================
def project_capped_simplex(w_raw: Dict[str, float]) -> Dict[str, float]:
    keys = list(ASSET_TICKERS.keys())
    lo = np.array([FLOORS.get(k, 0.0) for k in keys], dtype=float)
    hi = np.array([CAPS.get(k, 1.0)   for k in keys], dtype=float)
    x0 = np.array([max(0.0, float(w_raw.get(k, 0.0))) for k in keys], dtype=float)
    if lo.sum() > 1.0:
        lo = lo / lo.sum()
    x0 = np.minimum(hi, np.maximum(lo, x0))
    if abs(float(x0.sum()) - 1.0) <= 1e-12:
        return {k: float(v) for k, v in zip(keys, x0)}
    tau_lo = (x0 - hi).min() - 1.0
    tau_hi = (x0 - lo).max() + 1.0
    for _ in range(80):
        tau = 0.5 * (tau_lo + tau_hi)
        x = np.clip(x0 - tau, lo, hi)
        s = float(x.sum())
        if s > 1.0: tau_lo = tau
        else:       tau_hi = tau
    x = np.clip(x0 - 0.5*(tau_lo+tau_hi), lo, hi)
    x = x / max(1e-12, x.sum())
    return {k: float(v) for k, v in zip(keys, x)}

def apply_turnover(prev_w: Optional[Dict[str, float]], new_w: Dict[str, float]) -> Dict[str, float]:
    if prev_w is None:
        return new_w
    keys = list(ASSET_TICKERS.keys())
    p = np.array([prev_w[k] for k in keys], dtype=float)
    n = np.array([new_w[k]  for k in keys], dtype=float)
    diff = n - p
    l1 = float(np.abs(diff).sum())
    if l1 <= TURNOVER_LIMIT + 1e-12:
        return new_w
    scale = TURNOVER_LIMIT / max(1e-12, l1)
    adj = p + diff * scale
    adj = adj / max(1e-12, adj.sum())
    out = {k: float(v) for k, v in zip(keys, adj)}
    return project_capped_simplex(out)


# ========================
# One monthly decision (for export)
# ========================
async def build_month_row(daily_assets: pd.DataFrame,
                          aux_daily: pd.DataFrame,
                          decision_date: pd.Timestamp,
                          period_end: pd.Timestamp,
                          prev_w: Optional[Dict[str, float]]) -> Tuple[Dict, Dict]:
    # Trend & vol (as-of decision date)
    muD  = ewma_mu_daily(daily_assets, decision_date)
    muH  = muD * HORIZON_DAYS
    volA = trailing_vol_ann(daily_assets, decision_date)

    denom = volA.replace(0.0, np.nan)
    sigRA_pre = (muH / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # KPIs (as-of decision date; monthly cadence)
    asof = decision_date.date().isoformat()
    eq, gd, rt, bc = await asyncio.gather(
        kpi_equities(asof, "M"), kpi_gold(asof, "M"), kpi_reit(asof, "M"), kpi_btc(asof, "M")
    )
    kpis = {"Equities": eq, "Gold": gd, "REITs": rt, "Bitcoin": bc}

    muTilt = apply_kpi_tilt(muH, kpis)
    sigRA_post = (muTilt / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Convert positive tilted mu to raw long-only weights
    mu_vec = muTilt.reindex(ASSET_TICKERS.keys()).fillna(0.0).to_numpy()
    if (mu_vec > 0).any():
        w_raw = mu_vec.clip(min=0.0); w_raw = w_raw / w_raw.sum()
    else:
        w_raw = np.ones_like(mu_vec) / len(mu_vec)
    w0 = {k: float(v) for k, v in zip(ASSET_TICKERS.keys(), w_raw)}

    # Constraints and turnover
    w_proj  = project_capped_simplex(w0)
    w_final = apply_turnover(prev_w, w_proj)

    # Expected portfolio (tilted horizon mu · weights)
    exp_port = float(np.dot(
        muTilt.reindex(ASSET_TICKERS.keys()).values,
        np.array([w_final[k] for k in ASSET_TICKERS.keys()])
    ))

    # Realized for the coming month (grade)
    realized_by_asset = {
        a: realized_period_return(daily_assets[f"{a}_ret"], decision_date, period_end)
        for a in ASSET_TICKERS.keys()
    }
    real_port = float(sum(w_final[a] * realized_by_asset[a] for a in ASSET_TICKERS.keys()))

    # Regime & shocks
    vix_val = _last_value_on_or_before(aux_daily.get("VIX", pd.Series(dtype=float)), decision_date)
    regime  = classify_regime(vix_val)
    shocks  = shock_flags(aux_daily, decision_date)

    row = {
        "decision_date": decision_date.date().isoformat(),
        "period_end":    period_end.date().isoformat(),
        "regime": regime, "VIX": vix_val,
        **shocks,
        # weights
        **{f"w_{a}": w_final[a] for a in ASSET_TICKERS.keys()},
        # portfolio exp & realized
        "exp_portfolio": exp_port,
        **{f"real_{a}": realized_by_asset[a] for a in ASSET_TICKERS.keys()},
        "real_portfolio": real_port,
        # signals
        **{f"muD_{a}": float(muD.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"muH_{a}": float(muH.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"volAnn_{a}": float(volA.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"sigRA_pre_{a}": float(sigRA_pre.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"muTilt_{a}": float(muTilt.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"sigRA_post_{a}": float(sigRA_post.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        # KPI composites (flatten to a single column each)
        "kpi_equities": float(eq.get("composite_sentiment", 0.0) or 0.0),
        "kpi_gold":     float(gd.get("composite_sentiment", 0.0) or 0.0),
        "kpi_reit":     float(rt.get("composite_sentiment", 0.0) or 0.0),
        "kpi_btc":      float(bc.get("composite_sentiment", 0.0) or 0.0),
    }
    return row, w_final


# ========================
# Writers
# ========================
def write_table(df: pd.DataFrame, out_file: str):
    if out_file.lower().endswith(".xlsx"):
        # XLSX with basic formatting
        try:
            writer = pd.ExcelWriter(out_file, engine="xlsxwriter")
            engine = "xlsxwriter"
        except Exception:
            writer = pd.ExcelWriter(out_file, engine="openpyxl")
            engine = "openpyxl"
        with writer as xw:
            df.to_excel(xw, sheet_name="DECISIONS_MONTHLY", index=False)
            if engine == "xlsxwriter":
                wb = xw.book
                fmt_pct = wb.add_format({"num_format": "0.00%"})
                fmt_3dp = wb.add_format({"num_format": "0.000"})
                ws = xw.sheets["DECISIONS_MONTHLY"]
                for j, col in enumerate(df.columns):
                    name = str(col)
                    if name.startswith(("w_", "real_", "exp_portfolio")):
                        ws.set_column(j, j, 13, fmt_pct)
                    elif name.startswith(("muD_", "muH_", "volAnn_", "sigRA_", "muTilt_")) or name.startswith("kpi_"):
                        ws.set_column(j, j, 14, fmt_3dp)
                    else:
                        ws.set_column(j, j, 18)
    else:
        df.to_csv(out_file, index=False)


# ========================
# CLI
# ========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monthly research export (one row per month, backtester-grade features & KPI composites)."
    )
    p.add_argument("--start", type=str, default="2015-01-01", help="YYYY-MM-DD")
    p.add_argument("--end",   type=str, default="2024-12-31", help="YYYY-MM-DD")
    p.add_argument("--out",   type=str, default="Research_Monthly.csv", help="CSV or XLSX filename")
    return p.parse_args()


async def run_monthly_export(start: str, end: str, out: str):
    # Load data once
    daily_assets = load_assets_daily(start, end)
    aux_daily    = load_aux_daily(start, end)

    rows: List[Dict] = []
    prev_w: Optional[Dict[str, float]] = None

    for dd, pe in period_dates_monthly(start, end):
        row, prev_w = await build_month_row(daily_assets, aux_daily, dd, pe, prev_w)
        rows.append(row)

    df = pd.DataFrame(rows)
    write_table(df, out)

    print(f"✅ Monthly research file written: {out}")
    print(f"Rows: {len(df)}  |  Period: {start} → {end}")


def main():
    args = parse_args()
    asyncio.run(run_monthly_export(args.start, args.end, args.out))


if __name__ == "__main__":
    main()
