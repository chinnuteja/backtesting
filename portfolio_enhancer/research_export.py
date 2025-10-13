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

# ===== Assets & Aux series used in the model =====
ASSET_TICKERS = {
    "Equities": "^NSEI",   # Nifty 50
    "Gold":     "GLD",     # SPDR Gold Shares
    "REITs":    "VNQ",     # US REIT proxy
    "Bitcoin":  "BTC-USD",
}
AUX_TICKERS_PRIMARY = {"VIX": "^VIX", "TNX": "^TNX", "DXY": "DX-Y.NYB"}
AUX_TICKERS_FALLBACK = {"DXY": "DXY"}  # sometimes Yahoo exposes "DXY"

# ===== Portfolio constraints (same as backtester) =====
FLOORS = {"Equities": 0.00, "Gold": 0.00, "REITs": 0.00, "Bitcoin": 0.00}
CAPS   = {"Equities": 0.50, "Gold": 0.35, "REITs": 0.30, "Bitcoin": 0.20}
TURNOVER_LIMIT = 0.30
EWMA_SPAN = 63
TILT_ALPHA = 0.30  # μ * (1 + α*KPI), KPI clipped to [-1,1]

# ===== KPI modules (same ones used by backtester) =====
from .kpis.gold.gold_kpi_composite import gold_kpi_composite
from .kpis.bitcoin.btc_sentiment import analyze_btc_sentiment
from .kpis.reit import reit_kpi_composite
from .kpis.nifty import analyze_nifty_composite


# ---------------- Utilities ----------------
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


# ---------------- Data loaders ----------------
def _yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)), auto_adjust=True)
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    return df

def load_assets_daily(start: str, end: str) -> pd.DataFrame:
    close_cols = {}
    for name, tk in ASSET_TICKERS.items():
        df = _yf(tk, start, end)
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
            df = _yf(tk, start, end)
            if not df.empty:
                out[name] = df["Close"].astype(float)
            else:
                raise RuntimeError("empty")
        except Exception:
            fb = AUX_TICKERS_FALLBACK.get(name)
            if fb:
                try:
                    df = _yf(fb, start, end)
                    if not df.empty:
                        out[name] = df["Close"].astype(float)
                except Exception:
                    pass
    aux = pd.DataFrame(out).sort_index()
    for c in aux.columns:
        aux[f"{c}_chg1"] = aux[c].pct_change()
    return aux


# ---------------- Period helpers ----------------
def period_dates_monthly(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for m in range(1, 13):
        pe = _to_ts(f"{year}-{m:02d}-01") + pd.offsets.MonthEnd(0)
        dd = pe - pd.offsets.MonthEnd(1)
        out.append((_to_ts(dd), _to_ts(pe)))
    return out

def period_dates_quarterly(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    q_ends = [f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
    out = []
    for e in q_ends:
        pe = _to_ts(e)
        dd = pe - pd.offsets.QuarterEnd(1)
        out.append((_to_ts(dd), _to_ts(pe)))
    return out

def horizon_multiplier(letter: str) -> int:
    return {"M": 21, "Q": 63, "H": 126, "Y": 252}.get(letter, 21)

def realized_period_return(series: pd.Series, start_exclusive: pd.Timestamp, end_inclusive: pd.Timestamp) -> float:
    sub = series.loc[(series.index > start_exclusive) & (series.index <= end_inclusive)]
    if sub.empty:
        return 0.0
    return float((1.0 + sub.fillna(0.0)).prod() - 1.0)


# ---------------- Signals, regimes, shocks ----------------
def ewma_mu_daily(daily_df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """EWMA of daily returns (3y lookback) as of decision date."""
    end = asof - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=3)
    cols = [f"{a}_ret" for a in ASSET_TICKERS.keys()]
    sub = daily_df.loc[(daily_df.index >= start) & (daily_df.index <= end), cols]
    if sub.empty:
        return pd.Series(0.0, index=list(ASSET_TICKERS.keys()))
    mu = sub.ewm(span=EWMA_SPAN, adjust=False).mean().iloc[-1]
    mu.index = [c.replace("_ret", "") for c in mu.index]
    return mu.fillna(0.0)

def trailing_vol_ann(daily_df: pd.DataFrame, asof: pd.Timestamp, window: int = 63) -> pd.Series:
    end = asof - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=3)
    cols = [f"{a}_ret" for a in ASSET_TICKERS.keys()]
    sub = daily_df.loc[(daily_df.index >= start) & (daily_df.index <= end), cols]
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


# ---------------- Silence-anything KPI wrapper ----------------
async def _silent_call(factory: Callable[[], object], timeout: float = 60.0) -> dict:
    """
    Capture prints both during factory() and while awaiting the coroutine (if any).
    Returns dict with 'composite_sentiment' on success, neutral 0.0 on failure.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = factory()
        # If coroutine, await with prints captured too
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


# ---------------- Optimizer helpers ----------------
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


# ---------------- Core forecasting (one period) ----------------
async def forecast_one_period(daily_assets: pd.DataFrame,
                              aux_daily: pd.DataFrame,
                              decision_date: pd.Timestamp,
                              period_end: pd.Timestamp,
                              prev_w: Optional[Dict[str, float]],
                              H: str  # 'M' or 'Q'
                              ) -> Tuple[Dict, Dict[str, float]]:
    mu_d   = ewma_mu_daily(daily_assets, decision_date)
    mu_h   = mu_d * horizon_multiplier(H)
    vol_a  = trailing_vol_ann(daily_assets, decision_date)

    # risk-adjust (pre-tilt)
    denom = vol_a.replace(0.0, np.nan)
    sig_ra_pre = (mu_h / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # KPIs
    asof = decision_date.date().isoformat()
    eq, gd, rt, bc = await asyncio.gather(
        kpi_equities(asof, H), kpi_gold(asof, H), kpi_reit(asof, H), kpi_btc(asof, H)
    )
    kpis = {"Equities": eq, "Gold": gd, "REITs": rt, "Bitcoin": bc}

    mu_tilt = apply_kpi_tilt(mu_h, kpis)
    sig_ra_post = (mu_tilt / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Raw weights from positive tilted μ
    mu_vec = mu_tilt.reindex(ASSET_TICKERS.keys()).fillna(0.0).to_numpy()
    if (mu_vec > 0).any():
        w_raw = mu_vec.clip(min=0.0); w_raw = w_raw / w_raw.sum()
    else:
        w_raw = np.ones_like(mu_vec) / len(mu_vec)
    w0 = {k: float(v) for k, v in zip(ASSET_TICKERS.keys(), w_raw)}

    # Constraints
    w_proj  = project_capped_simplex(w0)
    w_final = apply_turnover(prev_w, w_proj)

    # Portfolio expected & realized
    exp_port = float(np.dot(
        mu_tilt.reindex(ASSET_TICKERS.keys()).values,
        np.array([w_final[k] for k in ASSET_TICKERS.keys()])
    ))

    realized_by_asset = {}
    for a in ASSET_TICKERS.keys():
        realized_by_asset[a] = realized_period_return(
            daily_assets[f"{a}_ret"], decision_date, period_end
        )
    real_port = float(sum(w_final[a] * realized_by_asset[a] for a in ASSET_TICKERS.keys()))

    # Regime & shocks
    vix_val = _last_value_on_or_before(aux_daily.get("VIX", pd.Series(dtype=float)), decision_date)
    regime = classify_regime(vix_val)
    shocks = shock_flags(aux_daily, decision_date)

    # Flatten KPIs
    eq_f, gd_f, rt_f, bc_f = _flatten(eq), _flatten(gd), _flatten(rt), _flatten(bc)

    row = {
        "decision_date": decision_date.date().isoformat(),
        "period_end": period_end.date().isoformat(),
        "horizon": H,
        "regime": regime, "VIX": vix_val,
        **shocks,
        # weights
        **{f"w_{a}": w_final[a] for a in ASSET_TICKERS.keys()},
        # portfolio exp & realized
        "exp_portfolio": exp_port,
        **{f"real_{a}": realized_by_asset[a] for a in ASSET_TICKERS.keys()},
        "real_portfolio": real_port,
        # signals
        **{f"muD_{a}": float(mu_d.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"muH_{a}": float(mu_h.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"volAnn_{a}": float(vol_a.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"sigRA_pre_{a}": float(sig_ra_pre.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"muTilt_{a}": float(mu_tilt.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        **{f"sigRA_post_{a}": float(sig_ra_post.get(a, 0.0)) for a in ASSET_TICKERS.keys()},
        # KPI components, flattened
        **{f"eq.{k}": v for k, v in eq_f.items()},
        **{f"gold.{k}": v for k, v in gd_f.items()},
        **{f"reit.{k}": v for k, v in rt_f.items()},
        **{f"btc.{k}": v for k, v in bc_f.items()},
    }
    return row, w_final


# ---------------- Excel writer ----------------
def save_excel(daily_assets: pd.DataFrame, aux_daily: pd.DataFrame,
               monthly_df: pd.DataFrame, quarterly_df: pd.DataFrame,
               out_file: str, year: int):
    da = daily_assets.copy(); da.index.name = "date"
    aux = aux_daily.copy();   aux.index.name = "date"

    try:
        writer = pd.ExcelWriter(out_file, engine="xlsxwriter")
        use_xlsxwriter = True
    except Exception:
        writer = pd.ExcelWriter(out_file, engine="openpyxl")
        use_xlsxwriter = False

    with writer as xw:
        da.to_excel(xw, sheet_name="DAILY_ASSETS")
        aux.to_excel(xw, sheet_name="DAILY_AUX")
        monthly_df.to_excel(xw, sheet_name=f"DECISIONS_MONTHLY_{year}", index=False)
        quarterly_df.to_excel(xw, sheet_name=f"DECISIONS_QUARTERLY_{year}", index=False)

        if use_xlsxwriter:
            wb = xw.book
            fmt_pct = wb.add_format({"num_format": "0.00%"})
            fmt_3dp = wb.add_format({"num_format": "0.000"})
            # DAILY_ASSETS
            ws = xw.sheets["DAILY_ASSETS"]
            for i, c in enumerate(da.columns):
                if c.endswith("_ret"): ws.set_column(i+1, i+1, 12, fmt_pct)
                else:                  ws.set_column(i+1, i+1, 12)
            # DAILY_AUX
            ws = xw.sheets["DAILY_AUX"]
            for i, c in enumerate(aux.columns):
                if c.endswith("_chg1"): ws.set_column(i+1, i+1, 12, fmt_pct)
                else:                   ws.set_column(i+1, i+1, 12)
            # DECISIONS_* (best-effort formatting)
            for name, df in [(f"DECISIONS_MONTHLY_{year}", monthly_df),
                             (f"DECISIONS_QUARTERLY_{year}", quarterly_df)]:
                ws = xw.sheets[name]
                for j, col in enumerate(df.columns):
                    if col.startswith(("w_", "real_", "exp_portfolio")):
                        ws.set_column(j, j, 12, fmt_pct)
                    elif col.startswith(("muD_", "muH_", "volAnn_", "sigRA_", "muTilt_")):
                        ws.set_column(j, j, 12, fmt_3dp)
                    else:
                        ws.set_column(j, j, 18)


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export all datapoints used by the model (assets, aux, monthly & quarterly decisions)."
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--lookback-years", type=int, default=5,
                   help="years of history to load before Jan 1 of the target year")
    p.add_argument("--start", type=str, default=None, help="override start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="override end date (YYYY-MM-DD)")
    p.add_argument("--out", type=str, default=None, help="Excel filename (default: Research_<year>_full.xlsx)")
    return p.parse_args()

def main():
    args = parse_args()
    # FIX: argparse uses underscore for hyphenated options
    start = args.start or f"{args.year - args.lookback_years}-01-01"
    end   = args.end   or f"{args.year}-12-31"
    out   = args.out   or f"Research_{args.year}_full.xlsx"

    # Load data with sufficient history (fixes cold-start/empty windows)
    daily_assets = load_assets_daily(start, end)
    aux_daily    = load_aux_daily(start, end)

    # ---- Monthly decisions
    rows_m: List[Dict] = []
    prev_w_m: Optional[Dict[str, float]] = None
    for dd, pe in period_dates_monthly(args.year):
        row, prev_w_m = asyncio.run(
            forecast_one_period(daily_assets, aux_daily, dd, pe, prev_w_m, "M")
        )
        rows_m.append(row)
    df_m = pd.DataFrame(rows_m)

    # ---- Quarterly decisions
    rows_q: List[Dict] = []
    prev_w_q: Optional[Dict[str, float]] = None
    for dd, pe in period_dates_quarterly(args.year):
        row, prev_w_q = asyncio.run(
            forecast_one_period(daily_assets, aux_daily, dd, pe, prev_w_q, "Q")
        )
        rows_q.append(row)
    df_q = pd.DataFrame(rows_q)

    save_excel(daily_assets, aux_daily, df_m, df_q, out, args.year)

    print(f"✅ Research pack written to: {out}")
    print(f"Sheets:\n  DAILY_ASSETS                  {len(daily_assets):,} rows"
          f"\n  DAILY_AUX                     {len(aux_daily):,} rows"
          f"\n  DECISIONS_MONTHLY_{args.year}   {len(df_m)} rows"
          f"\n  DECISIONS_QUARTERLY_{args.year} {len(df_q)} rows")

if __name__ == "__main__":
    main()
