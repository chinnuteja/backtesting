# portfolio_enhancer/backtester.py
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf

# --- External KPI modules (async) ---
from .kpis.gold.gold_kpi_composite import gold_kpi_composite
from .kpis.bitcoin.btc_sentiment import analyze_btc_sentiment
from .kpis.reit import reit_kpi_composite
from .kpis.nifty import analyze_nifty_composite

# Optional investor visuals (if you have investor_report.py)
from .investor_report import generate_investor_report


# =========================
# Global config
# =========================
START_DATE = "2015-01-01"
END_DATE   = "2023-12-31"

TICKERS = {
    "Equities": "^NSEI",
    "Gold":     "GLD",
    "REITs":    "VNQ",
    "Bitcoin":  "BTC-USD",
}

FLOORS_BASE = {"Equities": 0.00, "Gold": 0.00, "REITs": 0.00, "Bitcoin": 0.00}
CAPS_BASE   = {"Equities": 0.50, "Gold": 0.35, "REITs": 0.30, "Bitcoin": 0.20}
TURNOVER_LIMIT = 0.30  # L1 per decision

# ---------- SIGNAL UPGRADE ----------
# Multi-horizon momentum spans
MH_SPANS = [21, 63, 126]

# Regime-specific horizon weights
MH_WEIGHTS_BY_REGIME = {
    "risk_on":   np.array([0.50, 0.35, 0.15]),
    "neutral":   np.array([0.30, 0.40, 0.30]),
    "risk_off":  np.array([0.15, 0.35, 0.50]),
}

VOL_WIN = 63  # trailing vol window for risk-adjusted signal

# Regime gating via VIX + KPI alpha + regime caps
VIX_THRESHOLDS = {"risk_on_max": 15.0, "risk_off_min": 25.0}
TILT_ALPHA = {"risk_on": 0.15, "neutral": 0.22, "risk_off": 0.35}

CAPS_BY_REGIME = {
    "risk_on":  {"Equities": 0.60, "Gold": 0.30, "REITs": 0.35, "Bitcoin": 0.20},
    "neutral":  CAPS_BASE,
    "risk_off": {"Equities": 0.45, "Gold": 0.45, "REITs": 0.25, "Bitcoin": 0.15},
}

# Shock detector (global rate / dollar jumps)
SHOCK_WINDOW_D = 10
TNX_ABS_BPS_SHOCK = 45.0
DXY_PCT_SHOCK    = 2.5
SHOCK_RISK_MULT  = 0.92
SHOCK_GOLD_MULT  = 1.05

# Calibration (expected-return print)
CALIB_MONTHS_BACK = 24       # history length for learning bins / slopes
MIN_REGIME_SAMPLES = 12      # if fewer, fallback to pooled across regimes
RIDGE_L2 = 0.5               # small ridge for slope fallback
SLOPE_BOUNDS = (0.25, 2.0)   # slope clamp if we must use regression
BIN_Q = [0.0, .2, .4, .6, .8, 1.0]  # quintiles
EXP_CLIP_MONTHLY = 0.15      # cap extreme monthly printed exp to +/-15%

# Output controls
PRINT_MODE = "summary"
SHOW_KPI_SUMMARY = True
SILENCE_KPI_LOGS = True
PRINT_LAST_N_MONTHS = 0
KPI_TIMEOUT_SEC = 60

np.set_printoptions(suppress=True, linewidth=120)


# =========================
# Utils
# =========================
def _to_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_localize(None) if x.tzinfo is not None else x
    return pd.Timestamp(x).tz_localize(None)

def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:+.2f}%"

def _fmt_w_short(w: Dict[str, float]) -> str:
    return "E={:.1f}% G={:.1f}% R={:.1f}% B={:.1f}%".format(
        100*w["Equities"], 100*w["Gold"], 100*w["REITs"], 100*w["Bitcoin"]
    )

def _hletter(h: str) -> str:
    return {"monthly":"M","quarterly":"Q","half":"H","yearly":"Y"}.get(h,"M")

def _should_print_period(i: int, total: int) -> bool:
    if PRINT_MODE == "quiet": return False
    if PRINT_LAST_N_MONTHS and (total - i) > PRINT_LAST_N_MONTHS: return False
    return True

async def _await_quietly(coro):
    if not SILENCE_KPI_LOGS: return await coro
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return await coro

async def _safe_kpi(label: str, coro) -> Dict:
    try:
        return await asyncio.wait_for(_await_quietly(coro), timeout=KPI_TIMEOUT_SEC)
    except Exception as e:
        if PRINT_MODE != "quiet":
            print(f"[WARN] {label} KPI failed: {e!s} — using neutral 0.0")
        return {"composite_sentiment": 0.0, "components": {}}


# =========================
# Data
# =========================
def _fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    if PRINT_MODE == "full":
        print(f"Fetching {ticker} {start} → {end}")
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1), auto_adjust=True)
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    return df

def load_returns_fresh() -> pd.DataFrame:
    series = []
    for name, tk in TICKERS.items():
        df = _fetch_history(tk, START_DATE, END_DATE)
        s = df["Close"].pct_change().dropna().rename(name)
        series.append(s)
    out = pd.concat(series, axis=1).sort_index()
    out.index = out.index.tz_localize(None)
    return out

def _yf_hist_inc(ticker: str, end_incl: pd.Timestamp, days: int) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    start = (end_incl - pd.Timedelta(days=days*2)).strftime("%Y-%m-%d")
    endex = (end_incl + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = t.history(start=start, end=endex, interval="1d", auto_adjust=False)
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    return df


# =========================
# Estimators (Ensemble)
# =========================
def _slice_training(returns: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    end = _to_ts(decision_date) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=3)
    return returns.loc[(returns.index >= start) & (returns.index <= end)]

def _ewma_component(sub: pd.DataFrame, span: int) -> pd.Series:
    return sub.ewm(span=span, adjust=False).mean().iloc[-1]  # daily return estimate

def _trailing_vol_ann(sub: pd.DataFrame, window: int = 63) -> pd.Series:
    if len(sub) < window:
        return pd.Series(0.20, index=sub.columns)  # fallback
    return sub.tail(window).std() * np.sqrt(252.0)

def _market_regime(decision_date: pd.Timestamp) -> Tuple[str, float]:
    try:
        end = _to_ts(decision_date)
        vix = _yf_hist_inc("^VIX", end, 60).dropna()
        if vix.empty:
            return "neutral", 20.0
        v = float(vix["Close"].iloc[-1])
        if v <= VIX_THRESHOLDS["risk_on_max"]:  return "risk_on", v
        if v >= VIX_THRESHOLDS["risk_off_min"]: return "risk_off", v
        return "neutral", v
    except Exception:
        return "neutral", 20.0

def _shock_multipliers(decision_date: pd.Timestamp) -> Tuple[float, float]:
    try:
        end = _to_ts(decision_date)
        tnx = _yf_hist_inc("^TNX", end, 30)
        dxy = _yf_hist_inc("DX-Y.NYB", end, 30)
        risk_mult, gold_mult = 1.0, 1.0
        if not tnx.empty:
            last = float(tnx["Close"].iloc[-1])
            prev = float(tnx["Close"].iloc[max(-SHOCK_WINDOW_D, -len(tnx))])
            bps = (last - prev) * 10.0  # ^TNX *10 ≈ bps
            if abs(bps) >= TNX_ABS_BPS_SHOCK:
                risk_mult *= SHOCK_RISK_MULT
                gold_mult *= SHOCK_GOLD_MULT
        if not dxy.empty:
            last = float(dxy["Close"].iloc[-1])
            prev = float(dxy["Close"].iloc[max(-SHOCK_WINDOW_D, -len(dxy))])
            pct = 100.0 * (last - prev) / prev if prev else 0.0
            if abs(pct) >= DXY_PCT_SHOCK:
                risk_mult *= SHOCK_RISK_MULT
                gold_mult *= SHOCK_GOLD_MULT
        return risk_mult, gold_mult
    except Exception:
        return 1.0, 1.0

def _ensemble_mu_components(returns: pd.DataFrame, decision_date: pd.Timestamp, regime: str) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      mu_raw_daily : estimated daily return (unadjusted) per asset
      vol_ann      : trailing annualized volatility per asset
    """
    sub = _slice_training(returns, decision_date)
    if sub.empty:
        idx = returns.columns
        return pd.Series(0.0, index=idx), pd.Series(0.20, index=idx)

    w = MH_WEIGHTS_BY_REGIME.get(regime, MH_WEIGHTS_BY_REGIME["neutral"])
    w = w / w.sum()

    comps = [ _ewma_component(sub, s) for s in MH_SPANS ]
    mat = pd.DataFrame(comps, index=MH_SPANS, columns=sub.columns)  # daily μ per horizon
    mu_raw_daily = (mat.T @ w)                                     # daily μ (unadjusted)
    vol_ann = _trailing_vol_ann(sub, VOL_WIN)                       # annual vol
    return mu_raw_daily, vol_ann.replace(0.0, 1e-6)

def _horizon_days(h: str) -> int:
    return {"monthly":21,"quarterly":63,"half":126,"yearly":252}.get(h,21)


# =========================
# Optimizer helpers
# =========================
def _project_capped_simplex(w_raw: Dict[str, float], floors: Dict[str, float], caps: Dict[str, float]) -> Dict[str, float]:
    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    lo = np.array([floors.get(k,0.0) for k in keys], float)
    hi = np.array([caps.get(k,1.0)  for k in keys], float)
    x0 = np.array([max(0.0, float(w_raw.get(k,0.0))) for k in keys], float)

    if lo.sum() > 1.0: lo = lo / lo.sum()
    x0 = np.minimum(hi, np.maximum(lo, x0))
    if abs(float(x0.sum()) - 1.0) <= 1e-12:
        return {k: float(v) for k,v in zip(keys,x0)}

    tau_lo = (x0 - hi).min() - 1.0
    tau_hi = (x0 - lo).max() + 1.0
    for _ in range(80):
        tau = 0.5*(tau_lo+tau_hi)
        x = np.clip(x0 - tau, lo, hi)
        s = float(x.sum())
        if s > 1.0: tau_lo = tau
        else:       tau_hi = tau
    x = np.clip(x0 - 0.5*(tau_lo+tau_hi), lo, hi)
    x = x / max(1e-12, x.sum())
    return {k: float(v) for k,v in zip(keys,x)}

def _apply_turnover(prev_w: Optional[Dict[str, float]], new_w: Dict[str, float], limit: float) -> Dict[str, float]:
    if prev_w is None: return new_w
    keys = ["Equities","Gold","REITs","Bitcoin"]
    p = np.array([prev_w[k] for k in keys], float)
    n = np.array([new_w[k]  for k in keys], float)
    diff = n - p
    l1 = float(np.abs(diff).sum())
    if l1 <= limit + 1e-12: return new_w
    scale = limit / max(1e-12, l1)
    adj = p + diff*scale
    adj = adj / max(1e-12, adj.sum())
    out = {k: float(v) for k,v in zip(keys, adj)}
    return _project_capped_simplex(out, FLOORS_BASE, CAPS_BASE)


# =========================
# KPI wrappers
# =========================
async def _btc_kpi(dd: pd.Timestamp, h: str) -> Dict:
    asof = dd.date().isoformat()
    return await _safe_kpi("BTC", analyze_btc_sentiment(backtest_date=asof, horizon=_hletter(h), historical_cutoff=asof))

async def _gold_kpi(dd: pd.Timestamp, h: str) -> Dict:
    asof = dd.date().isoformat()
    try:
        coro = gold_kpi_composite(asof, _hletter(h))
    except TypeError:
        coro = gold_kpi_composite(asof)
    return await _safe_kpi("GOLD", coro)

async def _reit_kpi(dd: pd.Timestamp, h: str) -> Dict:
    asof = dd.date().isoformat()
    return await _safe_kpi("REIT", reit_kpi_composite(asof, _hletter(h)))

async def _eq_kpi(dd: pd.Timestamp, h: str) -> Dict:
    asof = dd.date().isoformat()
    try:
        return await _safe_kpi("EQ", analyze_nifty_composite(asof, _hletter(h)))
    except TypeError:
        return await _safe_kpi("EQ", analyze_nifty_composite(asof))


# =========================
# Period helpers
# =========================
def _period_dates_monthly(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for m in range(1,13):
        end = _to_ts(f"{year}-{m:02d}-01") + pd.offsets.MonthEnd(0)
        dec = end - pd.offsets.MonthEnd(1)
        out.append((_to_ts(dec), _to_ts(end)))
    return out

def _period_dates_quarterly(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    q_ends = [f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
    out = []
    for q in q_ends:
        end = _to_ts(q)
        dec = end - pd.offsets.QuarterEnd(1)
        out.append((_to_ts(dec), _to_ts(end)))
    return out

def _period_dates_halfyear(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    ends = [f"{year}-06-30", f"{year}-12-31"]
    out = []
    for e in ends:
        end = _to_ts(e)
        dec = _to_ts(f"{year-1}-12-31") if e.endswith("06-30") else _to_ts(f"{year}-06-30")
        out.append((dec, end))
    return out

def _period_dates_year(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    end = _to_ts(f"{year}-12-31")
    dec = _to_ts(f"{year-1}-12-31")
    return [(dec, end)]

def _realized_period_return(returns: pd.DataFrame, start_exclusive: pd.Timestamp, end_inclusive: pd.Timestamp) -> pd.Series:
    mask = (returns.index > start_exclusive) & (returns.index <= end_inclusive)
    sub = returns.loc[mask]
    if sub.empty:
        return pd.Series(0.0, index=returns.columns)
    gross = (1.0 + sub).prod(axis=0)
    return gross - 1.0


# =========================
# Calibration helpers (walk-forward)
# =========================
def _past_month_end(d: pd.Timestamp) -> pd.Timestamp:
    return (d + pd.offsets.MonthEnd(0)).normalize()

def _decision_dates_back(decision_date: pd.Timestamp, n_back: int) -> List[pd.Timestamp]:
    """Return the previous n_back month-ends strictly before decision_date."""
    end = _past_month_end(decision_date - pd.offsets.MonthBegin(1))
    dates = []
    cur = end
    for _ in range(n_back):
        dates.append(cur)
        cur = cur - pd.offsets.MonthEnd(1)
    return list(reversed(dates))

def _mu_raw_h_for_decision(returns: pd.DataFrame, dec: pd.Timestamp, horizon_key: str, regime_at_dec: str) -> pd.Series:
    mu_raw_daily, _ = _ensemble_mu_components(returns, dec, regime_at_dec)
    return mu_raw_daily * float(_horizon_days(horizon_key))  # raw horizon momentum (no KPIs/shocks)

def _forward_realized_from_dec(returns: pd.DataFrame, dec: pd.Timestamp, horizon_key: str) -> pd.Series:
    pe = _past_month_end(dec)
    hdays = _horizon_days(horizon_key)
    fwd_end = pe + pd.Timedelta(days=int(hdays * 1.5))  # crude map
    fwd_end = returns.index[returns.index <= fwd_end].max()
    return _realized_period_return(returns, pe, fwd_end)

def _collect_pairs(returns: pd.DataFrame, decision_date: pd.Timestamp, horizon_key: str):
    """Collect (signal, realized) pairs per asset, both per-regime and pooled."""
    dates = _decision_dates_back(decision_date, CALIB_MONTHS_BACK)
    per_regime = {"risk_on": {}, "neutral": {}, "risk_off": {}}
    pooled = {}

    for dec in dates:
        regime, _ = _market_regime(dec)
        x = _mu_raw_h_for_decision(returns, dec, horizon_key, regime)
        y = _forward_realized_from_dec(returns, dec, horizon_key)
        for asset in returns.columns:
            per_regime[regime].setdefault(asset, {"x": [], "y": []})
            per_regime[regime][asset]["x"].append(float(x.get(asset, 0.0)))
            per_regime[regime][asset]["y"].append(float(y.get(asset, 0.0)))
            pooled.setdefault(asset, {"x": [], "y": []})
            pooled[asset]["x"].append(float(x.get(asset, 0.0)))
            pooled[asset]["y"].append(float(y.get(asset, 0.0)))
    return per_regime, pooled

def _expected_from_bins(x_hist: np.ndarray, y_hist: np.ndarray, x_now: float) -> Optional[float]:
    """Piecewise-constant-to-linear mapping using quintile bins."""
    if len(x_hist) < 6:
        return None
    qs = np.quantile(x_hist, BIN_Q)
    # avoid degenerate equal quantiles
    for i in range(1, len(qs)):
        if qs[i] <= qs[i-1]:
            qs[i] = qs[i-1] + 1e-12
    # means per bin
    means = []
    centers = []
    for j in range(len(qs)-1):
        lo, hi = qs[j], qs[j+1]
        mask = (x_hist >= lo) & (x_hist <= hi) if j == len(qs)-2 else (x_hist >= lo) & (x_hist < hi)
        if mask.sum() == 0:
            means.append(np.nan)
        else:
            means.append(float(np.mean(y_hist[mask])))
        centers.append(0.5*(lo + hi))
    means = np.array(means, dtype=float)
    centers = np.array(centers, dtype=float)

    # fill NaNs by interpolation (fallback to global mean if needed)
    if np.isnan(means).all():
        return float(np.mean(y_hist))
    if np.isnan(means).any():
        not_nan = ~np.isnan(means)
        fill = np.interp(np.arange(len(means)), np.where(not_nan)[0], means[not_nan])
        means = fill

    # interpolate expected at x_now
    return float(np.interp(x_now, centers, means, left=means[0], right=means[-1]))

def _expected_from_ridge(x_hist: np.ndarray, y_hist: np.ndarray, x_now: float) -> float:
    """Small-ridge linear map with intercept; slope clamped."""
    if len(x_hist) < 2:
        return float(np.mean(y_hist)) if len(y_hist) else 0.0
    xm = float(np.mean(x_hist))
    ym = float(np.mean(y_hist))
    xt = x_hist - xm
    yt = y_hist - ym
    den = float(np.dot(xt, xt) + RIDGE_L2)
    a = float(np.dot(xt, yt) / den) if den > 1e-12 else 1.0
    a = float(np.clip(a, SLOPE_BOUNDS[0], SLOPE_BOUNDS[1]))
    b = ym - a * xm
    return float(a * x_now + b)

def _calibrated_expected_assets(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    horizon_key: str,
    regime_now: str,
    mu_raw_h_now: pd.Series,   # today's raw horizon momentum per-asset (no KPIs/shocks)
) -> pd.Series:
    per_regime, pooled = _collect_pairs(returns, decision_date, horizon_key)
    out = {}
    for asset in returns.columns:
        # regime sample first
        xr = np.array(per_regime[regime_now].get(asset, {"x": []})["x"], dtype=float)
        yr = np.array(per_regime[regime_now].get(asset, {"y": []})["y"], dtype=float)
        xp = np.array(pooled.get(asset, {"x": []})["x"], dtype=float)
        yp = np.array(pooled.get(asset, {"y": []})["y"], dtype=float)

        x0 = float(mu_raw_h_now.get(asset, 0.0))

        val = None
        if len(xr) >= MIN_REGIME_SAMPLES:
            val = _expected_from_bins(xr, yr, x0)
            if val is None:
                val = _expected_from_ridge(xr, yr, x0)
        else:
            # fallback to pooled
            val = _expected_from_bins(xp, yp, x0)
            if val is None:
                val = _expected_from_ridge(xp, yp, x0)

        out[asset] = float(val)
    return pd.Series(out)


# =========================
# Forecasting
# =========================
def _apply_kpi_tilt(vec: pd.Series, eq: Dict, gold: Dict, reit: Dict, btc: Dict, alpha: float) -> pd.Series:
    scores = {
        "Equities": float(eq.get("composite_sentiment", 0.0) or 0.0),
        "Gold":     float(gold.get("composite_sentiment", 0.0) or 0.0),
        "REITs":    float(reit.get("composite_sentiment", 0.0) or 0.0),
        "Bitcoin":  float(btc.get("composite_sentiment", 0.0) or 0.0),
    }
    out = vec.copy()
    for k, s in scores.items():
        if k in out.index:
            out.loc[k] = float(out.loc[k]) * (1.0 + alpha * float(np.clip(s, -1.0, 1.0)))
    return out

def _apply_regime_caps(regime: str) -> Dict[str, float]:
    return CAPS_BY_REGIME.get(regime, CAPS_BASE)

def _apply_shock_scaling(vec: pd.Series, risk_mult: float, gold_mult: float) -> pd.Series:
    out = vec.copy()
    for k in ["Equities","REITs","Bitcoin"]:
        if k in out.index:
            out.loc[k] *= risk_mult
    if "Gold" in out.index:
        out.loc["Gold"] *= gold_mult
    return out

async def forecast_one_period(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    period_end: pd.Timestamp,
    horizon_key: str,
    w_prev: Optional[Dict[str, float]],
    period_index: int = 1,
    period_count: int = 1,
) -> Tuple[Dict[str, float], float, Dict[str, Dict], float]:

    # Regime + shocks
    regime, vix = _market_regime(decision_date)
    alpha = TILT_ALPHA.get(regime, TILT_ALPHA["neutral"])
    risk_mult, gold_mult = _shock_multipliers(decision_date)

    # Base ensemble components (raw daily μ and annual vol)
    mu_raw_daily, vol_ann = _ensemble_mu_components(returns, decision_date, regime)

    # Risk-adjusted signal for allocation
    mu_sig = (mu_raw_daily / (vol_ann + 1e-9)).clip(-1.0, 1.0)

    # KPIs (forward-looking tilts)
    eq   = await _eq_kpi(decision_date, horizon_key)
    gold = await _gold_kpi(decision_date, horizon_key)
    reit = await _reit_kpi(decision_date, horizon_key)
    btc  = await _btc_kpi(decision_date, horizon_key)

    # Apply KPI tilt + shock scaling (weights side)
    mu_sig       = _apply_kpi_tilt(mu_sig,       eq, gold, reit, btc, alpha)
    mu_sig       = _apply_shock_scaling(mu_sig,  risk_mult, gold_mult)

    # Long-only allocation from (risk-adjusted) signal
    keys = ["Equities","Gold","REITs","Bitcoin"]
    sig_vec = mu_sig.reindex(keys).fillna(0.0).to_numpy()
    if (sig_vec > 0).any():
        w_raw = sig_vec.clip(min=0.0)
        w_raw = w_raw / w_raw.sum()
    else:
        w_raw = np.array([0.20, 0.40, 0.25, 0.15], dtype=float)
    w0 = {k: float(v) for k, v in zip(keys, w_raw)}

    # Project with regime caps and apply turnover control
    caps = _apply_regime_caps(regime)
    w_proj = _project_capped_simplex(w0, FLOORS_BASE, caps)
    w_final = _apply_turnover(w_prev, w_proj, TURNOVER_LIMIT)

    # ---------- NEW: Calibrated Expected Return (printed) ----------
    mu_raw_h_now = mu_raw_daily * float(_horizon_days(horizon_key))   # no KPIs/shocks
    exp_assets = _calibrated_expected_assets(returns, decision_date, horizon_key, regime, mu_raw_h_now)

    # Apply today's tilts/shocks to expected **levels**
    exp_assets = _apply_kpi_tilt(exp_assets, eq, gold, reit, btc, alpha)
    exp_assets = _apply_shock_scaling(exp_assets, risk_mult, gold_mult).reindex(keys).fillna(0.0)

    # Portfolio expected = weights dot calibrated expectations
    exp_h = float(np.dot(exp_assets.to_numpy(), np.array([w_final[k] for k in keys])))
    if horizon_key == "monthly":
        exp_h = float(np.clip(exp_h, -EXP_CLIP_MONTHLY, EXP_CLIP_MONTHLY))

    # Realized for the period
    realized_vec = _realized_period_return(returns, decision_date, period_end)
    realized = float(np.dot(realized_vec.reindex(keys).values,
                            np.array([w_final[k] for k in keys])))

    # Print
    if _should_print_period(period_index, period_count):
        line = f"{period_end.strftime('%Y-%m')} | w: {_fmt_w_short(w_final)} | exp={_fmt_pct(exp_h)} | real={_fmt_pct(realized)}"
        print(line)
        if SHOW_KPI_SUMMARY:
            print(f"    KPI: eq={float(eq.get('composite_sentiment',0.0)):+.3f} "
                  f"gold={float(gold.get('composite_sentiment',0.0)):+.3f} "
                  f"reit={float(reit.get('composite_sentiment',0.0)):+.3f} "
                  f"btc={float(btc.get('composite_sentiment',0.0)):+.3f}")

    kpi_pack = {"equities": eq, "gold": gold, "reit": reit, "bitcoin": btc, "regime": regime, "vix": vix}
    return w_final, exp_h, kpi_pack, realized


# =========================
# Orchestration
# =========================
def _summarize(label: str, realized_list: List[float]):
    gross = np.prod([1.0 + r for r in realized_list]) - 1.0 if realized_list else 0.0
    ann_ret = gross
    ann_vol = float(np.std(realized_list, ddof=1) * np.sqrt(12.0)) if len(realized_list) > 1 else 0.0
    sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
    print(f"\n========== {label} REALIZED PERFORMANCE ==========")
    print(f"Total: {_fmt_pct(gross)}  |  AnnRet: {_fmt_pct(ann_ret)}  |  AnnVol: {100*ann_vol:.2f}%  |  Sharpe: {sharpe:.2f}")

async def forecast_report_2023(charts: bool = False, prefix: Optional[str] = None):
    returns = load_returns_fresh()

    # ---- MONTHLY ----
    if PRINT_MODE != "quiet":
        print("\n\n========== MONTHLY FORECASTS (12) ==========")
        print("(For month M, decision date = previous month end; ensemble & KPIs use data up to that date.)\n")

    monthly_periods = _period_dates_monthly(2023)
    w_prev = None
    monthly_turnovers_l1: List[float] = []
    monthly_realized: List[float] = []

    decisions_for_chart: List[Dict] = []
    daily_strategy_parts: List[pd.Series] = []

    for idx, (decision_date, period_end) in enumerate(monthly_periods, start=1):
        w, _, _, realized = await forecast_one_period(
            returns, decision_date, period_end, "monthly", w_prev,
            period_index=idx, period_count=len(monthly_periods)
        )
        decisions_for_chart.append({"period_end": period_end.date().isoformat(), "weights": w})
        mask = (returns.index > decision_date) & (returns.index <= period_end)
        sub = returns.loc[mask, ["Equities","Gold","REITs","Bitcoin"]]
        if not sub.empty:
            w_vec = np.array([w["Equities"], w["Gold"], w["REITs"], w["Bitcoin"]], float)
            strat_daily = pd.Series(sub.values @ w_vec, index=sub.index, name="strategy")
            daily_strategy_parts.append(strat_daily)

        if w_prev is not None:
            keys = ["Equities","Gold","REITs","Bitcoin"]
            l1 = float(sum(abs(w[k] - w_prev[k]) for k in keys))
            monthly_turnovers_l1.append(l1)
        w_prev = w
        monthly_realized.append(realized)

    # ---- QUARTERLY ----
    quarterly_realized: List[float] = []
    if PRINT_MODE != "quiet":
        print("\n========== QUARTERLY FORECASTS (3: Q2, Q3, Q4) ==========")
        print("(Each quarter uses ensemble & KPIs up to previous quarter end; Q1 is the 2022-end decision.)\n")
    for idx, (dd, pe) in enumerate(_period_dates_quarterly(2023), start=1):
        _, _, _, realized = await forecast_one_period(returns, dd, pe, "quarterly", w_prev, idx, 3)
        quarterly_realized.append(realized)

    # ---- HALF-YEAR ----
    half_realized: List[float] = []
    if PRINT_MODE != "quiet":
        print("\n\n========== HALF-YEAR FORECASTS (2: H1, H2) ==========")
        print("(H1 uses 2022-12-31 ensemble/KPIs; H2 uses 2023-06-30 ensemble/KPIs.)\n")
    for idx, (dd, pe) in enumerate(_period_dates_halfyear(2023), start=1):
        _, _, _, realized = await forecast_one_period(returns, dd, pe, "half", w_prev, idx, 2)
        half_realized.append(realized)

    # ---- YEARLY ----
    yearly_realized: List[float] = []
    if PRINT_MODE != "quiet":
        print("\n\n========== YEARLY FORECAST (1) ==========")
        print("(Y2023 uses ensemble & KPIs up to 2022-12-31.)\n")
    for idx, (dd, pe) in enumerate(_period_dates_year(2023), start=1):
        _, _, _, realized = await forecast_one_period(returns, dd, pe, "yearly", w_prev, idx, 1)
        yearly_realized.append(realized)

    # ---- Final summaries ----
    _summarize("MONTHLY POLICY: 2023", monthly_realized)
    print(f"Avg monthly turnover: {100*float(np.mean(monthly_turnovers_l1)):.2f}%")
    _summarize("QUARTERLY POLICY: 2023", quarterly_realized)
    _summarize("HALF-YEAR POLICY: 2023", half_realized)
    _summarize("YEARLY POLICY: 2023", yearly_realized)
    print("Done. (12 + 3 + 2 + 1 = 18 forecasts printed.)")

    # ---- Investor charts (optional) ----
    if charts:
        if daily_strategy_parts:
            strategy_daily_2023 = pd.concat(daily_strategy_parts).sort_index()
            strategy_daily_2023.index = strategy_daily_2023.index.tz_localize(None)
        else:
            strategy_daily_2023 = pd.Series(dtype=float, name="strategy")
        results = {
            "daily_returns": strategy_daily_2023,
            "decisions": decisions_for_chart,
            "summary": {}  # not used by the chart function
        }
        try:
            generate_investor_report(results, output_prefix=(prefix or "Exec_Report_2023"))
            print(f"\nInvestor charts saved with prefix '{(prefix or 'Exec_Report_2023')}_*.png'")
        except Exception as e:
            print(f"[WARN] Investor report generation failed: {e}")


# =========================
# CLI
# =========================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest + Investor Charts (ensemble + regime + shocks + quantile-calibrated exp)")
    p.add_argument("--charts", action="store_true", help="save investor charts (3 PNG files)")
    p.add_argument("--prefix", type=str, default=None, help="filename prefix for charts")
    p.add_argument("--print", dest="print_mode", choices=["full","summary","quiet"], default="summary")
    return p.parse_args()

def main():
    global PRINT_MODE
    args = _parse_args()
    PRINT_MODE = args.print_mode
    asyncio.run(forecast_report_2023(charts=args.charts, prefix=args.prefix))

if __name__ == "__main__":
    main()
