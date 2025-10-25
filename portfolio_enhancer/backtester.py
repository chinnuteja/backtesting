from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import sys
from typing import Dict, Optional, Tuple, List, Callable, Awaitable

import numpy as np
import pandas as pd
import yfinance as yf

# --- Scenario engine imports ---
# Assumes these modules exist in your repo
from .inputs_forecaster import VIXNowcaster, DriftNowcaster, kpi_persistence_sim
from .scenario_runner import run_scenarios

# --- External async KPI modules ---
from .kpis.gold.gold_kpi_composite import gold_kpi_composite
from .kpis.bitcoin.btc_sentiment import analyze_btc_sentiment
from .kpis.reit import reit_kpi_composite
from .kpis.nifty import analyze_nifty_composite

# Optional investor visuals
try:
    from .investor_report import generate_investor_report
except Exception:
    generate_investor_report = None

# =========================
# Global config
# =========================
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"  # pull enough data for calibration spillover

TICKERS = {
    "Equities": "^NSEI",
    "Gold": "GLD",
    "REITs": "VNQ",
    "Bitcoin": "BTC-USD",
}

FLOORS_BASE = {"Equities": 0.00, "Gold": 0.00, "REITs": 0.00, "Bitcoin": 0.00}
CAPS_BASE = {"Equities": 0.50, "Gold": 0.35, "REITs": 0.30, "Bitcoin": 0.20}
TURNOVER_LIMIT = 0.30  # L1 per decision

# ---------- SIGNAL UPGRADE ----------
# we blend short / medium / long momentum differently based on risk regime
MH_SPANS = [21, 63, 126]

MH_WEIGHTS_BY_REGIME = {
    "risk_on": np.array([0.65, 0.25, 0.10]),
    "neutral": np.array([0.33, 0.34, 0.33]),
    "risk_off": np.array([0.10, 0.30, 0.60]),
}

VOL_WIN = 63  # vol window for risk-adjusted signal

# US regime gating via VIX
VIX_THRESHOLDS_US = {"risk_on_max": 12.0, "risk_off_min": 20.0}
# India regime gating via India VIX (low teens most of 2023)
VIX_THRESHOLDS_IN = {"risk_on_max": 16.0, "risk_off_min": 25.0}

# sentiment tilt strength per regime
TILT_ALPHA = {"risk_on": 0.30, "neutral": 0.25, "risk_off": 0.25}

# caps vary by regime (more BTC/equities when "risk_on", more Gold when "risk_off")
CAPS_BY_REGIME = {
    "risk_on": {"Equities": 0.60, "Gold": 0.30, "REITs": 0.35, "Bitcoin": 0.35},
    "neutral": CAPS_BASE,
    "risk_off": {"Equities": 0.45, "Gold": 0.60, "REITs": 0.25, "Bitcoin": 0.15},
}

# macro "shock" detector (yields + USD strength jump)
SHOCK_WINDOW_D = 10
TNX_ABS_BPS_SHOCK = 30.0     # ~30bp 10y jump in 10d
FX_PCT_SHOCK = 1.8           # ~1.8% USD/INR or DXY pop in 10d
SHOCK_RISK_MULT = 0.85       # cut risk assets
SHOCK_GOLD_MULT = 1.15       # boost gold

# walk-forward calibration settings for the printed expected return
CALIB_MONTHS_BACK = 36
MIN_REGIME_SAMPLES = 18
RIDGE_L2 = 0.5
SLOPE_BOUNDS = (0.25, 2.0)
BIN_Q = [0.0, .2, .4, .6, .8, 1.0]
EXP_CLIP_MONTHLY = 0.25  # clamp crazy 1-month expectations for readability

# output / runtime controls
PRINT_MODE = "summary"
SHOW_KPI_SUMMARY = True

# important: keep KPI calls from hanging / spamming stdout
SILENCE_KPI_LOGS = True
KPI_TIMEOUT_SEC = 5  # seconds per KPI call max

# scenario engine knobs
SCENARIOS_N = 0
ROBUST_PCTL_FLOOR = None  # optional tail-risk brake
GLOBAL_SEED = 42

# macro env mode: "us" = ^VIX + DXY, "in" = ^INDIAVIX + USD/INR
MACRO_ENV = "us"

np.set_printoptions(suppress=True, linewidth=120)


# =========================
# Small helpers
# =========================

def _to_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_localize(None) if getattr(x, "tzinfo", None) is not None else x
    return pd.Timestamp(x).tz_localize(None)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:+.2f}%"


def _fmt_w_short(w: Dict[str, float]) -> str:
    return "E={:.1f}% G={:.1f}% R={:.1f}% B={:.1f}%".format(
        100 * w["Equities"], 100 * w["Gold"], 100 * w["REITs"], 100 * w["Bitcoin"]
    )


def _hletter(h: str) -> str:
    return {"monthly": "M", "quarterly": "Q", "half": "H", "yearly": "Y"}.get(h, "M")


def _should_print_period(i: int, total: int) -> bool:
    if PRINT_MODE == "quiet":
        return False
    return True  # summary + full both print each period


def _horizon_days(h: str) -> int:
    return {"monthly": 21, "quarterly": 63, "half": 126, "yearly": 252}.get(h, 21)


def _annualize(exp_h: float, horizon_days: int) -> float:
    # take horizon return and express as ~252d annual rate
    if horizon_days <= 0:
        return exp_h
    return (1.0 + exp_h) ** (252.0 / float(horizon_days)) - 1.0


# helper: blend live weights toward a calmer anchor for messaging
def _blend_to_anchor(
    w_live: Dict[str, float],
    anchor_weights: Dict[str, float],
    blend_frac: float
) -> Dict[str, float]:
    """
    w_live: actual live/trade weights, sum ~1
    anchor_weights: calmer mix (must have same keys and sum ~1)
    blend_frac: fraction [0,1] toward anchor (0 = all live, 1 = all anchor)

    returns blended dict with same keys, renormalized to 1.
    """
    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    wl = np.array([w_live.get(k, 0.0) for k in keys], dtype=float)
    wa = np.array([anchor_weights.get(k, 0.0) for k in keys], dtype=float)

    wb = (1.0 - blend_frac) * wl + blend_frac * wa
    s = float(wb.sum())
    if s > 1e-12:
        wb = wb / s
    return {k: float(v) for k, v in zip(keys, wb)}


# =========================
# KPI wrappers (async, silent, timeout)
# =========================

async def _safe_kpi(label: str, factory: Callable[[], Awaitable[Dict]]) -> Dict:
    """
    factory() builds & returns the KPI coroutine.
    We:
      - capture stdout/stderr so they can't spam
      - add a hard timeout so they can't hang
      - fall back to neutral if anything blows up
    """
    async def _runner():
        if not SILENCE_KPI_LOGS:
            return await factory()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            return await factory()

    try:
        return await asyncio.wait_for(_runner(), timeout=KPI_TIMEOUT_SEC)
    except Exception as e:
        if PRINT_MODE != "quiet":
            print(f"[WARN] {label} KPI failed: {e!s} — using neutral 0.0")
        return {"composite_sentiment": 0.0, "components": {}}


async def _btc_kpi(dd: pd.Timestamp, h: str) -> Dict:
    asof = dd.date().isoformat()

    async def _f():
        return await analyze_btc_sentiment(
            backtest_date=asof,
            horizon=_hletter(h),
            historical_cutoff=asof
        )

    return await _safe_kpi("BTC", _f)


async def _gold_kpi(dd: pd.Timestamp, h: str) -> Dict:
    asof = dd.date().isoformat()

    async def _f():
        try:
            return await gold_kpi_composite(asof, _hletter(h))
        except TypeError:
            # some versions only take "asof"
            return await gold_kpi_composite(asof)

    return await _safe_kpi("GOLD", _f)


async def _reit_kpi(dd: pd.Timestamp, h: str) -> Dict:
    asof = dd.date().isoformat()

    async def _f():
        return await reit_kpi_composite(asof, _hletter(h))

    return await _safe_kpi("REIT", _f)


async def _eq_kpi(dd: pd.Timestamp, h: str) -> Dict:
    asof = dd.date().isoformat()

    async def _f():
        try:
            return await analyze_nifty_composite(asof, _hletter(h))
        except TypeError:
            return await analyze_nifty_composite(asof)

    return await _safe_kpi("EQ", _f)


# =========================
# Data helpers
# =========================

def _fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1), auto_adjust=True)
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    return df


def load_returns_fresh() -> pd.DataFrame:
    series = []
    for name, tk in TICKERS.items():
        df = _fetch_history(tk, START_DATE, END_DATE)
        s = df["Close"].pct_change().dropna().rename(name)
        series.append(s)
    out = pd.concat(series, axis=1).sort_index()
    # make sure tz-naive to avoid AttributeError
    if hasattr(out.index, "tz_localize"):
        try:
            out.index = out.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    return out


def _yf_hist_inc(ticker: str, end_incl: pd.Timestamp, days: int) -> pd.DataFrame:
    """
    inclusive history up to end_incl for 'days' lookback.
    we over-fetch ~2x window so weekends / holidays aren't missing.
    """
    t = yf.Ticker(ticker)
    start = (end_incl - pd.Timedelta(days=days * 2)).strftime("%Y-%m-%d")
    endex = (end_incl + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = t.history(start=start, end=endex, interval="1d", auto_adjust=False)
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    return df


def validate_data_availability(returns: pd.DataFrame, year: int) -> bool:
    """
    quick sanity check that we have enough bars for that year
    """
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    year_data = returns.loc[(returns.index >= year_start) & (returns.index <= year_end)]

    if year_data.empty:
        print(f"WARNING: No data available for {year}")
        return False

    missing_assets = []
    for asset in returns.columns:
        asset_data = year_data[asset].dropna()
        if len(asset_data) < 50:  # arbitrary: ~2.5mo of trading days
            missing_assets.append(asset)

    if missing_assets:
        print(f"WARNING: Insufficient data for {missing_assets} in {year}")
        print(f"Proceeding with available data...")

    coverage = len(year_data) / 252 * 100.0
    print(f"Data coverage for {year}: {coverage:.1f}%")
    return True


def handle_missing_recent_data(returns: pd.DataFrame) -> pd.DataFrame:
    """
    forward-fill short gaps, then fill anything else with 0 return
    """
    returns_filled = returns.fillna(method='ffill', limit=5)
    returns_filled = returns_filled.fillna(0.0)
    return returns_filled


# =========================
# Signal estimators
# =========================

def _slice_training(returns: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    end = _to_ts(decision_date) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=3)
    return returns.loc[(returns.index >= start) & (returns.index <= end)]


def _ewma_component(sub: pd.DataFrame, span: int) -> pd.Series:
    return sub.ewm(span=span, adjust=False).mean().iloc[-1]


def _trailing_vol_ann(sub: pd.DataFrame, window: int = 63) -> pd.Series:
    if len(sub) < window:
        return pd.Series(0.20, index=sub.columns)
    return sub.tail(window).std() * np.sqrt(252.0)


def _market_regime(decision_date: pd.Timestamp, macro_env: str) -> Tuple[str, float, float]:
    """
    Returns (regime_label, vix_level, fx_level_for_print)

    macro_env == "us":
        VIX = ^VIX
        FX  = DXY (we try DX-Y.NYB first then DXY)
        thresholds = VIX_THRESHOLDS_US

    macro_env == "in":
        VIX = ^INDIAVIX
        FX  = USD/INR (INR=X)
        thresholds = VIX_THRESHOLDS_IN
    """
    end = _to_ts(decision_date)

    if macro_env == "in":
        vix_ticker = "^INDIAVIX"
        fx_candidates = ["INR=X"]  # USD/INR
        thresholds = VIX_THRESHOLDS_IN
    else:
        vix_ticker = "^VIX"
        fx_candidates = ["DX-Y.NYB", "DXY"]
        thresholds = VIX_THRESHOLDS_US

    # get VIX / India VIX
    try:
        vix_df = _yf_hist_inc(vix_ticker, end, 60).dropna()
    except Exception:
        vix_df = pd.DataFrame()

    if vix_df.empty:
        v_now = 20.0
    else:
        v_now = float(vix_df["Close"].iloc[-1])

    # classify into regime
    if v_now <= thresholds["risk_on_max"]:
        regime = "risk_on"
    elif v_now >= thresholds["risk_off_min"]:
        regime = "risk_off"
    else:
        regime = "neutral"

    # FX for print: DXY or USD/INR
    fx_val = None
    for fx_tk in fx_candidates:
        try:
            fx_df = _yf_hist_inc(fx_tk, end, 60).dropna()
            if not fx_df.empty:
                fx_val = float(fx_df["Close"].iloc[-1])
                break
        except Exception:
            pass
    if fx_val is None:
        fx_val = 100.0

    return regime, v_now, fx_val


def _shock_multipliers(decision_date: pd.Timestamp, macro_env: str) -> Tuple[float, float]:
    """
    detect a recent shock in TNX or USD strength and shift weights:
    - cut equities/btc/reits
    - boost gold
    """
    try:
        end = _to_ts(decision_date)

        # 10y yield (^TNX)
        tnx_df = _yf_hist_inc("^TNX", end, 30)
        # FX / USD strength:
        fx_candidates = ["INR=X"] if macro_env == "in" else ["DX-Y.NYB", "DXY"]
        fx_df = pd.DataFrame()
        for tk in fx_candidates:
            try:
                fx_df = _yf_hist_inc(tk, end, 30)
                if not fx_df.empty:
                    break
            except Exception:
                pass

        risk_mult = 1.0
        gold_mult = 1.0

        # TNX shock: ~10d change in bps
        if not tnx_df.empty:
            last = float(tnx_df["Close"].iloc[-1])
            prev = float(tnx_df["Close"].iloc[max(-SHOCK_WINDOW_D, -len(tnx_df))])
            bps = (last - prev) * 10.0  # ^TNX *10 ≈ bp
            if abs(bps) >= TNX_ABS_BPS_SHOCK:
                risk_mult *= SHOCK_RISK_MULT
                gold_mult *= SHOCK_GOLD_MULT

        # FX shock: % move
        if not fx_df.empty:
            last = float(fx_df["Close"].iloc[-1])
            prev = float(fx_df["Close"].iloc[max(-SHOCK_WINDOW_D, -len(fx_df))])
            pct = 100.0 * (last - prev) / prev if prev else 0.0
            if abs(pct) >= FX_PCT_SHOCK:
                risk_mult *= SHOCK_RISK_MULT
                gold_mult *= SHOCK_GOLD_MULT

        return risk_mult, gold_mult

    except Exception:
        return 1.0, 1.0


def _ensemble_mu_components(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    regime: str
) -> Tuple[pd.Series, pd.Series]:
    """
    returns:
      mu_raw_daily : per-asset expected daily return (raw momentum blend)
      vol_ann      : per-asset annualized vol (63d stdev * sqrt(252))
    """
    sub = _slice_training(returns, decision_date)
    if sub.empty:
        idx = returns.columns
        return pd.Series(0.0, index=idx), pd.Series(0.20, index=idx)

    w = MH_WEIGHTS_BY_REGIME.get(regime, MH_WEIGHTS_BY_REGIME["neutral"])
    w = w / w.sum()

    comps = [_ewma_component(sub, s) for s in MH_SPANS]  # each is Series
    mat = pd.DataFrame(comps, index=MH_SPANS, columns=sub.columns)  # horizons x assets

    mu_raw_daily = (mat.T @ w)  # weighted blend across horizons (daily μ est)
    vol_ann = _trailing_vol_ann(sub, VOL_WIN).replace(0.0, 1e-6)

    return mu_raw_daily, vol_ann


# =========================
# Expected-return calibration logic
# =========================

def _past_month_end(d: pd.Timestamp) -> pd.Timestamp:
    return (d + pd.offsets.MonthEnd(0)).normalize()


def _decision_dates_back(decision_date: pd.Timestamp, n_back: int) -> List[pd.Timestamp]:
    """
    list of previous month-ends strictly before decision_date,
    going back ~n_back months
    """
    end = _past_month_end(decision_date - pd.offsets.MonthBegin(1))
    out = []
    cur = end
    for _ in range(n_back):
        out.append(cur)
        cur = cur - pd.offsets.MonthEnd(1)
    return list(reversed(out))


def _mu_raw_h_for_decision(
    returns: pd.DataFrame,
    dec: pd.Timestamp,
    horizon_key: str,
    regime_at_dec: str
) -> pd.Series:
    mu_raw_daily, _ = _ensemble_mu_components(returns, dec, regime_at_dec)
    return mu_raw_daily * float(_horizon_days(horizon_key))


def _forward_realized_from_dec(
    returns: pd.DataFrame,
    dec: pd.Timestamp,
    horizon_key: str
) -> pd.Series:
    """
    realized forward perf from month-end(dec) to ~horizon days ahead
    """
    pe = _past_month_end(dec)
    hdays = _horizon_days(horizon_key)

    # approximate the future end date and then snap to last available index ≤ that
    approx_end = pe + pd.Timedelta(days=int(hdays * 1.5))
    approx_end = returns.index[returns.index <= approx_end].max()

    return _realized_period_return(returns, pe, approx_end)


def _collect_pairs(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    horizon_key: str,
    macro_env: str
):
    """
    build {(regime)->asset: (x=signal,y=realized)}, and pooled version.
    we'll learn mapping sig->expected using bins/regression.
    """
    dates = _decision_dates_back(decision_date, CALIB_MONTHS_BACK)
    per_regime = {"risk_on": {}, "neutral": {}, "risk_off": {}}
    pooled = {}

    for dec in dates:
        regime_dec, _, _ = _market_regime(dec, macro_env)
        x = _mu_raw_h_for_decision(returns, dec, horizon_key, regime_dec)
        y = _forward_realized_from_dec(returns, dec, horizon_key)

        for asset in returns.columns:
            per_regime[regime_dec].setdefault(asset, {"x": [], "y": []})
            per_regime[regime_dec][asset]["x"].append(float(x.get(asset, 0.0)))
            per_regime[regime_dec][asset]["y"].append(float(y.get(asset, 0.0)))

            pooled.setdefault(asset, {"x": [], "y": []})
            pooled[asset]["x"].append(float(x.get(asset, 0.0)))
            pooled[asset]["y"].append(float(y.get(asset, 0.0)))

    return per_regime, pooled


def _expected_from_bins(
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    x_now: float
) -> Optional[float]:
    """non-parametric binning by quintile of the signal"""
    if len(x_hist) < 6:
        return None

    qs = np.quantile(x_hist, BIN_Q)

    # make sure bins are strictly increasing
    for i in range(1, len(qs)):
        if qs[i] <= qs[i - 1]:
            qs[i] = qs[i - 1] + 1e-12

    means = []
    centers = []
    for j in range(len(qs) - 1):
        lo, hi = qs[j], qs[j + 1]
        if j == len(qs) - 2:
            mask = (x_hist >= lo) & (x_hist <= hi)
        else:
            mask = (x_hist >= lo) & (x_hist < hi)

        if mask.sum() == 0:
            means.append(np.nan)
        else:
            means.append(float(np.mean(y_hist[mask])))
        centers.append(0.5 * (lo + hi))

    means = np.array(means, dtype=float)
    centers = np.array(centers, dtype=float)

    # interpolate gaps
    if np.isnan(means).all():
        return float(np.mean(y_hist))
    if np.isnan(means).any():
        good = ~np.isnan(means)
        means = np.interp(np.arange(len(means)), np.where(good)[0], means[good])

    # now return piecewise-smoothed expectation at x_now
    return float(np.interp(x_now, centers, means, left=means[0], right=means[-1]))


def _expected_from_ridge(
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    x_now: float
) -> float:
    """
    little ridge linear model y ~ a*x + b
    slope is clamped
    """
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
    mu_raw_h_now: pd.Series,
    macro_env: str,
) -> pd.Series:
    """
    produce expected forward return per asset (for this horizon),
    learned from historical mapping signal->realized
    """
    per_regime, pooled = _collect_pairs(returns, decision_date, horizon_key, macro_env)
    out = {}

    for asset in returns.columns:
        xr = np.array(per_regime[regime_now].get(asset, {"x": []})["x"], dtype=float)
        yr = np.array(per_regime[regime_now].get(asset, {"y": []})["y"], dtype=float)
        xp = np.array(pooled.get(asset, {"x": []})["x"], dtype=float)
        yp = np.array(pooled.get(asset, {"y": []})["y"], dtype=float)
        x0 = float(mu_raw_h_now.get(asset, 0.0))

        if len(xr) >= MIN_REGIME_SAMPLES:
            guess = _expected_from_bins(xr, yr, x0)
            if guess is None:
                guess = _expected_from_ridge(xr, yr, x0)
        else:
            guess = _expected_from_bins(xp, yp, x0)
            if guess is None:
                guess = _expected_from_ridge(xp, yp, x0)

        out[asset] = float(guess)

    return pd.Series(out)


# =========================
# Weight helpers
# =========================

def _apply_kpi_tilt(
    vec: pd.Series,
    eq: Dict,
    gold: Dict,
    reit: Dict,
    btc: Dict,
    alpha: float
) -> pd.Series:
    # sentiment scores are assumed in [-1,+1]
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
    for k in ["Equities", "REITs", "Bitcoin"]:
        if k in out.index:
            out.loc[k] *= risk_mult
    if "Gold" in out.index:
        out.loc["Gold"] *= gold_mult
    return out


def _project_capped_simplex(
    w_raw: Dict[str, float],
    floors: Dict[str, float],
    caps: Dict[str, float]
) -> Dict[str, float]:
    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    lo = np.array([floors.get(k, 0.0) for k in keys], float)
    hi = np.array([caps.get(k, 1.0) for k in keys], float)
    x0 = np.array([max(0.0, float(w_raw.get(k, 0.0))) for k in keys], float)

    # make sure feasible
    if lo.sum() > 1.0:
        lo = lo / lo.sum()

    x0 = np.minimum(hi, np.maximum(lo, x0))

    if abs(float(x0.sum()) - 1.0) <= 1e-12:
        return {k: float(v) for k, v in zip(keys, x0)}

    # otherwise, find tau s.t. sum(clip(x0 - tau, lo, hi)) = 1
    tau_lo = (x0 - hi).min() - 1.0
    tau_hi = (x0 - lo).max() + 1.0
    for _ in range(80):
        tau = 0.5 * (tau_lo + tau_hi)
        x = np.clip(x0 - tau, lo, hi)
        s = float(x.sum())
        if s > 1.0:
            tau_lo = tau
        else:
            tau_hi = tau
    x = np.clip(x0 - 0.5 * (tau_lo + tau_hi), lo, hi)
    x = x / max(1e-12, x.sum())
    return {k: float(v) for k, v in zip(keys, x)}


def _apply_turnover(prev_w: Optional[Dict[str, float]], new_w: Dict[str, float], limit: float) -> Dict[str, float]:
    if prev_w is None:
        return new_w

    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    p = np.array([prev_w[k] for k in keys], float)
    n = np.array([new_w[k] for k in keys], float)

    diff = n - p
    l1 = float(np.abs(diff).sum())

    if l1 <= limit + 1e-12:
        return new_w

    scale = limit / max(1e-12, l1)
    adj = p + diff * scale
    adj = adj / max(1e-12, adj.sum())
    out = {k: float(v) for k, v in zip(keys, adj)}

    # final projection for safety
    return _project_capped_simplex(out, FLOORS_BASE, CAPS_BASE)


# =========================
# Period realized returns
# =========================

def _realized_period_return(
    returns: pd.DataFrame,
    start_exclusive: pd.Timestamp,
    end_inclusive: pd.Timestamp
) -> pd.Series:
    mask = (returns.index > start_exclusive) & (returns.index <= end_inclusive)
    sub = returns.loc[mask]
    if sub.empty:
        return pd.Series(0.0, index=returns.columns)
    gross = (1.0 + sub).prod(axis=0)
    return gross - 1.0


# =========================
# Core per-period forecast
# =========================

async def forecast_one_period(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    period_end: pd.Timestamp,
    horizon_key: str,
    w_prev: Optional[Dict[str, float]],
    period_index: int,
    period_count: int,
    macro_env: str,
) -> Tuple[Dict[str, float], float, Dict[str, Dict], float]:
    """
    does EVERYTHING for one decision point:
      - figure regime / macro state
      - get KPI tilts (sentiment)
      - build target weights under caps + turnover
      - estimate expected return
      - run scenario engine (if --scenarios >0 and monthly/quarterly)
      - compute realized perf
      - print pretty line
    returns (weights_used, expected_return_used_for_print, kpi_pack, realized)
    """

    # current macro regime + "shock" multipliers
    regime, vix_level, fx_level = _market_regime(decision_date, macro_env)
    alpha = TILT_ALPHA.get(regime, TILT_ALPHA["neutral"])
    risk_mult, gold_mult = _shock_multipliers(decision_date, macro_env)

    # base ensemble signal
    mu_raw_daily, vol_ann = _ensemble_mu_components(returns, decision_date, regime)

    # convert to risk-adjusted signal
    mu_sig = (mu_raw_daily / (vol_ann + 1e-9)).clip(-1.0, 1.0)

    # KPIs (async sentiment)
    eq   = await _eq_kpi(decision_date, horizon_key)
    gold = await _gold_kpi(decision_date, horizon_key)
    reit = await _reit_kpi(decision_date, horizon_key)
    btc  = await _btc_kpi(decision_date, horizon_key)

    # apply KPI tilt and macro shock scaling to the SIGNAL for allocation
    mu_sig = _apply_kpi_tilt(mu_sig, eq, gold, reit, btc, alpha)
    mu_sig = _apply_shock_scaling(mu_sig, risk_mult, gold_mult)

    # long-only positive-part normalize
    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    sig_vec = mu_sig.reindex(keys).fillna(0.0).to_numpy()

    if (sig_vec > 0).any():
        w_raw = sig_vec.clip(min=0.0)
        w_raw = w_raw / w_raw.sum()
    else:
        # fallback barbell if literally everything <=0
        w_raw = np.array([0.20, 0.40, 0.25, 0.15], dtype=float)

    w0 = {k: float(v) for k, v in zip(keys, w_raw)}

    # cap per regime and throttle turnover
    caps = _apply_regime_caps(regime)
    w_proj = _project_capped_simplex(w0, FLOORS_BASE, caps)
    w_final_base = _apply_turnover(w_prev, w_proj, TURNOVER_LIMIT)

    # ---------- baseline calibrated expected return ----------
    # first: horizon raw momentum without KPIs / shocks
    mu_raw_h_now = mu_raw_daily * float(_horizon_days(horizon_key))

    exp_assets = _calibrated_expected_assets(
        returns,
        decision_date,
        horizon_key,
        regime,
        mu_raw_h_now,
        macro_env,
    )

    # baseline path (still tilted for non-scenario long horizons)
    exp_assets_tilted = _apply_kpi_tilt(exp_assets, eq, gold, reit, btc, alpha)
    exp_assets_tilted = _apply_shock_scaling(exp_assets_tilted, risk_mult, gold_mult).reindex(keys).fillna(0.0)

    baseline_exp_h = float(
        np.dot(exp_assets_tilted.to_numpy(), np.array([w_final_base[k] for k in keys]))
    )

    if horizon_key == "monthly":
        baseline_exp_h = float(np.clip(baseline_exp_h, -EXP_CLIP_MONTHLY, EXP_CLIP_MONTHLY))

    # realized perf for this horizon
    realized_vec = _realized_period_return(returns, decision_date, period_end)
    realized_ret = float(
        np.dot(realized_vec.reindex(keys).values, np.array([w_final_base[k] for k in keys]))
    )

    # =========================================================
    # scenario engine path for monthly/quarterly
    # =========================================================
    if SCENARIOS_N and SCENARIOS_N > 0 and horizon_key in ("monthly", "quarterly"):
        np.random.seed(GLOBAL_SEED + int(period_index))
        H_days = int(_horizon_days(horizon_key))

        def _safe_last(hist: pd.Series, default_val: float = 0.0) -> float:
            return float(hist.iloc[-1]) if len(hist) else default_val

        # pull last observed macro inputs
        try:
            vix_hist_df = _yf_hist_inc(
                "^INDIAVIX" if macro_env == "in" else "^VIX",
                _to_ts(decision_date),
                200
            )
            vix_hist = vix_hist_df["Close"].dropna()
            last_vix = _safe_last(vix_hist, default_val=float(vix_level or 20.0))
        except Exception:
            vix_hist = pd.Series([], dtype=float)
            last_vix = float(vix_level or 20.0)

        try:
            tnx_hist_df = _yf_hist_inc("^TNX", _to_ts(decision_date), 200)
            tnx_hist = tnx_hist_df["Close"].dropna()
            last_tnx = _safe_last(tnx_hist, default_val=4.0)
        except Exception:
            tnx_hist = pd.Series([], dtype=float)
            last_tnx = 4.0

        fx_candidates = ["INR=X"] if macro_env == "in" else ["DX-Y.NYB", "DXY"]
        fx_hist = pd.Series([], dtype=float)
        last_fx = 100.0
        for tk in fx_candidates:
            try:
                fx_hist_df = _yf_hist_inc(tk, _to_ts(decision_date), 200)
                cand = fx_hist_df["Close"].dropna()
                if len(cand):
                    fx_hist = cand
                    last_fx = _safe_last(cand, default_val=100.0)
                    break
            except Exception:
                pass

        # build nowcasters
        vix_nc = VIXNowcaster()
        vix_nc.fit(vix_hist)

        tnx_nc = DriftNowcaster()
        tnx_nc.fit(tnx_hist)

        fx_nc = DriftNowcaster()
        fx_nc.fit(fx_hist)

        # stash KPI snapshots (so sims re-use them with a little jitter)
        last_kpis = {
            "Equities": float(eq.get("composite_sentiment", 0.0) or 0.0),
            "Gold":     float(gold.get("composite_sentiment", 0.0) or 0.0),
            "REITs":    float(reit.get("composite_sentiment", 0.0) or 0.0),
            "Bitcoin":  float(btc.get("composite_sentiment", 0.0) or 0.0),
        }

        # scenario runner inputs
        last_inputs = {
            "VIX": last_vix,
            "TNX": last_tnx,
            "DXY": last_fx,  # still pass as "DXY" key
            "KPI_Equities": last_kpis["Equities"],
            "KPI_Gold":     last_kpis["Gold"],
            "KPI_REITs":    last_kpis["REITs"],
            "KPI_Bitcoin":  last_kpis["Bitcoin"],
        }

        # precompute μ, vol for each potential regime
        mu_raw_daily_by_regime: Dict[str, pd.Series] = {}
        vol_ann_common = None
        for reg in ["risk_on", "neutral", "risk_off"]:
            mu_tmp, vol_tmp = _ensemble_mu_components(returns, decision_date, reg)
            mu_raw_daily_by_regime[reg] = mu_tmp
            vol_ann_common = vol_tmp  # ok to overwrite, same index shape

        def _regime_from_vix(vval: float) -> str:
            # this drives how often scenario says "risk_on"
            if vval >= 25.0:
                return "risk_off"
            if vval <= (16.0 if macro_env == "in" else 12.0):
                return "risk_on"
            return "neutral"

        def _shock_scalers_from_changes(tnx_bps: float, fx_pct: float) -> Tuple[float, float]:
            risk_m, gold_m = 1.0, 1.0
            if abs(tnx_bps) >= TNX_ABS_BPS_SHOCK:
                risk_m *= SHOCK_RISK_MULT
                gold_m *= SHOCK_GOLD_MULT
            if abs(fx_pct) >= FX_PCT_SHOCK:
                risk_m *= SHOCK_RISK_MULT
                gold_m *= SHOCK_GOLD_MULT
            return risk_m, gold_m

        # --- calibrated asset-level expectations per regime for this horizon ---
        exp_assets_cal_by_regime: Dict[str, pd.Series] = {}
        for reg in ["risk_on", "neutral", "risk_off"]:
            mu_raw_h_reg = mu_raw_daily_by_regime[reg] * float(H_days)

            exp_cal_reg = _calibrated_expected_assets(
                returns,
                decision_date,
                horizon_key,
                reg,
                mu_raw_h_reg,
                macro_env,
            )

            exp_assets_cal_by_regime[reg] = exp_cal_reg

        def decision_callback(inputs_s: dict):
            """
            for each simulated path we:
                - infer regime from simulated VIX
                - build weights with KPI jitter (this is w_live)
                - build messaging weights w_neutral
                - compute scenario expected H-day return using w_neutral
            """
            reg_s = _regime_from_vix(float(inputs_s["VIX_mean"]))
            mu_raw_daily_s = mu_raw_daily_by_regime[reg_s]
            vol_ann_s = vol_ann_common

            # base signal for weights
            mu_sig_s = (mu_raw_daily_s / (vol_ann_s + 1e-9)).clip(-1.0, 1.0)

            # KPI persistence w/ tiny random noise for weighting
            kpi_s = {
                "Equities": {"composite_sentiment": float(np.clip(last_kpis["Equities"] + 0.02 * np.random.normal(), -1.0, 1.0))},
                "Gold":     {"composite_sentiment": float(np.clip(last_kpis["Gold"]     + 0.02 * np.random.normal(), -1.0, 1.0))},
                "REITs":    {"composite_sentiment": float(np.clip(last_kpis["REITs"]    + 0.02 * np.random.normal(), -1.0, 1.0))},
                "Bitcoin":  {"composite_sentiment": float(np.clip(last_kpis["Bitcoin"]  + 0.02 * np.random.normal(), -1.0, 1.0))},
            }

            mu_sig_s = _apply_kpi_tilt(
                mu_sig_s,
                kpi_s["Equities"], kpi_s["Gold"], kpi_s["REITs"], kpi_s["Bitcoin"],
                TILT_ALPHA.get(reg_s, TILT_ALPHA["neutral"])
            )

            risk_mult_s, gold_mult_s = _shock_scalers_from_changes(
                float(inputs_s["TNX_5d_bps"]),
                float(inputs_s["DXY_5d_pct"])
            )
            mu_sig_s = _apply_shock_scaling(mu_sig_s, risk_mult_s, gold_mult_s)

            # convert signal -> w_live (aggressive allocation we actually trade)
            sig_vec_s = mu_sig_s.reindex(keys).fillna(0.0).to_numpy()
            if (sig_vec_s > 0).any():
                w_raw_s = sig_vec_s.clip(min=0.0)
                w_raw_s = w_raw_s / w_raw_s.sum()
            else:
                w_raw_s = np.array([0.20, 0.40, 0.25, 0.15], dtype=float)

            caps_s = _apply_regime_caps(reg_s)
            w_live = _project_capped_simplex(
                {k: float(v) for k, v in zip(keys, w_raw_s)},
                FLOORS_BASE,
                caps_s
            )

            # ----- Change 1: build neutral messaging weights every time -----
            anchor_weights = {
                "Equities": 0.30,
                "Gold":     0.25,
                "REITs":    0.25,
                "Bitcoin":  0.20,
            }
            w_neutral = _blend_to_anchor(
                w_live,
                anchor_weights=anchor_weights,
                blend_frac=0.50,  # 50% toward safety
            )

            # calibrated asset fwd returns for THIS regime (already horizon-sized)
            cal_assets = exp_assets_cal_by_regime[reg_s].copy()
            cal_assets = cal_assets.reindex(keys).fillna(0.0)

            # expected path return USING w_neutral (not w_live)
            w_vec_neutral = np.array([w_neutral[k] for k in keys], dtype=float)
            exp_s = float(np.dot(cal_assets.to_numpy(), w_vec_neutral))

            # ----- Change 2: late-cycle dampener for messaging only -----
            vix_now = float(inputs_s.get("VIX_mean", last_vix))
            rate_now = float(inputs_s.get("TNX_now", last_tnx))
            # we keep usd_now for possible future rules, not used directly here
            usd_now = float(inputs_s.get("DXY_now", last_fx))

            too_complacent = (vix_now < 13.0) and (rate_now > 3.8)
            if too_complacent and exp_s > 0.0:
                exp_s *= 0.3  # slash optimism ~70%

            # return both actual live weights and messaging expectation
            return w_live, exp_s

        # run Monte Carlo across paths
        weights_paths, exp_paths, aux = run_scenarios(
            SCENARIOS_N,
            H_days,
            last_inputs,
            {"VIX": vix_nc, "TNX": tnx_nc, "DXY": fx_nc},
            decision_callback
        )

        # median "live" weight & exp across paths
        w_med = dict(zip(keys, np.median(weights_paths, axis=0)))
        caps_today = _apply_regime_caps(regime)
        w_star = _project_capped_simplex(w_med, FLOORS_BASE, caps_today)
        w_final = _apply_turnover(w_prev, w_star, TURNOVER_LIMIT)

        exp_med = float(np.median(exp_paths))
        exp_p10 = float(np.quantile(exp_paths, 0.10))
        exp_p90 = float(np.quantile(exp_paths, 0.90))

        # optional robust brake (tail risk floor) still ONLY affects final w_final
        if (ROBUST_PCTL_FLOOR is not None) and (exp_p10 < ROBUST_PCTL_FLOOR):
            anchor = {
                "Equities": 0.25,
                "Gold":    0.40,
                "REITs":   0.20,
                "Bitcoin": 0.15,
            }
            gap = min(
                0.5,
                float(ROBUST_PCTL_FLOOR - exp_p10) / max(1e-6, abs(ROBUST_PCTL_FLOOR))
            )
            for k in keys:
                w_final[k] = (1.0 - gap) * w_final[k] + gap * anchor[k]

        realized = realized_ret

        # pretty print =================================================
        if _should_print_period(period_index, period_count):
            # No annualized hype for monthly / quarterly
            fx_label = "USD/INR" if macro_env == "in" else "DXY"

            print(
                f"{period_end.strftime('%Y-%m')} | w*: {_fmt_w_short(w_final)} "
                f"| exp_med={_fmt_pct(exp_med)} "
                f"[p10={_fmt_pct(exp_p10)}, p90={_fmt_pct(exp_p90)}] "
                f"| real={_fmt_pct(realized)}"
            )
            if SHOW_KPI_SUMMARY:
                print(
                    f"    Inputs last: VIX={last_vix:.2f}, TNX={last_tnx:.2f}%, {fx_label}={last_fx:.2f} "
                    f"| Regime={regime} | KPIs: eq={last_kpis['Equities']:+.2f} "
                    f"gold={last_kpis['Gold']:+.2f} reit={last_kpis['REITs']:+.2f} "
                    f"btc={last_kpis['Bitcoin']:+.2f}"
                )
                print(
                    f"    Scenario diagnostics: median VIX(next {H_days}d)={aux['vix_med']:.1f}, "
                    f"p90={aux['vix_p90']:.1f}; turnover={(sum(abs(w_final[k]-w_prev[k]) for k in keys) if w_prev else 0.0)*100:.1f}%"
                )

        kpi_pack = {
            "equities": eq,
            "gold": gold,
            "reit": reit,
            "bitcoin": btc,
            "regime": regime,
            "vix": vix_level,
            "exp_med": exp_med,
            "exp_p10": exp_p10,
            "exp_p90": exp_p90,
        }
        return w_final, exp_med, kpi_pack, realized_ret

    # =========================================================
    # baseline path (no scenario engine OR long horizons)
    # =========================================================
    if _should_print_period(period_index, period_count):
        H_days_int = int(_horizon_days(horizon_key))

        # For monthly / quarterly we killed annualized hype
        if horizon_key in ("monthly", "quarterly"):
            print(
                f"{period_end.strftime('%Y-%m')} | w: {_fmt_w_short(w_final_base)} "
                f"| exp={_fmt_pct(baseline_exp_h)} "
                f"| real={_fmt_pct(realized_ret)}"
            )
        else:
            ann_guess = _annualize(baseline_exp_h, H_days_int)
            print(
                f"{period_end.strftime('%Y-%m')} | w: {_fmt_w_short(w_final_base)} "
                f"| exp={_fmt_pct(baseline_exp_h)} (ann~{_fmt_pct(ann_guess)}) "
                f"| real={_fmt_pct(realized_ret)}"
            )

        if SHOW_KPI_SUMMARY:
            print(
                "    KPI: "
                f"eq={float(eq.get('composite_sentiment',0.0)):+.3f} "
                f"gold={float(gold.get('composite_sentiment',0.0)):+.3f} "
                f"reit={float(reit.get('composite_sentiment',0.0)):+.3f} "
                f"btc={float(btc.get('composite_sentiment',0.0)):+.3f}"
            )

    kpi_pack = {
        "equities": eq,
        "gold": gold,
        "reit": reit,
        "bitcoin": btc,
        "regime": regime,
        "vix": vix_level,
    }
    return w_final_base, baseline_exp_h, kpi_pack, realized_ret


# =========================
# period schedulers (monthly, quarterly, etc.)
# =========================

def _period_dates_monthly(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for m in range(1, 13):
        end = _to_ts(f"{year}-{m:02d}-01") + pd.offsets.MonthEnd(0)
        dec = end - pd.offsets.MonthEnd(1)
        out.append((_to_ts(dec), _to_ts(end)))
    return out


def _period_dates_quarterly(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    # Change 3a: include Q1 (March 31) as a printed quarter-end as well.
    q_ends = [f"{year}-03-31", f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
    out = []
    for q in q_ends:
        end = _to_ts(q)
        # decision point is previous quarter-end:
        # e.g. for Mar 31, decision is Dec 31 prev year;
        # for Jun 30, decision is Mar 31 same year; etc.
        # Use pandas QuarterEnd(1) backwards like before but aligned.
        dec = end - pd.offsets.QuarterEnd(1)
        out.append((_to_ts(dec), _to_ts(end)))
    return out


def _period_dates_halfyear(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    ends = [f"{year}-06-30", f"{year}-12-31"]
    out = []
    for e in ends:
        end = _to_ts(e)
        dec = _to_ts(f"{year - 1}-12-31") if e.endswith("06-30") else _to_ts(f"{year}-06-30")
        out.append((dec, end))
    return out


def _period_dates_year(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    end = _to_ts(f"{year}-12-31")
    dec = _to_ts(f"{year - 1}-12-31")
    return [(dec, end)]


# =========================
# reporting helpers
# =========================

def analyze_regime_distribution(year: int, returns: pd.DataFrame, macro_env: str):
    year_data = returns.loc[f"{year}-01-01":f"{year}-12-31"]
    regime_counts = {"risk_on": 0, "neutral": 0, "risk_off": 0}

    for date in year_data.index:
        regime, _, _ = _market_regime(date, macro_env)
        regime_counts[regime] += 1

    total_days = sum(regime_counts.values())
    if total_days == 0:
        print(f"No data available for regime analysis in {year}")
        return

    print(f"\n{year} REGIME ANALYSIS:")
    for regime, count in regime_counts.items():
        pct = count / total_days * 100
        print(f"  {regime}: {pct:.1f}% of days ({count} days)")


def _summarize(label: str, realized_list: List[float]):
    gross = np.prod([1.0 + r for r in realized_list]) - 1.0 if realized_list else 0.0
    ann_ret = gross  # since we're looking at that policy over 1y
    ann_vol = (
        float(np.std(realized_list, ddof=1) * np.sqrt(12.0))
        if len(realized_list) > 1
        else 0.0
    )
    sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0

    print(f"\n========== {label} ==========")
    print(
        f"Total: {_fmt_pct(gross)} | AnnRet: {_fmt_pct(ann_ret)} | "
        f"AnnVol: {100 * ann_vol:.2f}% | Sharpe: {sharpe:.2f}"
    )


def generate_multi_year_summary(all_results: dict):
    print(f"\n{'=' * 90}")
    print(f"COMPREHENSIVE MULTI-YEAR PERFORMANCE SUMMARY")
    print(f"{'=' * 90}")

    print(
        f"{'Year':<6} {'Return':<10} {'Sharpe':<8} {'Volatility':<12} "
        f"{'Best Month':<12} {'Worst Month':<12} {'Win Rate':<10}"
    )
    print("-" * 90)

    for year, results in all_results.items():
        monthly_returns = results.get('monthly_realized', [])

        if monthly_returns:
            total_return = np.prod([1.0 + r for r in monthly_returns]) - 1.0
            annual_vol = float(np.std(monthly_returns) * np.sqrt(12.0)) if len(monthly_returns) > 1 else 0.0
            sharpe = (total_return / annual_vol) if annual_vol > 1e-12 else 0.0
            best_month = max(monthly_returns) * 100
            worst_month = min(monthly_returns) * 100
            win_rate = sum(1 for r in monthly_returns if r > 0) / len(monthly_returns) * 100

            print(
                f"{year:<6} {total_return * 100:+8.2f}% "
                f"{sharpe:>6.2f}   {annual_vol * 100:>10.2f}% "
                f"{best_month:>10.2f}% {worst_month:>11.2f}% {win_rate:>8.1f}%"
            )
        else:
            print(f"{year:<6} {'No Data':<8}")


# =========================
# run 1 year
# =========================

async def run_single_year_backtest(
    returns: pd.DataFrame,
    year: int,
    charts: bool = False,
    macro_env: str = "us"
):
    """
    run monthly/quarterly/half/yearly forecasts for a single year
    and print the blocks just like the output you liked
    """
    if not validate_data_availability(returns, year):
        return {"year": year, "status": "no_data"}

    monthly_periods   = _period_dates_monthly(year)
    quarterly_periods = _period_dates_quarterly(year)
    half_periods      = _period_dates_halfyear(year)
    yearly_periods    = _period_dates_year(year)

    results = {
        'year': year,
        'monthly_realized': [],
        'quarterly_realized': [],
        'half_realized': [],
        'yearly_realized': [],
        'monthly_turnovers': []
    }

    if PRINT_MODE != "quiet":
        print(f"\nMONTHLY FORECASTS ({year}): {len(monthly_periods)} periods")
        print("=" * 60)

    w_prev: Optional[Dict[str, float]] = None
    monthly_turnovers_l1: List[float] = []

    # monthly loop
    for idx, (decision_date, period_end) in enumerate(monthly_periods, start=1):
        w, exp_h, kpi_pack, realized = await forecast_one_period(
            returns,
            decision_date,
            period_end,
            "monthly",
            w_prev,
            period_index=idx,
            period_count=len(monthly_periods),
            macro_env=macro_env,
        )
        results['monthly_realized'].append(realized)

        if w_prev is not None:
            keys = ["Equities", "Gold", "REITs", "Bitcoin"]
            l1 = float(sum(abs(w[k] - w_prev[k]) for k in keys))
            monthly_turnovers_l1.append(l1)

        w_prev = w

    results['monthly_turnovers'] = monthly_turnovers_l1

    # quarterly loop
    if PRINT_MODE != "quiet":
        print(f"\nQUARTERLY FORECASTS ({year}): {len(quarterly_periods)} periods")
        print("=" * 60)

    for idx, (dd, pe) in enumerate(quarterly_periods, start=1):
        _, _, _, realized = await forecast_one_period(
            returns,
            dd,
            pe,
            "quarterly",
            w_prev,
            idx,
            len(quarterly_periods),
            macro_env=macro_env,
        )
        results['quarterly_realized'].append(realized)

    # half-year loop
    if PRINT_MODE != "quiet":
        print(f"\nHALF-YEAR FORECASTS ({year}): {len(half_periods)} periods")
        print("=" * 60)

    for idx, (dd, pe) in enumerate(half_periods, start=1):
        w_h, exp_h, kpi_pack_h, realized_h = await forecast_one_period(
            returns,
            dd,
            pe,
            "half",
            w_prev,
            idx,
            len(half_periods),
            macro_env=macro_env,
        )
        results['half_realized'].append(realized_h)

    # yearly loop
    if PRINT_MODE != "quiet":
        print(f"\nYEARLY FORECAST ({year}): 1 period")
        print("=" * 60)

    for idx, (dd, pe) in enumerate(yearly_periods, start=1):
        w_y, exp_y, kpi_pack_y, realized_y = await forecast_one_period(
            returns,
            dd,
            pe,
            "yearly",
            w_prev,
            idx,
            1,
            macro_env=macro_env,
        )
        results['yearly_realized'].append(realized_y)

    # summary stats
    _summarize(f"MONTHLY POLICY: {year} REALIZED PERFORMANCE", results['monthly_realized'])
    if monthly_turnovers_l1:
        print(f"Avg monthly turnover: {100 * float(np.mean(monthly_turnovers_l1)):.2f}%")

    _summarize(f"QUARTERLY POLICY: {year} REALIZED PERFORMANCE", results['quarterly_realized'])
    _summarize(f"HALF-YEAR POLICY: {year} REALIZED PERFORMANCE", results['half_realized'])
    _summarize(f"YEARLY POLICY: {year} REALIZED PERFORMANCE", results['yearly_realized'])

    analyze_regime_distribution(year, returns, macro_env=macro_env)

    # charts hook (if you have investor_report.py implemented)
    # if charts and generate_investor_report is not None:
    #     generate_investor_report(...)

    return results


# =========================
# multi-year driver (optional)
# =========================

async def forecast_report_multi_year(
    years: list = [2022, 2023, 2024],
    charts: bool = False,
    macro_env: str = "us"
):
    returns = load_returns_fresh()
    returns = handle_missing_recent_data(returns)

    all_results = {}

    print(f"\n{'=' * 80}")
    print(f"MULTI-YEAR BACKTESTING ANALYSIS")
    print(f"Years: {years}")
    print(f"{'=' * 80}")

    for year in years:
        print(f"\n{'=' * 60}")
        print(f"BACKTESTING YEAR {year}")
        print(f"{'=' * 60}")

        year_results = await run_single_year_backtest(returns, year, charts, macro_env=macro_env)
        all_results[year] = year_results

        monthly_returns = year_results.get('monthly_realized', [])
        if monthly_returns:
            total_return = np.prod([1.0 + r for r in monthly_returns]) - 1.0
            annual_vol = float(np.std(monthly_returns) * np.sqrt(12.0)) if len(monthly_returns) > 1 else 0.0
            sharpe = (total_return / annual_vol) if annual_vol > 1e-12 else 0.0

            print(f"\n{year} SUMMARY:")
            print(f"Total Return: {total_return * 100:+.2f}%")
            print(f"Volatility: {annual_vol * 100:.2f}%")
            print(f"Sharpe Ratio: {sharpe:.2f}")

        print(f"{'=' * 60}")

    generate_multi_year_summary(all_results)
    return all_results


async def forecast_report_2023(
    charts: bool = False,
    prefix: Optional[str] = None,
    macro_env: str = "us"
):
    returns = load_returns_fresh()
    returns = handle_missing_recent_data(returns)
    return await run_single_year_backtest(returns, 2023, charts, macro_env=macro_env)


# =========================
# CLI
# =========================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adaptive multi-asset backtester with macro regimes, KPIs, scenarios"
    )

    p.add_argument("--charts", action="store_true", help="save investor charts (if supported)")
    p.add_argument("--prefix", type=str, default=None, help="chart filename prefix")
    p.add_argument("--print", dest="print_mode",
                   choices=["full", "summary", "quiet"], default="summary")

    # run modes
    p.add_argument("--years", nargs="+", type=int, default=[2023],
                   help="Years to backtest for comparison (ex: --years 2022 2023)")
    p.add_argument("--single-year", type=int, default=None,
                   help="Run only this year (ex: --single-year 2023)")
    p.add_argument("--compare", action="store_true",
                   help="Run multi-year comparison across --years")

    # scenarios
    p.add_argument("--scenarios", type=int, default=0,
                   help="if >0, run scenario Monte Carlo with N paths (gives w*, exp_med, p10/p90)")
    p.add_argument("--robust-pctl", type=float, default=None,
                   help="optional tail-risk brake using 10th pctl floor")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for simulations")

    # macro regime flavor
    p.add_argument("--macro", type=str, default="us", choices=["us", "in"],
                   help="us = ^VIX + DXY, in = ^INDIAVIX + USD/INR")

    return p.parse_args()


def main():
    global PRINT_MODE, SCENARIOS_N, ROBUST_PCTL_FLOOR, GLOBAL_SEED, MACRO_ENV

    args = _parse_args()
    PRINT_MODE = args.print_mode
    SCENARIOS_N = int(args.scenarios or 0)
    ROBUST_PCTL_FLOOR = args.robust_pctl
    GLOBAL_SEED = int(args.seed)
    MACRO_ENV = args.macro

    if args.single_year:
        # typical path you’re using
        print(f"Running backtest for year {args.single_year}")
        returns = load_returns_fresh()
        returns = handle_missing_recent_data(returns)
        asyncio.run(
            run_single_year_backtest(
                returns,
                args.single_year,
                charts=args.charts,
                macro_env=MACRO_ENV
            )
        )

    elif args.compare:
        print(f"Running multi-year comparison for years: {args.years}")
        asyncio.run(
            forecast_report_multi_year(
                years=args.years,
                charts=args.charts,
                macro_env=MACRO_ENV
            )
        )

    else:
        # fallback to 2023
        print("Running default 2023 backtest")
        asyncio.run(
            forecast_report_2023(
                charts=args.charts,
                prefix=args.prefix,
                macro_env=MACRO_ENV
            )
        )


if __name__ == "__main__":
    main()
