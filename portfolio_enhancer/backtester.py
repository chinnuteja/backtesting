from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import sys
from typing import Dict, Optional, Tuple, List, Callable, Awaitable
from collections import defaultdict  # NEW

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor  # GBM forecaster for per-asset ERs
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # NEW

# --- Scenario engine imports ---
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
    "Equities": "^NSEI",      # Nifty 50 proxy
    "Gold": "GLD",            # Gold ETF
    "REITs": "VNQ",           # US REITs
    "Bitcoin": "BTC-USD",
}

FLOORS_BASE = {"Equities": 0.00, "Gold": 0.00, "REITs": 0.00, "Bitcoin": 0.00}
CAPS_BASE = {"Equities": 0.50, "Gold": 0.35, "REITs": 0.30, "Bitcoin": 0.20}

# default global turnover limit (used for half/yearly mostly)
TURNOVER_LIMIT = 0.30  # L1 per decision

# ---------- SIGNAL UPGRADE ----------
MH_SPANS = [21, 63, 126]

MH_WEIGHTS_BY_REGIME = {
    "risk_on": np.array([0.65, 0.25, 0.10]),
    "neutral": np.array([0.33, 0.34, 0.33]),
    "risk_off": np.array([0.10, 0.30, 0.60]),
}

VOL_WIN = 63  # vol window for risk-adjusted score

# legacy VIX-only thresholds kept for fallback (not primary classifier anymore)
VIX_THRESHOLDS_US = {"risk_on_max": 12.0, "risk_off_min": 20.0}
VIX_THRESHOLDS_IN = {"risk_on_max": 16.0, "risk_off_min": 25.0}

# tilt strength per regime (used for WEIGHTS ONLY, not to juice forecast magnitude)
TILT_ALPHA = {"risk_on": 0.30, "neutral": 0.25, "risk_off": 0.25}

# caps vary by regime
CAPS_BY_REGIME = {
    "risk_on": {"Equities": 0.60, "Gold": 0.30, "REITs": 0.35, "Bitcoin": 0.35},
    "neutral": CAPS_BASE,
    "risk_off": {"Equities": 0.45, "Gold": 0.60, "REITs": 0.25, "Bitcoin": 0.15},
}

# macro shock detector (yields & USD strength)
SHOCK_WINDOW_D = 10
TNX_ABS_BPS_SHOCK = 30.0      # 30bp 10y jump in 10d
FX_PCT_SHOCK = 1.8            # ~1.8% USD/INR or DXY pop in 10d
SHOCK_RISK_MULT = 0.85        # cut risk assets
SHOCK_GOLD_MULT = 1.15        # boost gold

# walk-forward calibration settings for expected return
CALIB_MONTHS_BACK = 60  # 5y lookback for stability
RIDGE_L2 = 0.5          # (kept around for completeness / legacy ridge helper)
EXP_CLIP_MONTHLY = 0.25  # last-resort safety clip for portfolio 1M exp (not per-asset caps)

# KPI gating (turn off useless/noisy KPI tilts per asset)
ACTIVE_KPIS = {
    "Equities": True,
    "Gold":     True,
    "REITs":    False,  # often noisy / laggy KPI, default off
    "Bitcoin":  True,
}

# output / runtime controls
PRINT_MODE = "summary"
SHOW_KPI_SUMMARY = True

# KPI call control
SILENCE_KPI_LOGS = True
KPI_TIMEOUT_SEC = 5

# scenario engine knobs
SCENARIOS_N = 0
ROBUST_PCTL_FLOOR = None  # if not None, floor for exp_p10 scenario
GLOBAL_SEED = 42

# macro env mode: "us" (VIX + DXY) vs "in" (^INDIAVIX + USD/INR)
MACRO_ENV = "us"

# regime streak memory so we don't sit in "risk_on" forever due to complacency
REGIME_STATE = {
    "last_label": None,
    "streak_len": 0,
}

# neutral / strategic anchor (used for messaging, sizing sanity, and final blend)
NEUTRAL_ANCHOR = {
    "Equities": 0.30,
    "Gold":     0.25,
    "REITs":    0.25,
    "Bitcoin":  0.20,
}

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
    return True


def _horizon_days(h: str) -> int:
    return {"monthly": 21, "quarterly": 63, "half": 126, "yearly": 252}.get(h, 21)


def _annualize(exp_h: float, horizon_days: int) -> float:
    """
    Only use for half-year / yearly display.
    For monthly / quarterly we avoid annualizing because it explodes numbers
    and looks like hype, not research.
    """
    if horizon_days <= 0:
        return exp_h
    return (1.0 + exp_h) ** (252.0 / float(horizon_days)) - 1.0


def _blend_to_anchor(
    w_live: Dict[str, float],
    anchor_weights: Dict[str, float],
    blend_frac: float
) -> Dict[str, float]:
    """
    Blend live weights with a calmer neutral anchor.
    Used both for messaging forecast and (now) for final live weights.
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
# KPI wrappers
# =========================

async def _safe_kpi(label: str, factory: Callable[[], Awaitable[Dict]]) -> Dict:
    """
    Run external async KPI coroutines under timeout, suppress spam.
    If KPI blows up, return neutral score.
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
    """
    Daily % returns for each asset, tz-naive index.
    """
    series = []
    for name, tk in TICKERS.items():
        df = _fetch_history(tk, START_DATE, END_DATE)
        s = df["Close"].pct_change().dropna().rename(name)
        series.append(s)
    out = pd.concat(series, axis=1).sort_index()
    if hasattr(out.index, "tz_localize"):
        try:
            out.index = out.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    return out


def _yf_hist_inc(ticker: str, end_incl: pd.Timestamp, days: int) -> pd.DataFrame:
    """
    inclusive download up to end_incl for 'days' lookback.
    we over-fetch ~2x for weekends/holidays.
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
    Sanity check we actually have bars for this year,
    warn if any asset has basically no data.
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
        if len(asset_data) < 50:  # ~2.5mo trading days
            missing_assets.append(asset)

    if missing_assets:
        print(f"WARNING: Insufficient data for {missing_assets} in {year}")
        print(f"Proceeding with available data...")

    coverage = len(year_data) / 252 * 100.0
    print(f"Data coverage for {year}: {coverage:.1f}%")
    return True


def handle_missing_recent_data(returns: pd.DataFrame) -> pd.DataFrame:
    """
    forward-fill short gaps, then 0 for anything still missing
    """
    returns_filled = returns.fillna(method='ffill', limit=5)
    returns_filled = returns_filled.fillna(0.0)
    return returns_filled


def _build_price_index(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Turn daily returns into price level starting at 1.0
    """
    return (1.0 + returns).cumprod()


def _rolling_max_drawdown_component(
    price_series: pd.Series,
    window_days: int = 21
) -> float:
    """
    % drawdown from rolling N-day high.
    e.g. if index is 5% off recent high -> 0.05
    """
    if price_series.empty:
        return 0.0
    roll_max = price_series.rolling(window_days).max()
    cur = float(price_series.iloc[-1])
    ref = float(roll_max.iloc[-1]) if not np.isnan(roll_max.iloc[-1]) else cur
    if ref == 0:
        return 0.0
    dd = max(0.0, (ref - cur) / ref)
    return float(dd)


def _pct_change(a: float, b: float) -> float:
    if b == 0.0 or np.isnan(b) or np.isnan(a):
        return 0.0
    return (a / b) - 1.0


def _realized_vol_21d(returns: pd.Series) -> float:
    """
    last 21d realized vol, annualized-ish
    """
    if len(returns) < 21:
        return 0.20
    vol = float(returns.tail(21).std() * np.sqrt(252.0))
    if not np.isfinite(vol):
        vol = 0.20
    return vol


def _recent_portfolio_pain(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    w_prev: Optional[Dict[str, float]],
) -> Tuple[float, float]:
    """
    Look at last ~21 trading days BEFORE decision_date and compute:
      - trailing portfolio P&L
      - trailing portfolio vol
    using w_prev as the held weights.

    Returns (pnl_21d, vol_annualized_21d)
    If w_prev is None or not enough data, returns (0.0,0.0)
    """
    if w_prev is None:
        return 0.0, 0.0

    dd = _to_ts(decision_date)
    look_start = dd - pd.Timedelta(days=40)
    sub = returns.loc[(returns.index <= dd) & (returns.index >= look_start)].copy()
    if sub.empty:
        return 0.0, 0.0

    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    wv = np.array([w_prev.get(k, 0.0) for k in keys], dtype=float)

    # daily portfolio returns
    port_daily = sub[keys].to_numpy() @ wv
    port_daily = pd.Series(port_daily, index=sub.index)

    tail21 = port_daily.tail(21)
    if len(tail21) < 5:
        return 0.0, 0.0

    pnl_21d = float((1.0 + tail21).prod() - 1.0)
    vol_21d = float(tail21.std() * np.sqrt(252.0))
    return pnl_21d, vol_21d


# =========================
# Macro snapshot and RISK SCORE regime
# =========================

def _macro_snapshot_at(decision_date: pd.Timestamp, macro_env: str) -> Tuple[float, float, float, float, float]:
    """
    Capture macro state as of decision_date.

    Returns:
        vix_level        (implied vol / fear proxy)
        tnx_level        (10y yield, %)
        usd_fx_change10d (% USD strengthening vs INR or DXY in last ~10 days)
        rate_jump_bps10d (10y yield jump in bp last ~10 days)
        eq_drawdown_1m   (NSEI drawdown off 1m high, fractional)
    """
    end = _to_ts(decision_date)

    # VIX or India VIX
    if macro_env == "in":
        vix_ticker = "^INDIAVIX"
        fx_candidates = ["INR=X"]  # USD/INR
    else:
        vix_ticker = "^VIX"
        fx_candidates = ["DX-Y.NYB", "DXY"]

    # pull vix
    try:
        vix_df = _yf_hist_inc(vix_ticker, end, 40).dropna()
    except Exception:
        vix_df = pd.DataFrame()
    if vix_df.empty:
        vix_level = 20.0
        vix_prev = 20.0
    else:
        vix_level = float(vix_df["Close"].iloc[-1])
        vix_prev = float(vix_df["Close"].iloc[max(-10, -len(vix_df))])

    # 10y yield (^TNX)
    try:
        tnx_df = _yf_hist_inc("^TNX", end, 40).dropna()
    except Exception:
        tnx_df = pd.DataFrame()
    if tnx_df.empty:
        tnx_level = 4.0
        tnx_prev = 4.0
    else:
        tnx_level = float(tnx_df["Close"].iloc[-1])
        tnx_prev = float(tnx_df["Close"].iloc[max(-10, -len(tnx_df))])

    # FX / USD squeeze proxy (DXY or USD/INR)
    usd_fx_now = 100.0
    usd_fx_prev = 100.0
    for fx_tk in fx_candidates:
        try:
            fx_df = _yf_hist_inc(fx_tk, end, 40).dropna()
            if not fx_df.empty:
                usd_fx_now = float(fx_df["Close"].iloc[-1])
                usd_fx_prev = float(fx_df["Close"].iloc[max(-10, -len(fx_df))])
                break
        except Exception:
            pass

    usd_fx_change10d = 100.0 * (usd_fx_now - usd_fx_prev) / usd_fx_prev if usd_fx_prev else 0.0
    rate_jump_bps10d = (tnx_level - tnx_prev) * 10.0  # ^TNX *10 ≈ bp change last ~10d

    # equity drawdown off 1m high (21d lookback) on the main equity index
    try:
        eq_hist_df = _yf_hist_inc(TICKERS["Equities"], end, 40)
        eq_hist_df.index = eq_hist_df.index.tz_localize(None)
        eq_px = eq_hist_df["Close"].dropna()
        eq_drawdown_1m = _rolling_max_drawdown_component(eq_px, 21)
    except Exception:
        eq_drawdown_1m = 0.0

    return (
        float(vix_level),
        float(tnx_level),
        float(usd_fx_change10d),
        float(rate_jump_bps10d),
        float(eq_drawdown_1m),
    )


def _risk_score_classifier(
    vix_level: float,
    tnx_level: float,
    usd_fx_change10d: float,
    rate_jump_bps10d: float,
    eq_drawdown_1m: float,
    macro_env: str,
) -> float:
    """
    Produce a 0..1 'risk_score'.

    High score = stressy (risk_off).
    Low score  = calm / carry (risk_on).
    Mid        = meh (neutral).

    Components:
    - implied vol / fear (VIX / India VIX)
    - rate shock
    - USD squeeze
    - local equity drawdown
    """

    # normalize VIX-ish: below ~12 is chill, above ~25 is panic (tuned per region)
    if macro_env == "in":
        vix_low = 16.0
        vix_high = 30.0
    else:
        vix_low = 12.0
        vix_high = 25.0
    vix_term = (vix_level - vix_low) / max(1e-6, (vix_high - vix_low))
    vix_term = float(np.clip(vix_term, 0.0, 1.0))

    # rate shock: big recent jump in yields = stress
    rate_term = abs(rate_jump_bps10d) / 30.0
    rate_term = float(np.clip(rate_term, 0.0, 1.0))

    # usd squeeze: strong dollar (esp vs INR) hurts risk
    fx_term = abs(usd_fx_change10d) / 2.0
    fx_term = float(np.clip(fx_term, 0.0, 1.0))

    # drawdown term: >5% off 1m high is notable
    dd_term = eq_drawdown_1m / 0.05
    dd_term = float(np.clip(dd_term, 0.0, 1.0))

    # weighted blend
    risk_score = (
        0.40 * vix_term +
        0.25 * rate_term +
        0.20 * dd_term +
        0.15 * fx_term
    )
    return float(np.clip(risk_score, 0.0, 1.0))


def _equity_stretch_raw(end: pd.Timestamp) -> float:
    """
    How overextended equities are vs ~100d avg.
    """
    try:
        eq_hist_df = _yf_hist_inc(TICKERS["Equities"], end, 150)
        eq_hist_df.index = eq_hist_df.index.tz_localize(None)
        eq_px = eq_hist_df["Close"].dropna()
        stretch_raw, _, _ma = _stretch_stats(eq_px, 100)
    except Exception:
        stretch_raw = 0.0
    return float(stretch_raw)


def _market_regime(
    decision_date: pd.Timestamp,
    macro_env: str,
    update_streak: bool = True
) -> Tuple[str, float, float]:
    """
    Returns (regime_label, vix_level_for_print, fx_level_for_print)

    Uses composite risk_score for regime classification,
    plus a complacency-overstretch override to avoid sitting in 'risk_on' forever.

    We maintain REGIME_STATE to track how long we've stayed in risk_on.
    After ~60 consecutive calls in risk_on with stretched equities and ultra-low vol,
    we downgrade to neutral.
    """
    global REGIME_STATE

    end = _to_ts(decision_date)

    if macro_env == "in":
        fx_candidates = ["INR=X"]
    else:
        fx_candidates = ["DX-Y.NYB", "DXY"]

    # macro snapshot for score
    (
        vix_level,
        tnx_level,
        usd_fx_change10d,
        rate_jump_bps10d,
        eq_drawdown_1m,
    ) = _macro_snapshot_at(decision_date, macro_env)

    # base risk_score -> base regime
    rscore = _risk_score_classifier(
        vix_level,
        tnx_level,
        usd_fx_change10d,
        rate_jump_bps10d,
        eq_drawdown_1m,
        macro_env,
    )

    if rscore <= 0.30:
        regime_candidate = "risk_on"
    elif rscore >= 0.70:
        regime_candidate = "risk_off"
    else:
        regime_candidate = "neutral"

    # complacency override:
    # if we've been in risk_on forever, equities are stretched (>5% above 100d MA),
    # and vol is super quiet (vix <13), force neutral.
    stretch_raw_eq = _equity_stretch_raw(end)  # ~ (px/100dma -1)

    # hypothetical streak if we stayed in regime_candidate
    if REGIME_STATE["last_label"] == regime_candidate:
        hypothetical_streak = REGIME_STATE["streak_len"] + 1
    else:
        hypothetical_streak = 1

    final_regime = regime_candidate
    if (
        regime_candidate == "risk_on"
        and hypothetical_streak > 60
        and stretch_raw_eq > 0.05       # >5% above 100d avg
        and vix_level < 13.0            # super low implied vol
    ):
        final_regime = "neutral"

    # FX print handle (DXY or USD/INR for context in logs)
    fx_val = 100.0
    for fx_tk in fx_candidates:
        try:
            fx_df = _yf_hist_inc(fx_tk, end, 60).dropna()
            if not fx_df.empty:
                fx_val = float(fx_df["Close"].iloc[-1])
                break
        except Exception:
            pass

    # update streak memory if requested
    if update_streak:
        if REGIME_STATE["last_label"] == final_regime:
            REGIME_STATE["streak_len"] += 1
        else:
            REGIME_STATE["last_label"] = final_regime
            REGIME_STATE["streak_len"] = 1

    return final_regime, vix_level, fx_val


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

        if not tnx_df.empty:
            last = float(tnx_df["Close"].iloc[-1])
            prev = float(tnx_df["Close"].iloc[max(-SHOCK_WINDOW_D, -len(tnx_df))])
            bps = (last - prev) * 10.0  # ^TNX points ~10bp
            if abs(bps) >= TNX_ABS_BPS_SHOCK:
                risk_mult *= SHOCK_RISK_MULT
                gold_mult *= SHOCK_GOLD_MULT

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


# =========================
# Base allocation signal (momentum blend / vol-normalized)
# =========================

def _slice_training(returns: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    end = _to_ts(decision_date) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=3)
    return returns.loc[(returns.index >= start) & (returns.index <= end)]


def _ewma_component(sub: pd.DataFrame, span: int) -> pd.Series:
    return sub.ewm(span=span, adjust=False).mean().iloc[-1]


def _trailing_vol_ann(sub: pd.DataFrame, window: int = VOL_WIN) -> pd.Series:
    if len(sub) < window:
        return pd.Series(0.20, index=sub.columns)
    return sub.tail(window).std() * np.sqrt(252.0)


def _ensemble_mu_components(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    regime: str
) -> Tuple[pd.Series, pd.Series]:
    """
    Build regime-weighted blend of short/med/long momentum.
    This is used as the *allocation signal*, not final ER forecast.
    """
    sub = _slice_training(returns, decision_date)
    if sub.empty:
        idx = returns.columns
        return pd.Series(0.0, index=idx), pd.Series(0.20, index=idx)

    w = MH_WEIGHTS_BY_REGIME.get(regime, MH_WEIGHTS_BY_REGIME["neutral"])
    w = w / w.sum()

    comps = [_ewma_component(sub, s) for s in MH_SPANS]  # a few momentum horizons
    mat = pd.DataFrame(comps, index=MH_SPANS, columns=sub.columns)

    mu_raw_daily = (mat.T @ w)
    vol_ann = _trailing_vol_ann(sub, VOL_WIN).replace(0.0, 1e-6)

    return mu_raw_daily, vol_ann


# =========================
# Period realized returns helper
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
# Expected-return model (multifactor GBM forecaster)
# =========================

def _past_month_end(d: pd.Timestamp) -> pd.Timestamp:
    return (d + pd.offsets.MonthEnd(0)).normalize()


def _decision_dates_back(decision_date: pd.Timestamp, n_back: int) -> List[pd.Timestamp]:
    """
    Get rolling month-ends strictly before decision_date, up to ~n_back months.
    Oldest first.
    """
    end = _past_month_end(decision_date - pd.offsets.MonthBegin(1))
    out: List[pd.Timestamp] = []
    cur = end
    for _ in range(n_back):
        out.append(cur)
        cur = cur - pd.offsets.MonthEnd(1)
    return list(reversed(out))


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

    # approximate end point ~1.5x horizon days into future index,
    # then snap to last known date up to that.
    approx_end = pe + pd.Timedelta(days=int(hdays * 1.5))
    approx_end = returns.index[returns.index <= approx_end].max()

    return _realized_period_return(returns, pe, approx_end)


def _stretch_stats(px: pd.Series, lookback: int = 100) -> Tuple[float, float, float]:
    """
    Returns:
      stretch_raw = (px_now / ma_lookback - 1)
      z_stretch   = (px_now - mean) / std
      ma_now      (for debug)
    """
    if px.empty:
        return 0.0, 0.0, 1.0
    last_px = float(px.iloc[-1])
    window = px.tail(lookback)
    ma = float(window.mean()) if len(window) else last_px
    std = float(window.std()) if len(window) > 1 else 1.0
    if std == 0.0:
        std = 1.0

    stretch_raw = (last_px / ma - 1.0) if ma != 0 else 0.0
    z_stretch = (last_px - ma) / std
    return float(stretch_raw), float(z_stretch), float(ma)


def _get_features_snapshot_per_asset(
    asset: str,
    price_idx: pd.Series,
    ret_series: pd.Series,
    vix_level: float,
    tnx_level: float
) -> Dict[str, float]:
    """
    Build per-asset feature vector at this decision date.
    Includes a trailing vol field we can use for vol-normalized training on long horizons.
    """
    # helper indices
    last_i = len(price_idx) - 1
    i_21  = max(0, last_i - 21)
    i_63  = max(0, last_i - 63)
    i_252 = max(0, last_i - 252)

    # 3m momentum (~63d)
    try:
        px_last = float(price_idx.iloc[last_i])
        px_63   = float(price_idx.iloc[i_63])
        mom_3m = (px_last / px_63 - 1.0) if px_63 else 0.0
    except Exception:
        mom_3m = 0.0

    # 12m-1m momentum
    try:
        px_252 = float(price_idx.iloc[i_252])
        px_21  = float(price_idx.iloc[i_21])
        if px_252:
            full_12m = (px_last / px_252) - 1.0
            skip_last_month = (px_21 / px_252) - 1.0
            mom_12m = full_12m - skip_last_month
        else:
            mom_12m = 0.0
    except Exception:
        mom_12m = 0.0

    # stretch vs 100d MA, plus z-score
    stretch_raw, stretch_z, _ma_100 = _stretch_stats(price_idx, 100)

    # drawdown from 1m high
    drawdown_1m = _rolling_max_drawdown_component(price_idx, 21)

    # realized vol (annualized-ish) from last ~21d
    realized_vol = _realized_vol_21d(ret_series)  # ~annualized

    # realized vol vs implied vol => VRP-ish
    vrp_like = vix_level - (realized_vol * 100.0)

    # trend accel
    trend_accel = mom_3m - mom_12m

    # macro_spread proxy by asset
    if asset in ["Equities", "REITs", "Gold"]:
        macro_spread = -tnx_level  # high rates = headwind for duration/risk assets
    elif asset == "Bitcoin":
        macro_spread = vrp_like     # risk appetite proxy for BTC
    else:
        macro_spread = -tnx_level

    return {
        "mom_3m": float(mom_3m),
        "mom_12m": float(mom_12m),
        "stretch_raw": float(stretch_raw),
        "stretch_z": float(stretch_z),
        "drawdown_1m": float(drawdown_1m),
        "vix_level": float(vix_level),
        "rate_level": float(tnx_level),
        "vrp_like": float(vrp_like),
        "trend_accel": float(trend_accel),
        "macro_spread": float(macro_spread),
        "trail_vol": float(realized_vol),
    }


def _get_features_snapshot_all_assets(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    vix_level: float,
    tnx_level: float
) -> Dict[str, Dict[str, float]]:
    """
    For *all* assets at decision_date, build feature dict.
    Includes "trail_vol".
    """
    dd = _to_ts(decision_date)
    hist = returns.loc[returns.index <= dd].copy()

    out: Dict[str, Dict[str, float]] = {}

    if hist.empty:
        for asset in returns.columns:
            out[asset] = {
                "mom_3m": 0.0,
                "mom_12m": 0.0,
                "stretch_raw": 0.0,
                "stretch_z": 0.0,
                "drawdown_1m": 0.0,
                "vix_level": float(vix_level),
                "rate_level": float(tnx_level),
                "vrp_like": float(vix_level) - 20.0,
                "trend_accel": 0.0,
                "macro_spread": -tnx_level,
                "trail_vol": 0.20,
            }
        return out

    price_idx = _build_price_index(hist)

    for asset in hist.columns:
        px_asset = price_idx[asset].dropna()
        rt_asset = hist[asset].dropna()
        if px_asset.empty or rt_asset.empty:
            out[asset] = {
                "mom_3m": 0.0,
                "mom_12m": 0.0,
                "stretch_raw": 0.0,
                "stretch_z": 0.0,
                "drawdown_1m": 0.0,
                "vix_level": float(vix_level),
                "rate_level": float(tnx_level),
                "vrp_like": float(vix_level) - 20.0,
                "trend_accel": 0.0,
                "macro_spread": -tnx_level,
                "trail_vol": 0.20,
            }
        else:
            out[asset] = _get_features_snapshot_per_asset(
                asset,
                px_asset,
                rt_asset,
                vix_level,
                tnx_level
            )

    return out


def _collect_training_examples_multifactor(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    horizon_key: str,
    macro_env: str,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Build rolling supervised samples for each asset:
        X_hist: feature vectors at each past decision point
        y_hist: realized forward return from that point over horizon_key
    Walk-forward only: we NEVER look past decision_date.
    """
    dates = _decision_dates_back(decision_date, CALIB_MONTHS_BACK)

    out: Dict[str, Dict[str, List[float]]] = {}
    for asset in returns.columns:
        out[asset] = {"X": [], "y": []}

    vol_norm = horizon_key in ("half", "yearly")

    for dec in dates:
        # macro snapshot as of that DECISION time
        (
            vix_level_dec,
            tnx_level_dec,
            usd_fx_change10d,
            rate_jump_bps10d,
            eq_drawdown_1m,
        ) = _macro_snapshot_at(dec, macro_env)

        feats_dec_all = _get_features_snapshot_all_assets(
            returns,
            dec,
            vix_level_dec,
            tnx_level_dec,
        )

        realized_fwd = _forward_realized_from_dec(returns, dec, horizon_key)

        for asset in returns.columns:
            f = feats_dec_all.get(asset)
            if f is None:
                continue

            y_val = float(realized_fwd.get(asset, 0.0))

            if vol_norm:
                denom = max(f.get("trail_vol", 0.20), 1e-6)
                y_train = y_val / denom
            else:
                y_train = y_val

            x_row = [
                f["mom_3m"],
                f["mom_12m"],
                f["stretch_raw"],
                f["stretch_z"],
                f["drawdown_1m"],
                f["vix_level"],
                f["rate_level"],
                f["vrp_like"],
                f["trend_accel"],
                f["macro_spread"],
            ]

            out[asset]["X"].append(x_row)
            out[asset]["y"].append(y_train)

    return out


def _ridge_fit_predict(
    X_hist: np.ndarray,
    y_hist: np.ndarray,
    x_now: np.ndarray,
    l2: float
) -> float:
    """
    tiny ridge y ~ a*x + b for 1D x.
    kept for backward compat if we ever want linear fallback.
    """
    if X_hist.ndim == 2 and X_hist.shape[1] > 1:
        # multifeature ridge could go here if we wanted it,
        # but GBM replaces it. We keep this mainly as legacy.
        pass

    if X_hist.ndim != 1:
        # fallback if not 1D
        return float(np.mean(y_hist)) if len(y_hist) else 0.0

    if len(X_hist) < 2:
        return float(np.mean(y_hist)) if len(y_hist) else 0.0

    xm = float(np.mean(X_hist))
    ym = float(np.mean(y_hist))
    xt = X_hist - xm
    yt = y_hist - ym

    den = float(np.dot(xt, xt) + l2)
    a = float(np.dot(xt, yt) / den) if den > 1e-12 else 1.0
    a = float(np.clip(a, 0.25, 2.0))
    b = ym - a * xm

    return float(a * float(x_now) + b)


def _gbm_fit_predict(
    X_hist: np.ndarray,
    y_hist: np.ndarray,
    x_now: np.ndarray,
    random_seed: int,
) -> float:
    """
    Train a small gradient boosting regressor on (X_hist -> y_hist)
    and predict y for x_now.

    - nonlinear (trees), so it can learn stuff like:
      "if VIX is very high AND drawdown is huge -> bounce next month"
    - rolling: we refit every decision_date with only past data
    - no leakage: we never use future returns

    If there's not enough data or the target is basically constant,
    we fall back to a simple mean.
    """
    # safety: need at least ~24 samples
    if X_hist.shape[0] < 24:
        return float(np.mean(y_hist)) if len(y_hist) else 0.0

    # if there's no variation in y, model can't learn slope
    if np.std(y_hist) < 1e-6:
        return float(np.mean(y_hist))

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        random_state=int(random_seed),
    )

    model.fit(X_hist, y_hist)
    pred_val = model.predict(x_now.reshape(1, -1))
    return float(pred_val[0])


def _vix_risk_premium_adjust(asset: str, base_guess: float, vix_now: float) -> float:
    """
    Volatility / risk-premium tweak. Fear today tends to mean higher forward return
    for high-beta assets. Extreme complacency should haircut forward return.
    """
    adj = float(base_guess)
    if asset in ["Equities", "Bitcoin"]:
        if vix_now > 25.0:
            adj += 0.02  # +2% over the horizon for panic regimes
        elif vix_now < 12.0:
            adj -= 0.01  # shave a bit if vol is insanely low / complacent
    return adj


def _cap_monthly_asset_expectations(asset: str, val: float) -> float:
    """
    Sanity cap per-asset 1M expectations.
    """
    if asset in ["Equities", "REITs"]:
        lo, hi = -0.05, 0.05
    elif asset == "Gold":
        lo, hi = -0.03, 0.03
    elif asset == "Bitcoin":
        lo, hi = -0.15, 0.15
    else:
        lo, hi = -0.05, 0.05
    return float(np.clip(val, lo, hi))


def _cap_long_expectations(asset: str, val: float, horizon_key: str) -> float:
    """
    For half-year (126d) and yearly (252d) forecasts.
    Keep numbers realistic for communication / sizing.
    """
    if horizon_key == "half":
        if asset in ["Equities", "REITs"]:
            lo, hi = -0.20, 0.20
        elif asset == "Gold":
            lo, hi = -0.10, 0.10
        elif asset == "Bitcoin":
            lo, hi = -0.50, 0.50
        else:
            lo, hi = -0.20, 0.20
    elif horizon_key == "yearly":
        if asset in ["Equities", "REITs"]:
            lo, hi = -0.40, 0.40
        elif asset == "Gold":
            lo, hi = -0.20, 0.20
        elif asset == "Bitcoin":
            lo, hi = -0.80, 0.80
        else:
            lo, hi = -0.40, 0.40
    else:
        return val

    return float(np.clip(val, lo, hi))


def _calibrated_expected_assets(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    horizon_key: str,
    macro_env: str,
    vix_level_now: float,
    tnx_level_now: float,
) -> pd.Series:
    """
    ML forecaster for per-asset forward returns at this decision point.

    Pipeline:
    1. build training set up to (and not including) decision_date
       using _collect_training_examples_multifactor()
       -> for each asset: X_hist (features), y_hist (forward realized return)

    2. build today's feature vector for each asset using
       _get_features_snapshot_all_assets()

    3. fit GradientBoostingRegressor per asset and predict next horizon return.
       For long horizons (half/yearly) we train on vol-normalized target
       and then scale prediction back up by current vol.

    4. apply a volatility risk-premium tweak from VIX for high-beta assets
       (_vix_risk_premium_adjust)

    5. sanity-cap extreme forecasts with _cap_monthly_asset_expectations()
       or _cap_long_expectations() depending on horizon.
    """
    # 1. rolling historical training data (X/y per asset)
    train_data = _collect_training_examples_multifactor(
        returns,
        decision_date,
        horizon_key,
        macro_env,
    )

    # 2. "right now" snapshot for all assets
    feats_now_all = _get_features_snapshot_all_assets(
        returns,
        decision_date,
        vix_level_now,
        tnx_level_now,
    )

    vol_norm = horizon_key in ("half", "yearly")

    out: Dict[str, float] = {}

    for asset in returns.columns:
        block = train_data.get(asset, None)
        if not block:
            out[asset] = 0.0
            continue

        X_hist_list = block["X"]  # list of feature vectors
        y_hist_list = block["y"]  # list of realized forward returns (possibly vol-normed)

        if (len(X_hist_list) < 2) or (len(y_hist_list) < 2):
            base_guess = float(np.mean(y_hist_list)) if y_hist_list else 0.0
            final_guess = _vix_risk_premium_adjust(asset, base_guess, vix_level_now)

            if horizon_key == "monthly":
                final_guess = _cap_monthly_asset_expectations(asset, final_guess)
            elif horizon_key in ("half", "yearly"):
                final_guess = _cap_long_expectations(asset, final_guess, horizon_key)

            out[asset] = float(final_guess)
            continue

        # arrays
        X_hist = np.array(X_hist_list, dtype=float)
        y_hist = np.array(y_hist_list, dtype=float)

        f_now = feats_now_all.get(asset, None)
        if f_now is None:
            base_guess = float(np.mean(y_hist_list))
            final_guess = _vix_risk_premium_adjust(asset, base_guess, vix_level_now)

            if horizon_key == "monthly":
                final_guess = _cap_monthly_asset_expectations(asset, final_guess)
            elif horizon_key in ("half", "yearly"):
                final_guess = _cap_long_expectations(asset, final_guess, horizon_key)

            out[asset] = float(final_guess)
            continue

        x_now = np.array([
            f_now["mom_3m"],
            f_now["mom_12m"],
            f_now["stretch_raw"],
            f_now["stretch_z"],
            f_now["drawdown_1m"],
            f_now["vix_level"],
            f_now["rate_level"],
            f_now["vrp_like"],
            f_now["trend_accel"],
            f_now["macro_spread"],
        ], dtype=float)

        # 3. fit ML model & predict
        ml_pred = _gbm_fit_predict(
            X_hist,
            y_hist,
            x_now,
            random_seed=GLOBAL_SEED,
        )

        # if we trained on vol-normalized y, rescale back up
        if vol_norm:
            trail_vol_now = max(f_now.get("trail_vol", 0.20), 1e-6)
            ml_pred = ml_pred * trail_vol_now

        # 4. apply volatility risk-premium tweak (fear gets paid)
        ml_pred_adj = _vix_risk_premium_adjust(asset, ml_pred, vix_level_now)

        # 5. cap crazy numbers
        if horizon_key == "monthly":
            ml_pred_adj = _cap_monthly_asset_expectations(asset, ml_pred_adj)
        elif horizon_key in ("half", "yearly"):
            ml_pred_adj = _cap_long_expectations(asset, ml_pred_adj, horizon_key)

        out[asset] = float(ml_pred_adj)

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
    # sentiment scores assumed in [-1,+1]
    scores = {
        "Equities": float(eq.get("composite_sentiment", 0.0) or 0.0) if ACTIVE_KPIS.get("Equities", True) else 0.0,
        "Gold":     float(gold.get("composite_sentiment", 0.0) or 0.0) if ACTIVE_KPIS.get("Gold", True) else 0.0,
        "REITs":    float(reit.get("composite_sentiment", 0.0) or 0.0) if ACTIVE_KPIS.get("REITs", True) else 0.0,
        "Bitcoin":  float(btc.get("composite_sentiment", 0.0) or 0.0) if ACTIVE_KPIS.get("Bitcoin", True) else 0.0,
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
    """
    L1 turnover throttle vs previous live weights.
    """
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

    # final projection for safety (global CAPS_BASE; we'll re-project again later w/ regime caps)
    return _project_capped_simplex(out, FLOORS_BASE, CAPS_BASE)


# =========================
# Core per-period forecast
# =========================

async def forecast_one_period(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    period_end: pd.Timestamp,
    horizon_key: str,
    w_prev: Optional[Dict[str, float]],
    prev_exp_assets_smooth: Optional[pd.Series],
    period_index: int,
    period_count: int,
    macro_env: str,
    perf_log: list,  # collects debug rows across the backtest
) -> Tuple[
    Dict[str, float],
    float,
    Dict[str, Dict],
    float,
    pd.Series,
    pd.Series,
    pd.Series
]:
    """
    does EVERYTHING for one decision point:
      - figure regime / macro state
      - build blended allocation signal (legacy momentum/vol + smoothed ML ERs)
      - get KPI tilts (sentiment), shock scaling
      - construct weights with caps / turnover / anchor / downside throttle
      - run scenario engine to get exp_med/p10/p90 for that allocation
      - compute realized perf
      - log model forecast vs realized per-asset into perf_log
      - print pretty line

    returns (
        weights_used,
        expected_return_used_for_print,
        kpi_pack,
        realized_portfolio_ret,
        exp_assets_smooth_out,
        exp_assets_series,      # NEW: per-asset forecast this period
        realized_vec            # NEW: per-asset realized fwd return this period
    )
    """

    # 1. current macro regime / snapshot / "shock" multipliers
    regime, vix_level, fx_level = _market_regime(decision_date, macro_env)
    alpha = TILT_ALPHA.get(regime, TILT_ALPHA["neutral"])
    risk_mult, gold_mult = _shock_multipliers(decision_date, macro_env)

    # 2. base ensemble signal (momentum blend / vol norm)
    mu_raw_daily, vol_ann = _ensemble_mu_components(returns, decision_date, regime)
    legacy_sig = (mu_raw_daily / (vol_ann + 1e-9)).clip(-1.0, 1.0)

    # 3. macro snapshot for ML ER model
    (
        vix_level_now,
        tnx_level_now,
        _usd_fx_change10d_now,
        _rate_jump_bps10d_now,
        _eq_drawdown_1m_now,
    ) = _macro_snapshot_at(decision_date, macro_env)

    # 4. calibrated per-asset expected returns via GBM (raw)
    exp_assets_series = _calibrated_expected_assets(
        returns,
        decision_date,
        horizon_key,
        macro_env,
        vix_level_now,
        tnx_level_now,
    )

    # 5. smooth ML forecast across time for this horizon
    if prev_exp_assets_smooth is None:
        exp_assets_smooth_out = exp_assets_series.copy()
    else:
        prev_aligned = prev_exp_assets_smooth.reindex(exp_assets_series.index).fillna(0.0)
        exp_assets_smooth_out = 0.6 * prev_aligned + 0.4 * exp_assets_series

    # 6. turn smoothed ML ER into normalized signal in [-1,1]
    ml_sig_raw = exp_assets_smooth_out.clip(-0.10, 0.10)  # cap insane stuff at +/-10%
    ml_norm_den = ml_sig_raw.abs().max() + 1e-9
    ml_sig_unit = ml_sig_raw / ml_norm_den  # scale to roughly [-1,1]

    # 7. blend legacy momentum/vol signal with ML ER view
    blended_sig_raw = 0.5 * legacy_sig + 0.5 * ml_sig_unit

    # 8. KPIs (async sentiment)
    eq   = await _eq_kpi(decision_date, horizon_key)
    gold = await _gold_kpi(decision_date, horizon_key)
    reit = await _reit_kpi(decision_date, horizon_key)
    btc  = await _btc_kpi(decision_date, horizon_key)

    # 9. apply KPI tilt (for WEIGHTS ONLY) and macro shock scaling
    alloc_sig = _apply_kpi_tilt(blended_sig_raw, eq, gold, reit, btc, alpha)
    alloc_sig = _apply_shock_scaling(alloc_sig, risk_mult, gold_mult)

    # 10. convert alloc_sig -> raw weights (long-only positives)
    keys_list = ["Equities", "Gold", "REITs", "Bitcoin"]
    sig_vec = alloc_sig.reindex(keys_list).fillna(0.0).to_numpy()

    if (sig_vec > 0).any():
        w_raw = sig_vec.clip(min=0.0)
        w_raw = w_raw / w_raw.sum()
    else:
        # fallback barbell if literally everything <=0
        w_raw = np.array([0.20, 0.40, 0.25, 0.15], dtype=float)

    w0 = {k: float(v) for k, v in zip(keys_list, w_raw)}

    # 11. caps for today's regime
    caps_today = _apply_regime_caps(regime)

    # 12. project into capped simplex
    w_proj = _project_capped_simplex(w0, FLOORS_BASE, caps_today)

    # 13. local turnover limit depends on horizon
    if horizon_key == "monthly":
        turnover_limit_local = 0.15
    elif horizon_key == "quarterly":
        turnover_limit_local = 0.20
    else:
        turnover_limit_local = TURNOVER_LIMIT

    # 14. trailing pain snapshot for downside discipline later
    recent_pnl_21d, recent_vol_21d = _recent_portfolio_pain(
        returns,
        decision_date,
        w_prev
    )

    # 15. realized vector for this horizon (we'll dot with final weights later)
    realized_vec = _realized_period_return(returns, decision_date, period_end)

    # --- per-decision log for diagnostics (not used for metrics print, safe to keep)
    hist_up_to_dec = returns.loc[returns.index <= decision_date]
    price_idx_full = _build_price_index(hist_up_to_dec)
    spot_row = price_idx_full.iloc[-1]

    for asset in returns.columns:
        pred_ret = float(exp_assets_series.get(asset, 0.0))
        real_ret = float(realized_vec.get(asset, 0.0))

        spot_price_idx = float(spot_row.get(asset, np.nan))

        pred_price_idx = spot_price_idx * (1.0 + pred_ret)
        real_price_idx = spot_price_idx * (1.0 + real_ret)

        perf_log.append({
            "horizon": horizon_key,
            "decision_date": pd.Timestamp(decision_date),
            "period_end": pd.Timestamp(period_end),
            "asset": asset,
            "spot_idx": spot_price_idx,
            "pred_ret": pred_ret,
            "real_ret": real_ret,
            "pred_idx_fwd": pred_price_idx,
            "real_idx_fwd": real_price_idx,
        })
    # --- end perf_log append ---

    # =========================================================
    # scenario engine path for monthly/quarterly
    # =========================================================
    if SCENARIOS_N and SCENARIOS_N > 0 and horizon_key in ("monthly", "quarterly"):
        np.random.seed(GLOBAL_SEED + int(period_index))
        H_days = int(_horizon_days(horizon_key))

        def _safe_last(hist: pd.Series, default_val: float = 0.0) -> float:
            return float(hist.iloc[-1]) if len(hist) else default_val

        # pull macro histories to seed simulators
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
                fx_df = _yf_hist_inc(tk, _to_ts(decision_date), 200)
                cand = fx_df["Close"].dropna()
                if len(cand):
                    fx_hist = cand
                    last_fx = _safe_last(cand, default_val=100.0)
                    break
            except Exception:
                pass

        # nowcasters for scenario sim
        vix_nc = VIXNowcaster()
        vix_nc.fit(vix_hist)

        tnx_nc = DriftNowcaster()
        tnx_nc.fit(tnx_hist)

        fx_nc = DriftNowcaster()
        fx_nc.fit(fx_hist)

        # stash KPI snapshots for logging / run_scenarios interface
        last_kpis = {
            "Equities": float(eq.get("composite_sentiment", 0.0) or 0.0),
            "Gold":     float(gold.get("composite_sentiment", 0.0) or 0.0),
            "REITs":    float(reit.get("composite_sentiment", 0.0) or 0.0),
            "Bitcoin":  float(btc.get("composite_sentiment", 0.0) or 0.0),
        }

        # scenario runner inputs (starting state)
        last_inputs = {
            "VIX": last_vix,
            "TNX": last_tnx,
            "DXY": last_fx,  # key "DXY" for sim (DXY or USD/INR)
            "KPI_Equities": last_kpis["Equities"],
            "KPI_Gold":     last_kpis["Gold"],
            "KPI_REITs":    last_kpis["REITs"],
            "KPI_Bitcoin":  last_kpis["Bitcoin"],
        }

        # decision_callback:
        # For each simulated path we do NOT recompute fresh weights.
        # We keep w_proj constant (this is our intended allocation),
        # and only compute exp_s under that path's macro state.
        def decision_callback(inputs_s: dict):
            vix_sim = float(inputs_s.get("VIX_mean", last_vix))
            tnx_sim = float(inputs_s.get("TNX_mean", last_tnx))

            # neutralized messaging weight vector (safer mix)
            w_neutral = _blend_to_anchor(
                w_proj,
                NEUTRAL_ANCHOR,
                blend_frac=0.50  # 50% toward safety for expectation messaging
            )

            cal_assets = exp_assets_series.copy().reindex(keys_list).fillna(0.0)

            w_vec_neutral = np.array([w_neutral[k] for k in keys_list], dtype=float)
            exp_s = float(np.dot(cal_assets.to_numpy(), w_vec_neutral))

            # late-cycle dampener for EXPECTATION ONLY:
            # if vol super low and rates elevated, slash optimism
            if (vix_sim < 13.0) and (tnx_sim > 3.8) and (exp_s > 0.0):
                exp_s *= 0.3

            # weights_paths just echo our intended allocation
            return w_proj, exp_s

        # run Monte Carlo
        weights_paths, exp_paths, aux = run_scenarios(
            SCENARIOS_N,
            H_days,
            last_inputs,
            {"VIX": vix_nc, "TNX": tnx_nc, "DXY": fx_nc},
            decision_callback
        )

        # median scenario weight (all same -> returns w_proj effectively)
        w_med = dict(zip(
            keys_list,
            np.median(weights_paths, axis=0)
        ))

        # start from regime-capped projection of the median scenario weight
        w_star = _project_capped_simplex(w_med, FLOORS_BASE, caps_today)

        # scenario distribution stats
        exp_med = float(np.median(exp_paths))
        exp_p10 = float(np.quantile(exp_paths, 0.10))
        exp_p90 = float(np.quantile(exp_paths, 0.90))

        # downside throttle:
        defensive_trigger = False
        if ROBUST_PCTL_FLOOR is not None and exp_p10 < ROBUST_PCTL_FLOOR:
            defensive_trigger = True
        if (recent_pnl_21d < -0.02) and (exp_p10 < -0.02):
            defensive_trigger = True

        if defensive_trigger:
            w_def = dict(w_star)
            for rk in ["Equities", "REITs", "Bitcoin"]:
                w_def[rk] = w_def.get(rk, 0.0) * 0.7
            # shift leftover into Gold
            total_tmp = sum(w_def.values())
            if total_tmp < 1.0:
                w_def["Gold"] = w_def.get("Gold", 0.0) + (1.0 - total_tmp)

            w_star = _project_capped_simplex(w_def, FLOORS_BASE, caps_today)

        # always blend live weights toward neutral anchor (risk budget discipline)
        w_star = _blend_to_anchor(
            w_star,
            NEUTRAL_ANCHOR,
            blend_frac=0.35  # ~1/3rd toward "core book"
        )
        w_star = _project_capped_simplex(w_star, FLOORS_BASE, caps_today)

        # apply turnover throttle vs prev live book
        w_after_turn = _apply_turnover(w_prev, w_star, turnover_limit_local)
        # reproject to today's regime caps (not the global caps)
        w_final = _project_capped_simplex(w_after_turn, FLOORS_BASE, caps_today)

        # realized perf for the horizon under final weights
        realized_ret = float(
            np.dot(
                realized_vec.reindex(keys_list).values,
                np.array([w_final[k] for k in keys_list], dtype=float)
            )
        )

        # pretty print =================================================
        if _should_print_period(period_index, period_count):
            fx_label = "USD/INR" if macro_env == "in" else "DXY"

            # NOTE: we do NOT annualize monthly/quarterly expectations anymore
            print(
                f"{period_end.strftime('%Y-%m')} | w*: {_fmt_w_short(w_final)} "
                f"| exp_med={_fmt_pct(exp_med)} "
                f"[p10={_fmt_pct(exp_p10)}, p90={_fmt_pct(exp_p90)}] "
                f"| real={_fmt_pct(realized_ret)}"
            )
            if SHOW_KPI_SUMMARY:
                asset_er_str = (
                    f"E={_fmt_pct(exp_assets_series.get('Equities',0.0))} "
                    f"G={_fmt_pct(exp_assets_series.get('Gold',0.0))} "
                    f"R={_fmt_pct(exp_assets_series.get('REITs',0.0))} "
                    f"B={_fmt_pct(exp_assets_series.get('Bitcoin',0.0))}"
                )

                print(
                    f"    Regime={regime} | VIX={vix_level:.2f} "
                    f"TNX={tnx_level_now:.2f}% {fx_label}={fx_level:.2f} "
                    f"| KPIs: eq={last_kpis['Equities']:+.2f} "
                    f"gold={last_kpis['Gold']:+.2f} reit={last_kpis['REITs']:+.2f} "
                    f"btc={last_kpis['Bitcoin']:+.2f}"
                )
                print(
                    f"    Model ER({H_days}d): {asset_er_str}"
                )
                l1_turnover = (
                    sum(abs(w_final[k] - (w_prev[k] if w_prev else 0.0)) for k in keys_list) * 100.0
                    if w_prev else 0.0
                )
                print(
                    f"    Scenario diagnostics: "
                    f"median VIX(next {H_days}d)={aux.get('vix_med',np.nan):.1f}, "
                    f"p90={aux.get('vix_p90',np.nan):.1f}; "
                    f"turnover={l1_turnover:.1f}%"
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
        return (
            w_final,
            exp_med,
            kpi_pack,
            realized_ret,
            exp_assets_smooth_out,
            exp_assets_series,
            realized_vec,
        )

    # =========================================================
    # baseline path (no scenario engine OR long horizons)
    # =========================================================

    # always blend weights toward neutral anchor BEFORE turnover
    w_anchor = _blend_to_anchor(
        w_proj,
        NEUTRAL_ANCHOR,
        blend_frac=0.35
    )
    w_anchor = _project_capped_simplex(w_anchor, FLOORS_BASE, caps_today)

    w_after_turn = _apply_turnover(w_prev, w_anchor, turnover_limit_local)
    w_final_base = _project_capped_simplex(w_after_turn, FLOORS_BASE, caps_today)

    # compute expected return for final base weights
    exp_assets_for_dot = exp_assets_series.reindex(keys_list).fillna(0.0).to_numpy()
    w_fb_vec = np.array([w_final_base[k] for k in keys_list], dtype=float)
    baseline_exp_h_final = float(np.dot(exp_assets_for_dot, w_fb_vec))

    if horizon_key == "monthly":
        baseline_exp_h_final = float(np.clip(baseline_exp_h_final, -EXP_CLIP_MONTHLY, EXP_CLIP_MONTHLY))

    realized_ret_final = float(
        np.dot(
            realized_vec.reindex(keys_list).values,
            w_fb_vec
        )
    )

    if _should_print_period(period_index, period_count):
        H_days_int = int(_horizon_days(horizon_key))

        # for half/yearly we still show annualized; for month/quarter we don't
        if horizon_key in ("half", "yearly"):
            ann_guess = _annualize(baseline_exp_h_final, H_days_int)
            ann_str = f"(ann~{_fmt_pct(ann_guess)}) "
        else:
            ann_str = ""

        print(
            f"{period_end.strftime('%Y-%m')} | w: {_fmt_w_short(w_final_base)} "
            f"| exp={_fmt_pct(baseline_exp_h_final)} {ann_str}"
            f"| real={_fmt_pct(realized_ret_final)}"
        )
        if SHOW_KPI_SUMMARY:
            asset_er_str = (
                f"E={_fmt_pct(exp_assets_series.get('Equities',0.0))} "
                f"G={_fmt_pct(exp_assets_series.get('Gold',0.0))} "
                f"R={_fmt_pct(exp_assets_series.get('REITs',0.0))} "
                f"B={_fmt_pct(exp_assets_series.get('Bitcoin',0.0))}"
            )
            print(
                "    KPI: "
                f"eq={float(eq.get('composite_sentiment',0.0)):+.3f} "
                f"gold={float(gold.get('composite_sentiment',0.0)):+.3f} "
                f"reit={float(reit.get('composite_sentiment',0.0)):+.3f} "
                f"btc={float(btc.get('composite_sentiment',0.0)):+.3f}"
            )
            print(
                f"    Model ER({H_days_int}d): {asset_er_str}"
            )

    kpi_pack = {
        "equities": eq,
        "gold": gold,
        "reit": reit,
        "bitcoin": btc,
        "regime": regime,
        "vix": vix_level,
    }
    return (
        w_final_base,
        baseline_exp_h_final,
        kpi_pack,
        realized_ret_final,
        exp_assets_smooth_out,
        exp_assets_series,
        realized_vec,
    )


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
    # include Q1 end (Mar 31) so we get Q1 even if we don't have full lookahead
    q_ends = [f"{year}-03-31", f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
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
        regime, _, _ = _market_regime(date, macro_env, update_streak=False)
        regime_counts[regime] += 1

    total_days = sum(regime_counts.values())
    if total_days == 0:
        print(f"No data available for regime analysis in {year}")
        return

    print(f"\n{year} REGIME ANALYSIS:")
    for regime_label, count in regime_counts.items():
        pct = count / total_days * 100
        print(f"  {regime_label}: {pct:.1f}% of days ({count} days)")


def _summarize(label: str, realized_list: List[float]):
    gross = np.prod([1.0 + r for r in realized_list]) - 1.0 if realized_list else 0.0
    ann_ret = gross  # since we're looking at that policy over ~1y block
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
            best_month = max(monthly_returns) * 100 if monthly_returns else 0.0
            worst_month = min(monthly_returns) * 100 if monthly_returns else 0.0
            win_rate = sum(1 for r in monthly_returns if r > 0) / len(monthly_returns) * 100 if monthly_returns else 0.0

            print(
                f"{year:<6} {total_return * 100:+8.2f}% "
                f"{sharpe:>6.2f}   {annual_vol * 100:>10.2f}% "
                f"{best_month:>10.2f}% {worst_month:>11.2f}% {win_rate:>8.1f}%"
            )
        else:
            print(f"{year:<6} {'No Data':<8}")


# === NEW: metrics helpers for calibration ===================

def compute_metrics(y_pred_list, y_real_list):
    """
    Compute avg_pred, avg_real, bias, MAE, MSE, R2, corr
    with guards so we don't crash on small sample sizes.
    """
    y_pred = np.array(y_pred_list, dtype=float)
    y_real = np.array(y_real_list, dtype=float)

    n = min(len(y_pred), len(y_real))

    if n == 0:
        return {
            "avg_pred": np.nan,
            "avg_real": np.nan,
            "bias": np.nan,
            "mae": np.nan,
            "mse": np.nan,
            "r2": np.nan,
            "corr": np.nan,
        }

    avg_pred = float(np.mean(y_pred))
    avg_real = float(np.mean(y_real))
    bias = avg_pred - avg_real

    # MAE / MSE are fine even with n==1
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)

    # R² needs at least 2 samples
    if n >= 2:
        try:
            r2 = r2_score(y_real, y_pred)
        except Exception:
            r2 = np.nan
    else:
        r2 = np.nan

    # corr (information coefficient) also needs variation
    if n > 1 and np.std(y_pred) > 0 and np.std(y_real) > 0:
        corr = float(np.corrcoef(y_pred, y_real)[0, 1])
    else:
        corr = np.nan

    return {
        "avg_pred": avg_pred,
        "avg_real": avg_real,
        "bias": bias,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "corr": corr,
    }


def print_calibration_block(tag: str, store: Dict[str, Dict[str, List[float]]]):
    """
    Pretty print the calibration stats for one horizon.
    store structure:
        {
          "pred": { "Equities": [...], "Gold": [...], ... },
          "real": { "Equities": [...], "Gold": [...], ... }
        }
    """
    print("")
    print(f"MODEL CALIBRATION ({tag})")
    print("===================================")
    for asset_label in ["Equities", "Gold", "REITs", "Bitcoin"]:
        m = compute_metrics(store["pred"][asset_label], store["real"][asset_label])
        print(
            f"{asset_label}: "
            f"avg_pred={m['avg_pred']:+.2%} "
            f"avg_real={m['avg_real']:+.2%} "
            f"bias={m['bias']:+.2%} "
            f"MAE={m['mae']*100:.2f}% "
            f"MSE={m['mse']:.4f} "
            f"R2={m['r2']:.2f} "
            f"corr={m['corr']:.2f}"
        )


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
    and print the blocks
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

    # log for debugging (kept from previous version)
    perf_log: List[dict] = []

    # --- NEW: central calibration store for all horizons
    calib_store = {
        "monthly":   {"pred": defaultdict(list), "real": defaultdict(list)},
        "quarterly": {"pred": defaultdict(list), "real": defaultdict(list)},
        "half":      {"pred": defaultdict(list), "real": defaultdict(list)},
        "yearly":    {"pred": defaultdict(list), "real": defaultdict(list)},
    }

    if PRINT_MODE != "quiet":
        print(f"\nMONTHLY FORECASTS ({year}): {len(monthly_periods)} periods")
        print("=" * 60)

    w_prev: Optional[Dict[str, float]] = None
    monthly_turnovers_l1: List[float] = []

    # smoothing memory for ML forecasts per horizon
    prev_exp_assets_smooth_by_horizon: Dict[str, Optional[pd.Series]] = {
        "monthly": None,
        "quarterly": None,
        "half": None,
        "yearly": None,
    }

    # -----------------
    # monthly loop (21d)
    # -----------------
    for idx, (decision_date, period_end) in enumerate(monthly_periods, start=1):
        (
            w,
            exp_h,
            kpi_pack,
            realized,
            exp_assets_smooth_next,
            exp_assets_series,      # per-asset forecast ER
            realized_vec            # per-asset realized return
        ) = await forecast_one_period(
            returns,
            decision_date,
            period_end,
            "monthly",
            w_prev,
            prev_exp_assets_smooth_by_horizon["monthly"],
            period_index=idx,
            period_count=len(monthly_periods),
            macro_env=macro_env,
            perf_log=perf_log,
        )

        results['monthly_realized'].append(realized)

        # collect per-asset forecast vs realized for calibration metrics
        for asset in ["Equities", "Gold", "REITs", "Bitcoin"]:
            calib_store["monthly"]["pred"][asset].append(
                float(exp_assets_series.get(asset, 0.0))
            )
            calib_store["monthly"]["real"][asset].append(
                float(realized_vec.get(asset, 0.0))
            )

        # turnover tracking
        if w_prev is not None:
            keys_list = ["Equities", "Gold", "REITs", "Bitcoin"]
            l1 = float(sum(abs(w[k] - w_prev[k]) for k in keys_list))
            monthly_turnovers_l1.append(l1)

        w_prev = w
        prev_exp_assets_smooth_by_horizon["monthly"] = exp_assets_smooth_next

    results['monthly_turnovers'] = monthly_turnovers_l1

    # --------------------
    # quarterly loop (63d)
    # --------------------
    if PRINT_MODE != "quiet":
        print(f"\nQUARTERLY FORECASTS ({year}): {len(quarterly_periods)} periods")
        print("=" * 60)

    for idx, (dd, pe) in enumerate(quarterly_periods, start=1):
        (
            w_q,
            exp_q,
            kpi_q,
            realized_q,
            exp_assets_smooth_next_q,
            exp_assets_series_q,
            realized_vec_q,
        ) = await forecast_one_period(
            returns,
            dd,
            pe,
            "quarterly",
            w_prev,
            prev_exp_assets_smooth_by_horizon["quarterly"],
            idx,
            len(quarterly_periods),
            macro_env=macro_env,
            perf_log=perf_log,
        )

        results['quarterly_realized'].append(realized_q)

        # stash prediction vs realized for 63d horizon
        for asset in ["Equities", "Gold", "REITs", "Bitcoin"]:
            calib_store["quarterly"]["pred"][asset].append(
                float(exp_assets_series_q.get(asset, 0.0))
            )
            calib_store["quarterly"]["real"][asset].append(
                float(realized_vec_q.get(asset, 0.0))
            )

        w_prev = w_q  # carry forward most recent allocation
        prev_exp_assets_smooth_by_horizon["quarterly"] = exp_assets_smooth_next_q

    # --------------------
    # half-year loop (126d)
    # --------------------
    if PRINT_MODE != "quiet":
        print(f"\nHALF-YEAR FORECASTS ({year}): {len(half_periods)} periods")
        print("=" * 60)

    for idx, (dd, pe) in enumerate(half_periods, start=1):
        (
            w_h,
            exp_h_h,
            kpi_pack_h,
            realized_h,
            exp_assets_smooth_next_h,
            exp_assets_series_h,
            realized_vec_h,
        ) = await forecast_one_period(
            returns,
            dd,
            pe,
            "half",
            w_prev,
            prev_exp_assets_smooth_by_horizon["half"],
            idx,
            len(half_periods),
            macro_env=macro_env,
            perf_log=perf_log,
        )

        results['half_realized'].append(realized_h)

        # stash prediction vs realized for 126d horizon
        for asset in ["Equities", "Gold", "REITs", "Bitcoin"]:
            calib_store["half"]["pred"][asset].append(
                float(exp_assets_series_h.get(asset, 0.0))
            )
            calib_store["half"]["real"][asset].append(
                float(realized_vec_h.get(asset, 0.0))
            )

        w_prev = w_h
        prev_exp_assets_smooth_by_horizon["half"] = exp_assets_smooth_next_h

    # -----------------
    # yearly loop (252d)
    # -----------------
    if PRINT_MODE != "quiet":
        print(f"\nYEARLY FORECAST ({year}): 1 period")
        print("=" * 60)

    for idx, (dd, pe) in enumerate(yearly_periods, start=1):
        (
            w_y,
            exp_y,
            kpi_pack_y,
            realized_y,
            exp_assets_smooth_next_y,
            exp_assets_series_y,
            realized_vec_y,
        ) = await forecast_one_period(
            returns,
            dd,
            pe,
            "yearly",
            w_prev,
            prev_exp_assets_smooth_by_horizon["yearly"],
            idx,
            1,
            macro_env=macro_env,
            perf_log=perf_log,
        )

        results['yearly_realized'].append(realized_y)

        # stash prediction vs realized for 252d horizon
        for asset in ["Equities", "Gold", "REITs", "Bitcoin"]:
            calib_store["yearly"]["pred"][asset].append(
                float(exp_assets_series_y.get(asset, 0.0))
            )
            calib_store["yearly"]["real"][asset].append(
                float(realized_vec_y.get(asset, 0.0))
            )

        w_prev = w_y
        prev_exp_assets_smooth_by_horizon["yearly"] = exp_assets_smooth_next_y

    # =========================
    # summary stats for realized performance
    # =========================
    _summarize(f"MONTHLY POLICY: {year} REALIZED PERFORMANCE", results['monthly_realized'])
    if monthly_turnovers_l1:
        print(f"Avg monthly turnover: {100 * float(np.mean(monthly_turnovers_l1)):.2f}%")

    _summarize(f"QUARTERLY POLICY: {year} REALIZED PERFORMANCE", results['quarterly_realized'])
    _summarize(f"HALF-YEAR POLICY: {year} REALIZED PERFORMANCE", results['half_realized'])
    _summarize(f"YEARLY POLICY: {year} REALIZED PERFORMANCE", results['yearly_realized'])

    # =========================
    # calibration metrics blocks (per horizon)
    # =========================
    print_calibration_block("monthly horizon (21d)",    calib_store["monthly"])
    print_calibration_block("quarterly horizon (63d)",  calib_store["quarterly"])
    print_calibration_block("half-year horizon (126d)", calib_store["half"])
    print_calibration_block("yearly horizon (252d)",    calib_store["yearly"])

    # regime distribution
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
        # typical path: run one backtest year
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
