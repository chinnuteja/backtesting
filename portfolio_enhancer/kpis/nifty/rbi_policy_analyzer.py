# -*- coding: utf-8 -*-
"""
NIFTY KPI — RBI/Rate Environment (Historical/Backtest-safe)
Weight hint in composite: 0.10

Uses global rate proxies (^TNX, ^IRX) + Bank Nifty (^NSEBANK) to infer policy backdrop.
- ^TNX is 10× yield → we normalize to % (divide by 10).
- ^IRX is a short-end proxy (13-week bills ~policy sensitive).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict

WEIGHT_HINT = 0.10
LOOKBACK_DAYS = 180
PROXIES = {
    "us_10y": "^TNX",       # normalize/10 to percent
    "short_end": "^IRX",    # short-end proxy (% already)
    "bank_nifty": "^NSEBANK",
}

POLICY_W = {"rate_environment": 0.40, "yield_curve_shape": 0.25, "banking_sector": 0.35}
IMPACT = {"high_bps": 0.25, "mod_bps": 0.15}  # thresholds in % points

def _log(msg: str, verbose: bool):
    if verbose:
        print(msg)

def _yf_hist_inc(ticker: str, start: pd.Timestamp, end_incl: pd.Timestamp) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    end_exc = end_incl + pd.Timedelta(days=1)
    df = t.history(start=start.strftime("%Y-%m-%d"), end=end_exc.strftime("%Y-%m-%d"), interval="1d")
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    return df

async def _fetch_rate_pack(asof: str, verbose: bool) -> Dict[str, pd.DataFrame]:
    end = pd.Timestamp(asof).tz_localize(None)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    out: Dict[str, pd.DataFrame] = {}
    for k, tic in PROXIES.items():
        try:
            df = _yf_hist_inc(tic, start, end)
            if not df.empty:
                out[k] = df
        except Exception:
            continue
    return out

def _rate_env(rate_pack: Dict[str, pd.DataFrame]) -> Dict:
    if "us_10y" not in rate_pack or "Close" not in rate_pack["us_10y"]:
        return {"rate_sentiment": 0.0, "trend": "unknown", "reason": "missing_us10y"}

    # Normalize ^TNX to percent
    ten = rate_pack["us_10y"]["Close"] / 10.0
    cur = float(ten.iloc[-1])
    ch1m = float(cur - (ten.iloc[-21] if len(ten) >= 21 else cur))
    ch3m = float(cur - (ten.iloc[-63] if len(ten) >= 63 else cur))

    if ch1m > IMPACT["high_bps"]:
        s, tr = -0.6, "rising_rates_negative"
    elif ch1m > IMPACT["mod_bps"]:
        s, tr = -0.3, "moderate_rate_rise"
    elif ch1m < -IMPACT["high_bps"]:
        s, tr = 0.6, "falling_rates_positive"
    elif ch1m < -IMPACT["mod_bps"]:
        s, tr = 0.3, "moderate_rate_fall"
    else:
        s, tr = 0.0, "stable_rates"

    lvl = "high" if cur > 4.0 else "moderate" if cur > 2.0 else "low"
    return {"current_rate_level_pct": cur, "rate_change_1m_pct": ch1m, "rate_change_3m_pct": ch3m,
            "rate_sentiment": s, "trend": tr, "rate_level_assessment": lvl}

def _yield_curve(rate_pack: Dict[str, pd.DataFrame]) -> Dict:
    if "short_end" not in rate_pack or "us_10y" not in rate_pack:
        return {"curve_sentiment": 0.0, "shape": "unknown"}
    se = rate_pack["short_end"]["Close"]  # already percent scale
    ten = rate_pack["us_10y"]["Close"] / 10.0
    se_cur = float(se.iloc[-1]); ten_cur = float(ten.iloc[-1])
    spread = ten_cur - se_cur  # 10Y minus short end
    if spread > 2.0: sc, shape = 0.3, "steep_positive"
    elif spread > 0.5: sc, shape = 0.1, "normal"
    elif spread > -0.5: sc, shape = -0.2, "flat_cautious"
    else: sc, shape = -0.5, "inverted_negative"
    return {"curve_spread_pct": spread, "curve_sentiment": sc, "shape": shape,
            "short_end_pct": se_cur, "us_10y_pct": ten_cur}

def _banking_block(rate_pack: Dict[str, pd.DataFrame]) -> Dict:
    if "bank_nifty" not in rate_pack or "Close" not in rate_pack["bank_nifty"]:
        return {"banking_sentiment": 0.0, "performance": "unknown"}
    bn = rate_pack["bank_nifty"]["Close"]
    perf = float((bn.iloc[-1] - (bn.iloc[-21] if len(bn) >= 21 else bn.iloc[-1])) / (bn.iloc[-21] if len(bn) >= 21 else bn.iloc[-1]))
    ret = bn.pct_change().dropna()
    vol = float(ret.std() * np.sqrt(252)) if len(ret) > 10 else 0.2
    if perf > 0.05: s, tag = 0.4, "banking_outperforming"
    elif perf > 0.02: s, tag = 0.2, "banking_moderate_positive"
    elif perf < -0.05: s, tag = -0.4, "banking_underperforming"
    elif perf < -0.02: s, tag = -0.2, "banking_moderate_negative"
    else: s, tag = 0.0, "banking_neutral"
    adj = float(max(0.5, min(1.0, 1 - (vol - 0.2) * 2)))  # dampen if very volatile
    return {"recent_performance_pct": perf * 100.0, "banking_sentiment": s * adj,
            "performance": tag, "volatility": vol, "volatility_adjustment": adj}

def _compose(blocks: Dict[str, Dict]) -> Dict[str, float]:
    rate_s = blocks["rate"]["rate_sentiment"]
    curve_s = blocks["curve"]["curve_sentiment"]
    bank_s = blocks["bank"]["banking_sentiment"]
    comp = (rate_s * POLICY_W["rate_environment"] +
            curve_s * POLICY_W["yield_curve_shape"] +
            bank_s * POLICY_W["banking_sector"])
    # confidence from data completeness (policy is noisy)
    avail = sum(b is not None for b in blocks.values())
    completeness = avail / 3.0
    conf = 0.6 * completeness
    return {"composite": float(np.clip(comp, -1, 1)), "confidence": float(conf), "completeness": float(completeness)}

def _summary(blocks: Dict[str, Dict]) -> str:
    rate_tr = blocks["rate"].get("trend", "unknown")
    curve = blocks["curve"].get("shape", "unknown")
    bank = blocks["bank"].get("performance", "unknown")
    if rate_tr == "rising_rates_negative":
        return "🔴 Rising rates negative for equities"
    if rate_tr == "falling_rates_positive":
        return "🟢 Falling rates positive for equities"
    if curve == "inverted_negative":
        return "🔴 Inverted/negative curve signal"
    if bank == "banking_outperforming":
        return "🟢 Banking strength supportive"
    if bank == "banking_underperforming":
        return "🔴 Banking weakness headwind"
    return "🟡 Neutral policy backdrop"

async def nifty_rbi_policy_kpi(asof: str, horizon: str = "M", *, verbose: bool = False) -> Dict:
    _log(f"🏦 RBI/Rate Environment (asof {asof})", verbose)
    try:
        pack = await _fetch_rate_pack(asof, verbose)
        blocks = {
            "rate": _rate_env(pack),
            "curve": _yield_curve(pack),
            "bank": _banking_block(pack),
        }
        comp = _compose(blocks)
        out = {
            "kpi": "NIFTY_RBI_POLICY",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": comp["composite"],
            "confidence": comp["confidence"],
            "components": {
                "rate_environment": float(blocks["rate"].get("rate_sentiment", 0.0)),
                "yield_curve_shape": float(blocks["curve"].get("curve_sentiment", 0.0)),
                "banking_sector_response": float(blocks["bank"].get("banking_sentiment", 0.0)),
            },
            "weight_hint": WEIGHT_HINT,
            "notes": {
                "data_completeness": comp["completeness"],
                "summary": _summary(blocks),
                "us10y_pct": blocks["rate"].get("current_rate_level_pct", 0.0),
            },
            "policy_analysis": blocks,
        }
        _log(f"✅ RBI/Rate composite={out['composite_sentiment']:+.3f} (conf {out['confidence']:.0%})", verbose)
        return out
    except Exception as e:
        _log(f"[WARN] RBI/Rate failed: {e}", verbose)
        return {
            "kpi": "NIFTY_RBI_POLICY",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": 0.0,
            "confidence": 0.3,
            "components": {},
            "weight_hint": WEIGHT_HINT * 0.5,
            "error": f"{type(e).__name__}: {e}",
        }
