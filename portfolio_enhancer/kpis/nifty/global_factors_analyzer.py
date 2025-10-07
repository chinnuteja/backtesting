# -*- coding: utf-8 -*-
"""
NIFTY KPI — Global Factors (Historical/Backtest-safe)
Weight hint in composite: 0.05

Uses US equities, DXY, crude oil, and VIX to infer global pull on NIFTY.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict

LOOKBACK_DAYS = 60
WEIGHT_HINT = 0.05
WEIGHTS = {
    "us_market_correlation": 0.35,
    "dollar_strength": 0.25,
    "oil_price_impact": 0.20,
    "global_risk_sentiment": 0.20,
}

TICKERS = {
    "nifty": "^NSEI",
    "us_market": "^GSPC",
    "dollar_index": "DX-Y.NYB",
    "crude_oil": "CL=F",
    "vix": "^VIX",
}

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

def _ret(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change().dropna() if "Close" in df else pd.Series(dtype=float)

def _analyze_us_corr(nifty: pd.DataFrame, spx: pd.DataFrame) -> Dict:
    r_n = _ret(nifty); r_s = _ret(spx)
    idx = r_n.index.intersection(r_s.index)
    if len(idx) < 10:
        return {"correlation": 0.0, "recent_momentum_alignment": 0.0, "strength": "insufficient"}
    corr = float(r_n.loc[idx].corr(r_s.loc[idx]))
    recent = np.sign(r_n.tail(5).mean()) == np.sign(r_s.tail(5).mean())
    align = 1.0 if recent else -0.5
    strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
    return {"correlation": corr, "recent_momentum_alignment": float(align), "strength": strength, "common_periods": len(idx)}

def _analyze_dxy(dxy: pd.DataFrame) -> Dict:
    if dxy.empty or "Close" not in dxy or len(dxy) < 10:
        return {"impact_score": 0.0, "trend": "unknown", "dxy_change_pct": 0.0, "volatility": 0.2}
    change = (dxy["Close"].iloc[-1] - dxy["Close"].iloc[-10]) / dxy["Close"].iloc[-10]
    if change > 0.02: score, trend = -0.6, "strong_dollar_negative"
    elif change < -0.02: score, trend = 0.6, "weak_dollar_positive"
    else: score, trend = float(np.clip(-10 * change, -1, 1)), "neutral"
    vol = float(_ret(dxy).std() * np.sqrt(252)) if len(dxy) > 5 else 0.2
    return {"impact_score": score, "dxy_change_pct": float(change * 100), "trend": trend, "volatility": vol}

def _analyze_oil(oil: pd.DataFrame) -> Dict:
    if oil.empty or "Close" not in oil or len(oil) < 21:
        return {"impact_score": 0.0, "trend": "unknown", "oil_change_pct": 0.0, "current_level": float(oil["Close"].iloc[-1]) if "Close" in oil else 0.0}
    change = (oil["Close"].iloc[-1] - oil["Close"].iloc[-20]) / oil["Close"].iloc[-20]
    if change > 0.10: score, trend = -0.5, "high_oil_negative"
    elif change < -0.10: score, trend = 0.5, "low_oil_positive"
    else: score, trend = float(np.clip(-2 * change, -1, 1)), "neutral"
    return {"impact_score": score, "oil_change_pct": float(change * 100), "trend": trend, "current_level": float(oil["Close"].iloc[-1])}

def _analyze_vix(vix: pd.DataFrame) -> Dict:
    if vix.empty or "Close" not in vix:
        return {"risk_score": 0.0, "regime": "unknown", "current_vix": 0.0, "vix_trend": 0.0}
    cur = float(vix["Close"].iloc[-1])
    if cur > 30: score, regime = -0.7, "high_fear"
    elif cur > 20: score, regime = -0.3, "elevated_uncertainty"
    elif cur < 15: score, regime = 0.4, "complacency"
    else: score, regime = 0.1, "normal"
    trend = (cur - float(vix["Close"].iloc[-5])) / float(vix["Close"].iloc[-5]) if len(vix) >= 6 else 0.0
    return {"risk_score": score, "current_vix": cur, "vix_trend": float(trend), "regime": regime}

def _compose(factors: Dict[str, Dict]) -> Dict[str, float]:
    us = factors["us"]
    dxy = factors["dxy"]
    oil = factors["oil"]
    risk = factors["risk"]
    comp = (
        us.get("recent_momentum_alignment", 0.0) * WEIGHTS["us_market_correlation"]
        + dxy.get("impact_score", 0.0) * WEIGHTS["dollar_strength"]
        + oil.get("impact_score", 0.0) * WEIGHTS["oil_price_impact"]
        + risk.get("risk_score", 0.0) * WEIGHTS["global_risk_sentiment"]
    )
    available = sum([
        1 if us else 0,
        1 if dxy else 0,
        1 if oil else 0,
        1 if risk else 0,
    ])
    completeness = available / 4.0
    confidence = 0.7 * completeness
    return {"composite": float(np.clip(comp, -1, 1)), "confidence": float(confidence), "completeness": float(completeness)}

async def nifty_global_factors_kpi(asof: str, horizon: str = "M", *, verbose: bool = False) -> Dict:
    _log(f"🌍 Global Factors (asof {asof})", verbose)
    try:
        end = pd.Timestamp(asof).tz_localize(None)
        start = end - pd.Timedelta(days=LOOKBACK_DAYS)

        data = {
            k: _yf_hist_inc(tic, start, end) for k, tic in TICKERS.items()
        }

        factors = {
            "us": _analyze_us_corr(data["nifty"], data["us_market"]),
            "dxy": _analyze_dxy(data["dollar_index"]),
            "oil": _analyze_oil(data["crude_oil"]),
            "risk": _analyze_vix(data["vix"]),
        }
        comp = _compose(factors)

        out = {
            "kpi": "NIFTY_GLOBAL_FACTORS",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": comp["composite"],
            "confidence": comp["confidence"],
            "components": {
                "us_market_alignment": float(factors["us"].get("recent_momentum_alignment", 0.0)),
                "dollar_strength_impact": float(factors["dxy"].get("impact_score", 0.0)),
                "oil_price_impact": float(factors["oil"].get("impact_score", 0.0)),
                "global_risk_sentiment": float(factors["risk"].get("risk_score", 0.0)),
            },
            "weight_hint": WEIGHT_HINT,
            "notes": {
                "data_completeness": comp["completeness"],
                "us_corr_strength": factors["us"].get("strength", "unknown"),
                "current_vix": factors["risk"].get("current_vix", 0.0),
            },
        }
        _log(f"✅ Global composite={out['composite_sentiment']:+.3f} (conf {out['confidence']:.0%})", verbose)
        return out
    except Exception as e:
        _log(f"[WARN] Global factors failed: {e}", verbose)
        return {
            "kpi": "NIFTY_GLOBAL_FACTORS",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": 0.0,
            "confidence": 0.3,
            "components": {},
            "weight_hint": WEIGHT_HINT * 0.5,
            "error": f"{type(e).__name__}: {e}",
        }


# ... keep the class code above unchanged ...

# ---- Uniform wrapper for composite/backtester ----
async def nifty_global_factors_kpi(asof: str, horizon: str = "M", *, verbose: bool = False) -> Dict:
    analyzer = GlobalFactorsAnalyzer()
    res = await analyzer.analyze_global_factors_sentiment(asof)
    return {
        "kpi": "NIFTY_GLOBAL_FACTORS",
        "asof": asof,
        "horizon": horizon,
        "composite_sentiment": float(res.get("component_sentiment", 0.0)),
        "confidence": float(res.get("component_confidence", 0.3)),
        "components": res.get("component_scores", {}),
        "weight_hint": 0.05,
        "raw": res,
    }
