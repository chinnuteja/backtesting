# -*- coding: utf-8 -*-
"""
NIFTY KPI — FII/DII Flows (Historical/Backtest-safe)
Weight hint in composite: 0.35

Key guarantees:
- Uses only data up to and including `asof` (inclusive).
- Quiet by default; optional `verbose=True`.
- Returns a consistent dict with `composite_sentiment`, `confidence`, `components`, `weight_hint`.
- Safe when Yahoo lacks Volume for ^NSEI (graceful volume proxy).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Optional

# -------- config --------
LOOKBACK_DAYS = 90
WEIGHTS = {"price_momentum": 0.40, "volume_analysis": 0.30, "volatility_regime": 0.30}
WEIGHT_HINT = 0.35

def _log(msg: str, verbose: bool):  # quiet by default
    if verbose:
        print(msg)

def _yf_history_inclusive(ticker: str, start: pd.Timestamp, end_inclusive: pd.Timestamp) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    # yfinance's `end` is exclusive → add one day
    end_exclusive = end_inclusive + pd.Timedelta(days=1)
    df = t.history(start=start.strftime("%Y-%m-%d"), end=end_exclusive.strftime("%Y-%m-%d"), interval="1d")
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    return df

def _safe_tail_mean(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) < max(5, n // 2):
        return 0.0
    return float(s.tail(n).mean())

def _compute_flows_proxy(nifty: pd.DataFrame, verbose: bool) -> Dict[str, float]:
    # Close returns
    if "Close" not in nifty:
        return {
            "price_momentum_score": 0.0,
            "volume_sentiment": 0.0,
            "volatility_sentiment": 0.0,
            "fii_flow_proxy": 0.0,
            "dii_flow_proxy": 0.0,
            "net_institutional_flow": 0.0,
            "current_volume_ratio": 1.0,
            "volatility_ratio": 1.0,
            "momentum_strength": 0.0,
            "data_quality": "insufficient_close",
        }

    ret = nifty["Close"].pct_change()

    # --- 1) Price momentum score (last 20d avg, sqrt scaled) ---
    m20 = _safe_tail_mean(ret, 20)
    # already a daily mean; rough "strength" proxy via sqrt(20)
    momentum_score_raw = m20 * np.sqrt(20)
    if momentum_score_raw > 0.02:
        price_momentum = min(1.0, momentum_score_raw * 25)
    elif momentum_score_raw < -0.02:
        price_momentum = max(-1.0, momentum_score_raw * 25)
    else:
        price_momentum = momentum_score_raw * 10

    # --- 2) Volume proxy (works even if Volume missing or all zeros) ---
    volume_sentiment = 0.0
    current_volume_ratio = 1.0
    last_return = float(ret.dropna().iloc[-1]) if ret.notna().any() else 0.0
    if "Volume" in nifty and nifty["Volume"].notna().sum() >= 20 and nifty["Volume"].sum() > 0:
        vol = nifty["Volume"].copy()
        vol_ma20 = vol.rolling(20).mean()
        if vol_ma20.notna().any() and vol_ma20.iloc[-1] and vol.iloc[-1]:
            current_volume_ratio = float(vol.iloc[-1] / vol_ma20.iloc[-1])
            if current_volume_ratio > 1.5 and last_return > 0:
                volume_sentiment = 0.6
            elif current_volume_ratio > 1.5 and last_return < 0:
                volume_sentiment = -0.6
            elif current_volume_ratio > 1.2:
                volume_sentiment = 0.3 if last_return > 0 else -0.3
            else:
                volume_sentiment = 0.0
    else:
        # fallback: treat volume as neutral; ratio ~1.0
        current_volume_ratio = 1.0
        volume_sentiment = 0.0

    # --- 3) Volatility regime (20d vol vs 60d mean) ---
    vol_ann = ret.rolling(20).std() * np.sqrt(252)
    cur_vol = float(vol_ann.dropna().iloc[-1]) if vol_ann.notna().any() else 0.2
    vol_ma60 = float(vol_ann.rolling(60).mean().dropna().iloc[-1]) if vol_ann.notna().sum() >= 60 else 0.2
    vol_ratio = cur_vol / vol_ma60 if vol_ma60 > 0 else 1.0
    if vol_ratio < 0.8:
        vol_sent = 0.3
    elif vol_ratio > 1.5:
        vol_sent = -0.3
    else:
        vol_sent = 0.0

    fii_proxy = (price_momentum + volume_sentiment) / 2.0
    dii_proxy = -0.6 * fii_proxy
    net_flow = fii_proxy + dii_proxy

    return {
        "price_momentum_score": float(price_momentum),
        "volume_sentiment": float(volume_sentiment),
        "volatility_sentiment": float(vol_sent),
        "fii_flow_proxy": float(fii_proxy),
        "dii_flow_proxy": float(dii_proxy),
        "net_institutional_flow": float(net_flow),
        "current_volume_ratio": float(current_volume_ratio),
        "volatility_ratio": float(vol_ratio),
        "momentum_strength": float(abs(momentum_score_raw)),
        "data_quality": "price_volume_proxy",
    }

def _composite_from_flows(est: Dict[str, float]) -> Dict[str, float]:
    pm = est["price_momentum_score"]
    va = est["volume_sentiment"]
    vr = est["volatility_sentiment"]
    comp = pm * WEIGHTS["price_momentum"] + va * WEIGHTS["volume_analysis"] + vr * WEIGHTS["volatility_regime"]

    sigs = [pm, va, vr]
    pos = sum(s > 0.10 for s in sigs)
    neg = sum(s < -0.10 for s in sigs)
    if pos >= 2 or neg >= 2:
        conf = 0.8
    else:
        conf = 0.6
    conf *= 0.85  # proxy method haircut
    return {"composite": float(np.clip(comp, -1, 1)), "confidence": float(conf)}

async def nifty_fii_dii_kpi(asof: str, horizon: str = "M", *, verbose: bool = False) -> Dict:
    """Main entry — returns a standard KPI dict."""
    _log(f"💰 FII/DII Flows (asof {asof})", verbose)
    try:
        end = pd.Timestamp(asof).tz_localize(None)
        start = end - pd.Timedelta(days=LOOKBACK_DAYS)
        data = _yf_history_inclusive("^NSEI", start, end)
        if data.empty:
            raise ValueError("No NIFTY data")

        est = _compute_flows_proxy(data, verbose)
        comp = _composite_from_flows(est)

        out = {
            "kpi": "NIFTY_FII_DII",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": comp["composite"],
            "confidence": comp["confidence"],
            "components": {
                "price_momentum": est["price_momentum_score"],
                "volume": est["volume_sentiment"],
                "volatility_regime": est["volatility_sentiment"],
            },
            "weight_hint": WEIGHT_HINT,
            "notes": {
                "fii_flow_proxy": est["fii_flow_proxy"],
                "dii_flow_proxy": est["dii_flow_proxy"],
                "net_flow_proxy": est["net_institutional_flow"],
                "volume_ratio": est["current_volume_ratio"],
                "volatility_ratio": est["volatility_ratio"],
                "method": est["data_quality"],
            },
        }
        _log(f"✅ FII/DII composite={out['composite_sentiment']:+.3f} (conf {out['confidence']:.0%})", verbose)
        return out
    except Exception as e:
        _log(f"[WARN] FII/DII failed: {e}", verbose)
        return {
            "kpi": "NIFTY_FII_DII",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": 0.0,
            "confidence": 0.4,
            "components": {},
            "weight_hint": WEIGHT_HINT * 0.5,
            "error": f"{type(e).__name__}: {e}",
        }


# ... keep the class code above unchanged ...

# ---- Uniform wrapper for composite/backtester ----
async def nifty_fii_dii_kpi(asof: str, horizon: str = "M", *, verbose: bool = False) -> Dict:
    analyzer = FIIDIIFlowsAnalyzer()
    res = await analyzer.analyze_fii_dii_sentiment(asof)
    # standard shape
    return {
        "kpi": "NIFTY_FII_DII",
        "asof": asof,
        "horizon": horizon,
        "composite_sentiment": float(res.get("component_sentiment", 0.0)),
        "confidence": float(res.get("component_confidence", 0.6)),
        "components": res.get("component_scores", {}),
        "weight_hint": 0.35,
        "raw": res,
    }
