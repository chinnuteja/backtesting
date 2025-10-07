# -*- coding: utf-8 -*-
"""
NIFTY KPI — Market Sentiment via (India)VIX (Historical/Backtest-safe)
Weight hint in composite: 0.20

Tries ^INDIAVIX first (India VIX), then ^VIX; falls back to a proxy built from ^GSPC rolling vol.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List

LOOKBACK_DAYS = 60
WEIGHT_HINT = 0.20

# Try India VIX then US VIX
VIX_TICKERS: List[str] = ["^INDIAVIX", "^VIX"]
PROXY_MARKET = "^GSPC"  # for volatility proxy if VIX data missing

THRESH = {"extreme_fear": 35, "fear": 25, "neutral": 20, "greed": 15, "extreme_greed": 10}
W = {"vix_level": 0.50, "vix_trend": 0.30, "vix_volatility": 0.20}

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

def _fetch_vix(asof: str, verbose: bool) -> pd.DataFrame:
    end = pd.Timestamp(asof).tz_localize(None)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    for tic in VIX_TICKERS:
        try:
            _log(f"  🔎 trying {tic}", verbose)
            df = _yf_hist_inc(tic, start, end)
            if not df.empty and len(df) >= 20 and "Close" in df:
                return df
        except Exception:
            continue
    # proxy
    mkt = _yf_hist_inc(PROXY_MARKET, start, end)
    if mkt.empty or "Close" not in mkt:
        # minimal fallback
        idx = pd.DatetimeIndex([end])
        return pd.DataFrame({"Close": [20.0], "Open": [20.0], "High": [21.0], "Low": [19.0]}, index=idx)
    ret = mkt["Close"].pct_change().dropna()
    vol = (ret.rolling(20).std() * np.sqrt(252) * 100.0).dropna()
    df = pd.DataFrame(index=vol.index)
    df["Close"] = vol
    df["Open"] = vol.shift(1).fillna(vol)
    df["High"] = vol * 1.1
    df["Low"] = vol * 0.9
    return df.dropna()

def _level_sentiment(vix: float) -> Dict:
    if vix >= THRESH["extreme_fear"]:
        return {"level_sentiment": 0.6, "regime": "extreme_fear_contrarian"}
    elif vix >= THRESH["fear"]:
        return {"level_sentiment": 0.3, "regime": "fear_cautious_positive"}
    elif vix >= THRESH["neutral"]:
        return {"level_sentiment": 0.0, "regime": "neutral_balanced"}
    elif vix >= THRESH["greed"]:
        return {"level_sentiment": -0.2, "regime": "greed_cautious_negative"}
    else:
        return {"level_sentiment": -0.5, "regime": "extreme_complacency_warning"}

def _trend_sentiment(series: pd.Series) -> Dict:
    if len(series) < 6:
        return {"trend_sentiment": 0.0, "short_term_trend_pct": 0.0, "medium_term_trend_pct": 0.0, "trend_interpretation": "insufficient"}
    short = (series.iloc[-1] - series.iloc[-6]) / series.iloc[-6]
    medium = (series.iloc[-1] - series.iloc[-21]) / series.iloc[-21] if len(series) >= 21 else 0.0
    if short > 0.10:
        if series.iloc[-1] > 30:
            ts, interp = 0.3, "rising_fear_contrarian_opportunity"
        else:
            ts, interp = -0.4, "rising_fear_negative"
    elif short < -0.10:
        ts, interp = 0.5, "falling_fear_positive"
    else:
        ts, interp = 0.0, "stable_sentiment"
    return {
        "trend_sentiment": float(ts),
        "short_term_trend_pct": float(short * 100.0),
        "medium_term_trend_pct": float(medium * 100.0),
        "trend_interpretation": interp,
    }

def _vol_regime(series: pd.Series) -> Dict:
    ret = series.pct_change().dropna()
    if len(ret) < 10:
        return {"volatility_sentiment": 0.0, "vix_volatility": 0.0, "regime": "insufficient"}
    sigma = float(ret.std())
    if sigma > 0.15: vs, reg = -0.3, "unstable_sentiment"
    elif sigma < 0.05: vs, reg = 0.2, "stable_sentiment"
    else: vs, reg = 0.0, "normal"
    return {"volatility_sentiment": float(vs), "vix_volatility": sigma, "regime": reg}

def _compose(level: float, trend: float, vol: float, vix_now: float) -> Dict[str, float]:
    comp = level * W["vix_level"] + trend * W["vix_trend"] + vol * W["vix_volatility"]
    if vix_now > 30 or vix_now < 15:
        conf = 0.8
    elif vix_now > 25 or vix_now < 18:
        conf = 0.7
    else:
        conf = 0.6
    return {"composite": float(np.clip(comp, -1, 1)), "confidence": float(conf)}

async def nifty_market_sentiment_kpi(asof: str, horizon: str = "M", *, verbose: bool = False) -> Dict:
    _log(f"😨 Market Sentiment (asof {asof})", verbose)
    try:
        vix_df = _fetch_vix(asof, verbose)
        closes = vix_df["Close"].dropna()
        vix_now = float(closes.iloc[-1])

        level_info = _level_sentiment(vix_now)
        trend_info = _trend_sentiment(closes)
        vol_info = _vol_regime(closes)

        comp = _compose(level_info["level_sentiment"], trend_info["trend_sentiment"], vol_info["volatility_sentiment"], vix_now)

        out = {
            "kpi": "NIFTY_MARKET_SENTIMENT_VIX",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": comp["composite"],
            "confidence": comp["confidence"],
            "components": {
                "vix_level_sentiment": float(level_info["level_sentiment"]),
                "vix_trend_sentiment": float(trend_info["trend_sentiment"]),
                "volatility_regime_sentiment": float(vol_info["volatility_sentiment"]),
            },
            "weight_hint": WEIGHT_HINT,
            "notes": {
                "current_vix": vix_now,
                "regime": level_info["regime"],
                "trend_interpretation": trend_info["trend_interpretation"],
                "source": "INDIAVIX/VIX_or_vol_proxy",
            },
        }
        _log(f"✅ Market Sentiment composite={out['composite_sentiment']:+.3f} (conf {out['confidence']:.0%})", verbose)
        return out
    except Exception as e:
        _log(f"[WARN] Market sentiment failed: {e}", verbose)
        return {
            "kpi": "NIFTY_MARKET_SENTIMENT_VIX",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": 0.0,
            "confidence": 0.5,
            "components": {},
            "weight_hint": WEIGHT_HINT * 0.5,
            "error": f"{type(e).__name__}: {e}",
        }


# ... keep the class code above unchanged ...

# ---- Uniform wrapper for composite/backtester ----
async def nifty_market_sentiment_kpi(asof: str, horizon: str = "M", *, verbose: bool = False) -> Dict:
    analyzer = MarketSentimentAnalyzer()
    res = await analyzer.analyze_market_sentiment(asof)
    return {
        "kpi": "NIFTY_MARKET_SENTIMENT_VIX",
        "asof": asof,
        "horizon": horizon,
        "composite_sentiment": float(res.get("component_sentiment", 0.0)),
        "confidence": float(res.get("component_confidence", 0.6)),
        "components": res.get("component_scores", {}),
        "weight_hint": 0.20,
        "raw": res,
    }
