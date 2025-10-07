# -*- coding: utf-8 -*-
"""
NIFTY KPI — Technicals (Historical / Backtest-safe)
Weight hint in composite: 0.30

- Uses only data up to `asof` (inclusive).
- Quiet by default (set verbose=True to print).
- Graceful when ^NSEI lacks Volume (neutral volume score).
- Returns a uniform structure compatible with other KPIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List

# -------------------------
# Config
# -------------------------
WEIGHT_HINT: float = 0.30
LOOKBACK_DAYS: int = 100
TECH_WEIGHTS: Dict[str, float] = {
    "trend_mas": 0.25,
    "momentum_rsi": 0.20,
    "trend_macd": 0.20,
    "volume_trend": 0.15,
    "bollinger": 0.10,
    "support_resistance": 0.10,
}
MA_PERIODS: List[int] = [20, 50, 200]
MACD = {"fast": 12, "slow": 26, "signal": 9}
# Try these in order; first one with adequate data wins
NIFTY_TICKERS: List[str] = ["^NSEI", "NIFTY50.NS", "^NSEBANK"]


# -------------------------
# Small helpers
# -------------------------
def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)

def _yf_hist_inclusive(ticker: str, start: pd.Timestamp, end_inclusive: pd.Timestamp) -> pd.DataFrame:
    """yfinance 'end' is exclusive -> add +1d; strip tz; return OHLCV frame."""
    t = yf.Ticker(ticker)
    end_exclusive = end_inclusive + pd.Timedelta(days=1)
    df = t.history(
        start=start.strftime("%Y-%m-%d"),
        end=end_exclusive.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    return df

async def _fetch_nifty(asof: str, verbose: bool) -> pd.DataFrame:
    """Fetch NIFTY price data (inclusive to asof) with fallbacks."""
    end = pd.Timestamp(asof).tz_localize(None)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    for tic in NIFTY_TICKERS:
        try:
            df = _yf_hist_inclusive(tic, start, end)
            if not df.empty and len(df) >= 50 and "Close" in df:
                _log(f"  ✅ Using {tic} ({len(df)} rows)", verbose)
                return df
            _log(f"  ⚠️ {tic} insufficient rows ({len(df)})", verbose)
        except Exception as e:
            _log(f"  ❌ {tic} failed: {e}", verbose)
            continue
    raise ValueError("Unable to fetch NIFTY data")

# -------------------------
# Indicator blocks
# -------------------------
def _calc_ma_block(df: pd.DataFrame) -> Dict:
    cur = float(df["Close"].iloc[-1])
    ma_info: Dict[str, Dict] = {}
    above = 0
    for p in MA_PERIODS:
        ma = df["Close"].rolling(p).mean()
        if ma.dropna().empty:
            v, rel, tr = cur, 0.0, "neutral"
        else:
            v = float(ma.iloc[-1])
            rel = float((cur - v) / v * 100.0) if v else 0.0
            tr = "above" if cur > v else "below"
        ma_info[f"ma_{p}"] = {"value": v, "price_vs_ma_pct": rel, "trend": tr}
        if tr == "above":
            above += 1
    trend_score = (above / len(MA_PERIODS) - 0.5) * 2.0  # [-1, +1]
    return {
        "individual_mas": ma_info,
        "trend_score": float(np.clip(trend_score, -1.0, 1.0)),
        "overall_trend": "bullish" if trend_score > 0.2 else "bearish" if trend_score < -0.2 else "neutral",
    }

def _calc_rsi_block(df: pd.DataFrame) -> Dict:
    period = 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    cur = float(rsi.dropna().iloc[-1]) if rsi.notna().any() else 50.0

    def lvl(x: float) -> str:
        if x >= 70: return "overbought"
        if x <= 30: return "oversold"
        return "neutral"

    def sig(x: float) -> str:
        if x >= 70: return "sell_signal"
        if x <= 30: return "buy_signal"
        return "neutral"

    return {"rsi_value": cur, "rsi_level": lvl(cur), "rsi_signal": sig(cur)}

def _calc_macd_block(df: pd.DataFrame) -> Dict:
    ema_fast = df["Close"].ewm(span=MACD["fast"], adjust=False).mean()
    ema_slow = df["Close"].ewm(span=MACD["slow"], adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=MACD["signal"], adjust=False).mean()
    if macd_line.dropna().empty or signal.dropna().empty:
        return {"macd_line": 0.0, "signal_line": 0.0, "crossover_status": "neutral"}
    m = float(macd_line.iloc[-1]); s = float(signal.iloc[-1])
    return {"macd_line": m, "signal_line": s, "crossover_status": "bullish" if m > s else "bearish"}

def _calc_volume_block(df: pd.DataFrame) -> Dict:
    # Some NSE proxies have 0/NaN volume → treat as neutral
    if "Volume" not in df or df["Volume"].sum() == 0 or df["Volume"].notna().sum() < 20:
        return {"current_volume": 0.0, "volume_ratio": 1.0, "volume_trend": "normal"}
    vol = df["Volume"]
    ma20 = vol.rolling(20).mean()
    cur = float(vol.iloc[-1])
    avg = float(ma20.iloc[-1]) if ma20.notna().any() else max(cur, 1.0)
    ratio = float(cur / avg) if avg else 1.0
    tr = "high" if ratio > 1.5 else ("normal" if ratio > 0.8 else "low")
    return {"current_volume": cur, "volume_ratio": ratio, "volume_trend": tr}

def _calc_bb_block(df: pd.DataFrame) -> Dict:
    period, std_k = 20, 2
    ma = df["Close"].rolling(period).mean()
    sd = df["Close"].rolling(period).std()
    if ma.dropna().empty or sd.dropna().empty:
        return {"bb_position": 0.5, "price_position": "within_bands"}
    upper = ma + std_k * sd
    lower = ma - std_k * sd
    cur = float(df["Close"].iloc[-1])
    rng = float(upper.iloc[-1] - lower.iloc[-1])
    pos = float((cur - float(lower.iloc[-1])) / rng) if rng else 0.5
    if cur > float(upper.iloc[-1]):
        flag = "above_upper"
    elif cur < float(lower.iloc[-1]):
        flag = "below_lower"
    else:
        flag = "within_bands"
    return {"bb_position": pos, "price_position": flag}

def _calc_sr_block(df: pd.DataFrame) -> Dict:
    hi = df["High"].tail(50).astype(float)
    lo = df["Low"].tail(50).astype(float)
    cur = float(df["Close"].iloc[-1])
    resistances: List[float] = []
    supports: List[float] = []

    for i in range(2, len(hi) - 2):
        if hi.iloc[i] > hi.iloc[i - 1] and hi.iloc[i] > hi.iloc[i + 1]:
            resistances.append(float(hi.iloc[i]))
    for i in range(2, len(lo) - 2):
        if lo.iloc[i] < lo.iloc[i - 1] and lo.iloc[i] < lo.iloc[i + 1]:
            supports.append(float(lo.iloc[i]))

    nearest_res = next((r for r in sorted(resistances) if r > cur), None)
    # choose the closest support below current
    supports_below = [s for s in supports if s < cur]
    nearest_sup = max(supports_below) if supports_below else None

    return {"nearest_resistance": nearest_res, "nearest_support": nearest_sup}

# -------------------------
# Scoring
# -------------------------
def _score_from_indicators(ind: Dict) -> Dict:
    scores: Dict[str, float] = {}

    # MA trend
    scores["trend_mas"] = float(np.clip(ind["moving_averages"]["trend_score"], -1.0, 1.0))

    # RSI → contrarian edges at extremes, moderate otherwise
    rsi = float(ind["rsi_analysis"]["rsi_value"])
    if rsi >= 70:
        rsi_s = -0.5
    elif rsi >= 60:
        rsi_s = 0.5
    elif rsi <= 30:
        rsi_s = 0.5
    elif rsi <= 40:
        rsi_s = -0.5
    else:
        rsi_s = 0.0
    scores["momentum_rsi"] = rsi_s

    # MACD crossover
    scores["trend_macd"] = 0.4 if ind["macd_analysis"]["crossover_status"] == "bullish" else -0.4

    # Volume regime
    vt = ind["volume_analysis"]["volume_trend"]
    scores["volume_trend"] = 0.3 if vt == "high" else (-0.3 if vt == "low" else 0.0)

    # Bollinger band location (center = neutral)
    bbp = float(ind["bollinger_bands"]["bb_position"])
    scores["bollinger"] = float((bbp - 0.5) * 0.4)

    # Support/Resistance (kept neutral; can be extended later)
    scores["support_resistance"] = 0.0

    composite = sum(scores[k] * TECH_WEIGHTS[k] for k in scores)
    # signal consistency → confidence
    vals = list(scores.values())
    pos = sum(v > 0.1 for v in vals)
    neg = sum(v < -0.1 for v in vals)
    tot = len(vals)
    if pos >= 0.7 * tot or neg >= 0.7 * tot:
        conf = 0.9
    elif pos >= 0.5 * tot or neg >= 0.5 * tot:
        conf = 0.7
    else:
        conf = 0.5

    return {
        "composite": float(np.clip(composite, -1.0, 1.0)),
        "confidence": float(conf),
        "scores": {k: float(v) for k, v in scores.items()},
    }

# -------------------------
# Public async KPI
# -------------------------
async def nifty_technical_kpi(asof: str, horizon: str = "M", *, verbose: bool = False) -> Dict:
    """
    Parameters
    ----------
    asof : str  (YYYY-MM-DD)  — last date of data to include.
    horizon : str  — "M", "Q", "H", "Y" (informational only).
    verbose : bool — prints debug logs if True.

    Returns
    -------
    dict with keys:
        kpi, asof, horizon, composite_sentiment, confidence,
        components (sub-scores), weight_hint, notes, sub_indicators
    """
    _log(f"📈 NIFTY Technicals (asof {asof})", verbose)
    try:
        df = await _fetch_nifty(asof, verbose)
        if df.empty:
            raise ValueError("empty NIFTY dataframe")

        indicators = {
            "moving_averages": _calc_ma_block(df),
            "rsi_analysis": _calc_rsi_block(df),
            "macd_analysis": _calc_macd_block(df),
            "volume_analysis": _calc_volume_block(df),
            "bollinger_bands": _calc_bb_block(df),
            "support_resistance": _calc_sr_block(df),
        }
        comp = _score_from_indicators(indicators)

        cur = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else cur

        out = {
            "kpi": "NIFTY_TECHNICALS",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": comp["composite"],
            "confidence": comp["confidence"],
            "components": comp["scores"],
            "weight_hint": WEIGHT_HINT,
            "notes": {
                "current_price": cur,
                "daily_change_pct": float(((cur - prev) / prev) * 100.0) if prev else 0.0,
                "source_ticker": " / ".join(NIFTY_TICKERS),
            },
            "sub_indicators": indicators,  # detailed block for optional export
        }
        _log(f"✅ Technicals composite={out['composite_sentiment']:+.3f} (conf {out['confidence']:.0%})", verbose)
        return out

    except Exception as e:
        _log(f"[WARN] Technicals failed: {e}", verbose)
        return {
            "kpi": "NIFTY_TECHNICALS",
            "asof": asof,
            "horizon": horizon,
            "composite_sentiment": 0.0,
            "confidence": 0.4,
            "components": {},
            "weight_hint": WEIGHT_HINT * 0.5,
            "error": f"{type(e).__name__}: {e}",
        }

__all__ = ["nifty_technical_kpi", "WEIGHT_HINT"]
