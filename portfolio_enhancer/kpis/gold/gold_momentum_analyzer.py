# portfolio_enhancer/gold_momentum_analyzer.py
# HISTORICAL VERSION — Technical Momentum KPI for Gold
# Aligned with BTC-style interface: accepts historical_cutoff and returns {"score": ...}

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class GoldMomentumAnalyzer:
    """
    Technical momentum KPI for Gold.
    Suggested component weight: 25% in Gold composite.
    """

    def __init__(self, historical_cutoff: Optional[str] = None, config: Dict = None):
        self.config = config or self._default_config()
        self.historical_cutoff = pd.to_datetime(historical_cutoff).strftime("%Y-%m-%d") if historical_cutoff else None
        self.cache: Dict = {}
        print("Gold Momentum Analyzer (HISTORICAL) initialized")
        if self.historical_cutoff:
            print(f"🔒 Historical cutoff set to {self.historical_cutoff}")

    def _default_config(self) -> Dict:
        return {
            "lookback_days": 120,
            "gold_tickers": ["GC=F", "GLD", "^XAU"],  # prefer futures & GLD
            "momentum_weights": {
                "price_momentum": 0.35,
                "rsi_momentum": 0.25,
                "macd_signal": 0.20,
                "volume_confirmation": 0.20,
            },
            "technical_params": {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "momentum_periods": [10, 20, 50],
            },
            "component_weight": 0.25,  # in Gold composite
        }

    async def _fetch_gold(self, cutoff_str: Optional[str]) -> pd.DataFrame:
        print(f"🥇 Fetching gold price data for {cutoff_str or 'current'}")
        end_date = pd.to_datetime(cutoff_str) if cutoff_str else pd.Timestamp.today().normalize()
        start_date = end_date - pd.Timedelta(days=int(self.config["lookback_days"]))
        if cutoff_str:
            print(f"  📅 Historical range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")

        for tk in self.config["gold_tickers"]:
            try:
                print(f"  📊 Trying {tk}...")
                t = yf.Ticker(tk)
                hist = t.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                )
                if not hist.empty and len(hist) >= 50:
                    print(f"    ✅ {tk}: {len(hist)} days")
                    return hist
                print(f"    ⚠️ Insufficient data from {tk}: {len(hist)} days")
            except Exception as e:
                print(f"    ❌ {tk}: {e}")
                continue
        raise ValueError("Unable to fetch sufficient gold data")

    # ---------- components ----------
    def _price_momentum(self, df: pd.DataFrame) -> Dict:
        c = df["Close"].dropna()
        cur = float(c.iloc[-1])

        scores = {}
        for p in self.config["technical_params"]["momentum_periods"]:
            if len(c) >= p + 1 and c.iloc[-(p + 1)] != 0:
                scores[f"{p}d_momentum"] = float((cur - float(c.iloc[-(p + 1)])) / float(c.iloc[-(p + 1)]))
            else:
                scores[f"{p}d_momentum"] = 0.0

        m10, m20, m50 = scores.get("10d_momentum", 0.0), scores.get("20d_momentum", 0.0), scores.get("50d_momentum", 0.0)
        comp = m10 * 0.25 + m20 * 0.50 + m50 * 0.25
        sent = float(np.tanh(comp * 20.0))
        strength = "strong" if abs(comp) > 0.05 else ("moderate" if abs(comp) > 0.02 else "weak")
        direction = "bullish" if comp > 0.01 else ("bearish" if comp < -0.01 else "neutral")
        return {"individual_momentums": scores, "composite_momentum_pct": comp * 100.0,
                "momentum_sentiment": sent, "trend_strength": strength, "trend_direction": direction}

    def _rsi(self, df: pd.DataFrame) -> Dict:
        period = self.config["technical_params"]["rsi_period"]
        c = df["Close"].dropna()
        d = c.diff()
        gain = d.where(d > 0, 0.0).rolling(period).mean()
        loss = (-d.where(d < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        cur = float(rsi.iloc[-1]) if not rsi.empty else 50.0

        if cur >= 70:
            s = -0.3
        elif cur >= 60:
            s = 0.5
        elif cur >= 40:
            s = 0.0
        elif cur >= 30:
            s = -0.5
        else:
            s = 0.3

        trend = "rising" if len(rsi) >= 5 and rsi.iloc[-1] > rsi.iloc[-5] else "falling"
        lvl = "overbought" if cur > 70 else ("oversold" if cur < 30 else "normal")
        return {"current_rsi": cur, "rsi_sentiment": float(s), "rsi_level": lvl, "rsi_trend": trend}

    def _macd(self, df: pd.DataFrame) -> Dict:
        p = self.config["technical_params"]
        c = df["Close"].dropna()
        ema_fast = c.ewm(span=p["macd_fast"]).mean()
        ema_slow = c.ewm(span=p["macd_slow"]).mean()
        macd_line = ema_fast - ema_slow
        signal = macd_line.ewm(span=p["macd_signal"]).mean()
        hist = macd_line - signal

        m, s, h = float(macd_line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
        if m > s and h > 0:
            ms = 0.6
        elif m > s:
            ms = 0.3
        elif m < s and h < 0:
            ms = -0.6
        elif m < s:
            ms = -0.3
        else:
            ms = 0.0

        cross = "none"
        if len(macd_line) >= 2 and len(signal) >= 2:
            if macd_line.iloc[-1] > signal.iloc[-1] and macd_line.iloc[-2] <= signal.iloc[-2]:
                cross = "bullish_crossover"
            elif macd_line.iloc[-1] < signal.iloc[-1] and macd_line.iloc[-2] >= signal.iloc[-2]:
                cross = "bearish_crossover"

        strength = "strong" if abs(m - s) > (abs(m) + 1e-9) * 0.1 else "moderate"
        return {"macd_line": m, "signal_line": s, "histogram": h,
                "macd_sentiment": float(ms), "crossover_status": cross, "signal_strength": strength}

    def _volume(self, df: pd.DataFrame) -> Dict:
        if "Volume" not in df.columns or df["Volume"].fillna(0).sum() == 0:
            return {"volume_sentiment": 0.0, "volume_trend": "unavailable", "price_volume_relationship": "unknown"}
        v = df["Volume"].fillna(0.0)
        c = df["Close"].dropna()
        ma20 = v.rolling(20).mean()
        vr = float(v.iloc[-1] / (ma20.iloc[-1] if ma20.iloc[-1] > 0 else v.iloc[-1]))
        pc = float((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2]) if len(c) >= 2 and c.iloc[-2] != 0 else 0.0
        vc = float((v.iloc[-1] - v.iloc[-2]) / v.iloc[-2]) if len(v) >= 2 and v.iloc[-2] != 0 else 0.0

        if vr > 1.5 and pc > 0:
            s = 0.5
        elif vr > 1.5 and pc < 0:
            s = -0.5
        elif vr > 1.2:
            s = 0.2 if pc > 0 else -0.2
        else:
            s = 0.0

        rel = "positive" if (pc > 0 and vc > 0) or (pc < 0 and vc < 0) else "negative"
        vt = "high" if vr > 1.3 else ("normal" if vr > 0.8 else "low")
        return {"current_volume": float(v.iloc[-1]), "volume_ratio": vr,
                "volume_sentiment": float(s), "volume_trend": vt, "price_volume_relationship": rel}

    def _composite(self, blocks: Dict) -> Dict:
        w = self.config["momentum_weights"]
        pm = float(blocks["price_momentum"]["momentum_sentiment"])
        rs = float(blocks["rsi_analysis"]["rsi_sentiment"])
        mc = float(blocks["macd_analysis"]["macd_sentiment"])
        vo = float(blocks["volume_analysis"]["volume_sentiment"])

        comp = float(np.clip(pm * w["price_momentum"] + rs * w["rsi_momentum"]
                             + mc * w["macd_signal"] + vo * w["volume_confirmation"], -1.0, 1.0))

        sigs = [pm, rs, mc, vo]
        pos = sum(1 for x in sigs if x > 0.1)
        neg = sum(1 for x in sigs if x < -0.1)
        if pos >= 3 or neg >= 3:
            conf = 0.85
        elif pos >= 2 or neg >= 2:
            conf = 0.75
        else:
            conf = 0.65
        return {"score": comp, "confidence": float(conf),
                "components": {"price_momentum": pm, "rsi_momentum": rs, "macd_signal": mc, "volume_confirmation": vo}}

    # -------- public entry --------
    async def analyze_gold_technical_momentum(self, cutoff_str: Optional[str] = None) -> Dict:
        cutoff = cutoff_str or self.historical_cutoff
        print(f"🥇 Analyzing Gold Momentum — {cutoff or 'current'}...")
        t0 = datetime.now()
        try:
            df = await self._fetch_gold(cutoff)
            blocks = {
                "price_momentum": self._price_momentum(df),
                "rsi_analysis": self._rsi(df),
                "macd_analysis": self._macd(df),
                "volume_analysis": self._volume(df),
            }
            comp = self._composite(blocks)
            took = (datetime.now() - t0).total_seconds()
            out = {
                "score": comp["score"],
                "confidence": comp["confidence"],
                "component_weight": float(self.config["component_weight"]),
                "weighted_contribution": comp["score"] * float(self.config["component_weight"]),
                "components": comp["components"],
                "market_context": {
                    "current_price": float(df["Close"].iloc[-1]),
                    "daily_change_pct": float(((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100.0)) if len(df) >= 2 else 0.0,
                    "data_points": len(df),
                    "lookback_days": self.config["lookback_days"],
                    "backtest_date": cutoff or "real_time",
                },
                "kpi_name": "Gold_Technical_Momentum",
                "execution_time_seconds": took,
                "analysis_timestamp": datetime.now().isoformat(),
            }
            print(f"✅ Gold Momentum completed: {out['score']:+.3f}  (conf {out['confidence']:.0%})")
            return out
        except Exception as e:
            print(f"❌ Gold momentum analysis failed: {e}")
            return {
                "score": 0.0,
                "confidence": 0.5,
                "component_weight": float(self.config["component_weight"]) * 0.5,
                "error": f"Gold momentum exception: {e}",
                "kpi_name": "Gold_Technical_Momentum",
                "analysis_timestamp": datetime.now().isoformat(),
            }


# quick test
if __name__ == "__main__":
    import asyncio
    async def _t():
        az = GoldMomentumAnalyzer(historical_cutoff="2023-10-31")
        res = await az.analyze_gold_technical_momentum("2023-10-31")
        print(res)
    asyncio.run(_t())
