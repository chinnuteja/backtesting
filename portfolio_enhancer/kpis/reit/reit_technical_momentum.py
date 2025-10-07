# HISTORICAL VERSION for backtesting - REIT Technical Momentum Analysis

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import asyncio
from typing import Dict
import warnings

warnings.filterwarnings("ignore")


class REITTechnicalMomentumAnalyzer:
    """HISTORICAL VERSION - REIT Technical Momentum Analysis - 35% weight in REIT composite"""

    def __init__(self, config: Dict | None = None):
        self.config = config or self._default_config()
        print("REIT Technical Momentum Analyzer (HISTORICAL) initialized")

    def _default_config(self) -> Dict:
        return {
            "lookback_days": 90,
            "reit_tickers": ["VNQ", "SCHH", "IYR"],
            "momentum_weights": {"price_momentum": 0.40, "rsi_analysis": 0.30, "volume_trend": 0.30},
        }

    async def fetch_reit_data(self, backtest_date: str | None = None) -> pd.DataFrame:
        print(f"🏢 Fetching REIT data for {backtest_date or 'current'}")
        end_ts = pd.Timestamp(backtest_date) if backtest_date else None
        start_ts = (end_ts - pd.Timedelta(days=self.config["lookback_days"])) if end_ts is not None else None

        for ticker in self.config["reit_tickers"]:
            try:
                hist = yf.Ticker(ticker).history(
                    start=None if start_ts is None else start_ts.strftime("%Y-%m-%d"),
                    end=None if end_ts is None else end_ts.strftime("%Y-%m-%d"),
                    period=None if end_ts is not None else f"{self.config['lookback_days']}d",
                    interval="1d",
                    auto_adjust=True,
                )
                if isinstance(hist.index, pd.DatetimeIndex):
                    hist.index = hist.index.tz_localize(None)
                if not hist.empty and len(hist) >= 30:
                    print(f"✅ Fetched {len(hist)} days from {ticker}")
                    return hist
            except Exception as e:
                print(f"❌ {ticker}: {e}")
                continue

        raise ValueError("Unable to fetch REIT data")

    def calculate_reit_momentum(self, reit_data: pd.DataFrame) -> Dict:
        closes = reit_data["Close"]
        cp = float(closes.iloc[-1])
        m1 = (cp - float(closes.iloc[-22])) / float(closes.iloc[-22]) if len(closes) >= 22 else 0.0
        m3 = (cp - float(closes.iloc[-66])) / float(closes.iloc[-66]) if len(closes) >= 66 else 0.0
        comp = m1 * 0.6 + m3 * 0.4
        ms = float(np.tanh(comp * 8.0))
        return {
            "momentum_1m_pct": float(m1 * 100),
            "momentum_3m_pct": float(m3 * 100),
            "momentum_sentiment": ms,
            "trend_direction": "bullish" if comp > 0.02 else "bearish" if comp < -0.02 else "neutral",
        }

    def calculate_rsi_analysis(self, reit_data: pd.DataFrame) -> Dict:
        closes = reit_data["Close"]
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        crsi = float(rsi.iloc[-1]) if not rsi.empty else 50.0

        if crsi >= 70:
            rsis = -0.3
        elif crsi >= 60:
            rsis = 0.4
        elif crsi <= 30:
            rsis = 0.4
        elif crsi <= 40:
            rsis = -0.3
        else:
            rsis = 0.0

        return {"current_rsi": crsi, "rsi_sentiment": float(rsis), "rsi_level": "overbought" if crsi > 70 else "oversold" if crsi < 30 else "normal"}

    def calculate_volume_trend(self, reit_data: pd.DataFrame) -> Dict:
        if "Volume" not in reit_data.columns or float(reit_data["Volume"].sum()) <= 0:
            return {"volume_sentiment": 0.0, "volume_trend": "unavailable"}
        volume = reit_data["Volume"]
        v20 = volume.rolling(20).mean()
        cur = float(volume.iloc[-1])
        avg = float(v20.iloc[-1]) if not v20.empty else cur
        vr = cur / avg if avg > 1e-12 else 1.0
        if vr > 1.5:
            vs = 0.3
        elif vr < 0.7:
            vs = -0.2
        else:
            vs = 0.0
        return {"volume_ratio": float(vr), "volume_sentiment": float(vs), "volume_trend": "high" if vr > 1.3 else "normal" if vr > 0.8 else "low"}

    def calculate_reit_technical_composite(self, technical_analyses: Dict) -> Dict:
        w = self.config["momentum_weights"]
        m = float(technical_analyses["momentum"]["momentum_sentiment"])
        r = float(technical_analyses["rsi"]["rsi_sentiment"])
        v = float(technical_analyses["volume"]["volume_sentiment"])
        score = m * w["price_momentum"] + r * w["rsi_analysis"] + v * w["volume_trend"]
        return {
            "composite_score": float(np.clip(score, -1.0, 1.0)),
            "confidence": 0.75,
            "component_scores": {"price_momentum": float(m), "rsi_analysis": float(r), "volume_trend": float(v)},
        }

    async def analyze_reit_technical_momentum(self, backtest_date: str | None = None) -> Dict:
        print(f"🏢 Analyzing REIT Technical Momentum for {backtest_date or 'current'}...")
        start = datetime.now()
        try:
            data = await self.fetch_reit_data(backtest_date)
            analyses = {
                "momentum": self.calculate_reit_momentum(data),
                "rsi": self.calculate_rsi_analysis(data),
                "volume": self.calculate_volume_trend(data),
            }
            agg = self.calculate_reit_technical_composite(analyses)
            rt = (datetime.now() - start).total_seconds()
            res = {
                "component_sentiment": float(agg["composite_score"]),
                "component_confidence": float(agg["confidence"]),
                "component_weight": 0.35,
                "weighted_contribution": float(agg["composite_score"]) * 0.35,
                "technical_analysis": analyses,
                "market_context": {
                    "current_price": float(data["Close"].iloc[-1]),
                    "data_points": int(len(data)),
                    "backtest_date": backtest_date or "real_time",
                },
                "component_scores": agg["component_scores"],
                "execution_time_seconds": rt,
                "kpi_name": "REIT_Technical_Momentum",
                "analysis_timestamp": datetime.now().isoformat(),
            }
            print(f"✅ REIT Technical Analysis completed: {res['component_sentiment']:.3f}")
            return res
        except Exception as e:
            print(f"❌ REIT technical analysis failed: {e}")
            return {
                "component_sentiment": 0.0,
                "component_confidence": 0.5,
                "component_weight": 0.35 * 0.5,
                "error": str(e),
                "kpi_name": "REIT_Technical_Momentum",
                "analysis_timestamp": datetime.now().isoformat(),
            }
