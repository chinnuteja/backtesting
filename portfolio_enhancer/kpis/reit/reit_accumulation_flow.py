# HISTORICAL VERSION for backtesting - REIT Accumulation Flow Analysis

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import asyncio
from typing import Dict
import warnings

warnings.filterwarnings("ignore")


class REITAccumulationFlowAnalyzer:
    """HISTORICAL VERSION - REIT Accumulation Flow Analysis - 30% weight in REIT composite"""

    def __init__(self, config: Dict | None = None):
        self.config = config or self._default_config()
        print("REIT Accumulation Flow Analyzer (HISTORICAL) initialized")

    def _default_config(self) -> Dict:
        return {
            "lookback_days": 90,
            "reit_tickers": ["VNQ", "SCHH", "IYR"],
            "flow_weights": {
                "accumulation_pattern": 0.50,
                "institutional_flow": 0.30,
                "volume_analysis": 0.20,
            },
        }

    async def fetch_reit_data(self, backtest_date: str | None = None) -> pd.DataFrame:
        """Fetch REIT data with historical constraint (rolling cutoff)."""
        print(f"🏢 Fetching REIT accumulation data for {backtest_date or 'current'}")

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

    def calculate_accumulation_pattern(self, reit_data: pd.DataFrame) -> Dict:
        closes = reit_data["Close"]
        highs = reit_data["High"]
        lows = reit_data["Low"]
        volumes = reit_data["Volume"] if "Volume" in reit_data.columns else pd.Series([1] * len(closes), index=closes.index)

        mfm = ((closes - lows) - (highs - closes)) / (highs - lows)
        mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mfv = mfm * volumes
        ad_line = mfv.cumsum()

        current_ad = ad_line.iloc[-1]
        past_ad = ad_line.iloc[-30] if len(ad_line) >= 30 else ad_line.iloc[0]
        ad_trend = (current_ad - past_ad) / (abs(past_ad) if abs(past_ad) > 1e-12 else 1.0)

        if ad_trend > 0.10:
            acc_s = 0.5
        elif ad_trend > 0.05:
            acc_s = 0.2
        elif ad_trend < -0.10:
            acc_s = -0.4
        elif ad_trend < -0.05:
            acc_s = -0.2
        else:
            acc_s = 0.0

        return {
            "ad_trend": float(ad_trend),
            "accumulation_sentiment": float(acc_s),
            "pattern": "accumulation" if ad_trend > 0.05 else "distribution" if ad_trend < -0.05 else "neutral",
        }

    def calculate_institutional_flow_proxy(self, reit_data: pd.DataFrame) -> Dict:
        if "Volume" not in reit_data.columns or float(reit_data["Volume"].sum()) <= 0:
            return {"institutional_sentiment": 0.0, "flow_trend": "no_volume_data"}

        closes = reit_data["Close"]
        volumes = reit_data["Volume"]

        pm = (closes.iloc[-1] - closes.iloc[-21]) / closes.iloc[-21] if len(closes) >= 21 else 0.0

        vol_ma_20 = volumes.rolling(20).mean()
        current_vol = float(volumes.iloc[-1])
        avg_vol = float(vol_ma_20.iloc[-1]) if not vol_ma_20.empty else current_vol
        vr = current_vol / avg_vol if avg_vol > 1e-12 else 1.0

        if vr > 1.3 and pm > 0.03:
            inst_s = 0.4
        elif vr > 1.3 and pm < -0.03:
            inst_s = -0.4
        elif vr > 1.1:
            inst_s = 0.1 if pm > 0 else -0.1
        else:
            inst_s = 0.0

        return {
            "price_momentum_pct": float(pm * 100),
            "volume_ratio": float(vr),
            "institutional_sentiment": float(inst_s),
            "flow_trend": "buying" if inst_s > 0.1 else "selling" if inst_s < -0.1 else "neutral",
        }

    def calculate_volume_analysis(self, reit_data: pd.DataFrame) -> Dict:
        if "Volume" not in reit_data.columns or float(reit_data["Volume"].sum()) <= 0:
            return {"volume_sentiment": 0.0, "volume_pattern": "unavailable"}

        volumes = reit_data["Volume"]
        closes = reit_data["Close"]

        vol_ma_10 = volumes.rolling(10).mean()
        vol_ma_30 = volumes.rolling(30).mean()
        short = float(vol_ma_10.iloc[-1]) if not vol_ma_10.empty else float(volumes.iloc[-1])
        long = float(vol_ma_30.iloc[-1]) if not vol_ma_30.empty else float(volumes.iloc[-1])

        vt = (short - long) / long if long > 1e-12 else 0.0
        pt = (float(closes.iloc[-1]) - float(closes.iloc[-10])) / float(closes.iloc[-10]) if len(closes) >= 10 else 0.0

        if vt > 0.2 and pt > 0:
            vs = 0.3
        elif vt > 0.2 and pt < 0:
            vs = -0.3
        elif vt < -0.2:
            vs = -0.1
        else:
            vs = 0.0

        return {
            "volume_trend": float(vt),
            "volume_sentiment": float(vs),
            "volume_pattern": "expanding" if vt > 0.15 else "contracting" if vt < -0.15 else "stable",
        }

    def calculate_accumulation_flow_composite(self, flow_analyses: Dict) -> Dict:
        w = self.config["flow_weights"]
        a = float(flow_analyses["accumulation"]["accumulation_sentiment"])
        i = float(flow_analyses["institutional"]["institutional_sentiment"])
        v = float(flow_analyses["volume"]["volume_sentiment"])
        score = a * w["accumulation_pattern"] + i * w["institutional_flow"] + v * w["volume_analysis"]
        return {
            "composite_score": float(np.clip(score, -1.0, 1.0)),
            "confidence": 0.70,
            "component_scores": {
                "accumulation_pattern": float(a),
                "institutional_flow": float(i),
                "volume_analysis": float(v),
            },
        }

    async def analyze_reit_accumulation_flow(self, backtest_date: str | None = None) -> Dict:
        print(f"🏢 Analyzing REIT Accumulation Flow for {backtest_date or 'current'}...")
        start = datetime.now()
        try:
            data = await self.fetch_reit_data(backtest_date)
            analyses = {
                "accumulation": self.calculate_accumulation_pattern(data),
                "institutional": self.calculate_institutional_flow_proxy(data),
                "volume": self.calculate_volume_analysis(data),
            }
            agg = self.calculate_accumulation_flow_composite(analyses)
            rt = (datetime.now() - start).total_seconds()
            res = {
                "component_sentiment": float(agg["composite_score"]),
                "component_confidence": float(agg["confidence"]),
                "component_weight": 0.30,
                "weighted_contribution": float(agg["composite_score"]) * 0.30,
                "accumulation_flow_analysis": analyses,
                "market_context": {
                    "current_price": float(data["Close"].iloc[-1]),
                    "data_points": int(len(data)),
                    "backtest_date": backtest_date or "real_time",
                },
                "component_scores": agg["component_scores"],
                "execution_time_seconds": rt,
                "kpi_name": "REIT_Accumulation_Flow",
                "analysis_timestamp": datetime.now().isoformat(),
            }
            print(f"✅ REIT Accumulation Flow Analysis completed: {res['component_sentiment']:.3f}")
            return res
        except Exception as e:
            print(f"❌ REIT accumulation flow analysis failed: {e}")
            return {
                "component_sentiment": 0.0,
                "component_confidence": 0.5,
                "component_weight": 0.30 * 0.5,
                "error": str(e),
                "kpi_name": "REIT_Accumulation_Flow",
                "analysis_timestamp": datetime.now().isoformat(),
            }
