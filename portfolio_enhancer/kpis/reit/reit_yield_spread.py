# HISTORICAL VERSION for backtesting - REIT Yield Spread Analysis

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import asyncio
from typing import Dict
import warnings

warnings.filterwarnings("ignore")


class REITYieldSpreadAnalyzer:
    """HISTORICAL VERSION - REIT Yield Spread Analysis - 35% weight in REIT composite"""

    def __init__(self, config: Dict | None = None):
        self.config = config or self._default_config()
        print("REIT Yield Spread Analyzer (HISTORICAL) initialized")

    def _default_config(self) -> Dict:
        return {
            "lookback_days": 120,
            "reit_tickers": ["VNQ", "SCHH"],
            "treasury_tickers": ["^TNX", "^IRX"],  # 10Y and 13W (used as short proxy)
            "spread_weights": {"yield_spread": 0.60, "spread_trend": 0.40},
        }

    async def fetch_yield_data(self, backtest_date: str | None = None) -> Dict[str, pd.DataFrame]:
        print(f"📈 Fetching yield data for {backtest_date or 'current'}")
        out: Dict[str, pd.DataFrame] = {}
        tickers = self.config["reit_tickers"] + self.config["treasury_tickers"]

        end_ts = pd.Timestamp(backtest_date) if backtest_date else None
        start_ts = (end_ts - pd.Timedelta(days=self.config["lookback_days"])) if end_ts is not None else None

        for t in tickers:
            try:
                hist = yf.Ticker(t).history(
                    start=None if start_ts is None else start_ts.strftime("%Y-%m-%d"),
                    end=None if end_ts is None else end_ts.strftime("%Y-%m-%d"),
                    period=None if end_ts is not None else f"{self.config['lookback_days']}d",
                    interval="1d",
                    auto_adjust=True,
                )
                if isinstance(hist.index, pd.DatetimeIndex):
                    hist.index = hist.index.tz_localize(None)
                if not hist.empty:
                    out[t] = hist
                    print(f"✅ Fetched {t}: {len(hist)} days")
            except Exception as e:
                print(f"❌ {t}: {e}")

        return out

    def estimate_reit_yield(self, reit_data: pd.DataFrame) -> float:
        rets = reit_data["Close"].pct_change().dropna()
        ann = float(rets.mean() * 252.0)
        est_div_yield = 0.035
        return float(est_div_yield + ann)

    def calculate_yield_spread_analysis(self, yield_data: Dict[str, pd.DataFrame]) -> Dict:
        reit_tkr = next((t for t in self.config["reit_tickers"] if t in yield_data), None)
        if not reit_tkr:
            return {"spread_sentiment": 0.0, "spread_regime": "no_reit_data"}

        tsy_tkr = next((t for t in self.config["treasury_tickers"] if t in yield_data), None)
        if not tsy_tkr:
            return {"spread_sentiment": 0.0, "spread_regime": "no_treasury_data"}

        reit_df = yield_data[reit_tkr]
        tsy_df = yield_data[tsy_tkr]

        reit_yld = self.estimate_reit_yield(reit_df)
        cur_tsy_yld = float(tsy_df["Close"].iloc[-1]) / 100.0  # percent to ratio

        spread = reit_yld - cur_tsy_yld
        tsy_avg = float(tsy_df["Close"].rolling(60).mean().iloc[-1]) / 100.0 if len(tsy_df) >= 60 else cur_tsy_yld
        hist_spread = reit_yld - tsy_avg  # not printed, but could be logged

        if spread > 0.04:
            s = 0.6
        elif spread > 0.02:
            s = 0.3
        elif spread < -0.01:
            s = -0.4
        else:
            s = 0.0

        return {
            "estimated_reit_yield_pct": float(reit_yld * 100),
            "treasury_yield_pct": float(cur_tsy_yld * 100),
            "yield_spread_pct": float(spread * 100),
            "spread_sentiment": float(s),
            "spread_regime": "attractive" if spread > 0.03 else "moderate" if spread > 0.01 else "unattractive",
        }

    def calculate_spread_trend(self, yield_data: Dict[str, pd.DataFrame]) -> Dict:
        tsy_tkr = next((t for t in self.config["treasury_tickers"] if t in yield_data), None)
        if not tsy_tkr:
            return {"trend_sentiment": 0.0, "trend_direction": "unknown"}

        s = yield_data[tsy_tkr]["Close"]
        cur = float(s.iloc[-1])
        past = float(s.iloc[-30]) if len(s) >= 30 else cur
        chg = (cur - past) / (past if abs(past) > 1e-12 else 1.0)

        if chg < -0.10:
            ts = 0.5
            d = "falling_yields_bullish"
        elif chg < -0.05:
            ts = 0.2
            d = "moderately_falling_yields"
        elif chg > 0.10:
            ts = -0.4
            d = "rising_yields_bearish"
        elif chg > 0.05:
            ts = -0.2
            d = "moderately_rising_yields"
        else:
            ts = 0.0
            d = "stable_yields"

        return {"yield_change_pct": float(chg * 100), "trend_sentiment": float(ts), "trend_direction": d}

    def calculate_yield_spread_composite(self, spread_analyses: Dict) -> Dict:
        w = self.config["spread_weights"]
        s = float(spread_analyses["yield_spread"]["spread_sentiment"])
        t = float(spread_analyses["spread_trend"]["trend_sentiment"])
        score = s * w["yield_spread"] + t * w["spread_trend"]
        return {
            "composite_score": float(np.clip(score, -1.0, 1.0)),
            "confidence": 0.70,
            "component_scores": {"yield_spread": float(s), "spread_trend": float(t)},
        }

    async def analyze_reit_yield_spread(self, backtest_date: str | None = None) -> Dict:
        print(f"📈 Analyzing REIT Yield Spread for {backtest_date or 'current'}...")
        start = datetime.now()
        try:
            yd = await self.fetch_yield_data(backtest_date)
            analyses = {
                "yield_spread": self.calculate_yield_spread_analysis(yd),
                "spread_trend": self.calculate_spread_trend(yd),
            }
            agg = self.calculate_yield_spread_composite(analyses)
            rt = (datetime.now() - start).total_seconds()
            res = {
                "component_sentiment": float(agg["composite_score"]),
                "component_confidence": float(agg["confidence"]),
                "component_weight": 0.35,
                "weighted_contribution": float(agg["composite_score"]) * 0.35,
                "yield_spread_analysis": analyses,
                "market_context": {
                    "data_sources": int(len(yd)),
                    "analysis_method": "estimated_yield_spread",
                    "backtest_date": backtest_date or "real_time",
                },
                "component_scores": agg["component_scores"],
                "execution_time_seconds": rt,
                "kpi_name": "REIT_Yield_Spread",
                "analysis_timestamp": datetime.now().isoformat(),
            }
            print(f"✅ REIT Yield Spread Analysis completed: {res['component_sentiment']:.3f}")
            return res
        except Exception as e:
            print(f"❌ REIT yield spread analysis failed: {e}")
            return {
                "component_sentiment": 0.0,
                "component_confidence": 0.4,
                "component_weight": 0.35 * 0.5,
                "error": str(e),
                "kpi_name": "REIT_Yield_Spread",
                "analysis_timestamp": datetime.now().isoformat(),
            }
