# portfolio_enhancer/gold_cot_positioning_analyzer.py
# HISTORICAL VERSION for backtesting — Gold Professional Positioning (COT proxies)
# Interface aligned with BTC analyzers: accepts historical_cutoff and returns {"score": ...}

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class ProfessionalPositioningAnalyzer:
    """
    HISTORICAL — COT-style professional positioning for Gold via proxies.
    Component weight suggestion: 15% inside the Gold KPI composite.
    """

    def __init__(self, historical_cutoff: Optional[str] = None, config: Dict = None):
        self.config = config or self._default_config()
        self.historical_cutoff = pd.to_datetime(historical_cutoff).strftime("%Y-%m-%d") if historical_cutoff else None
        self.cache: Dict = {}
        print("Professional Positioning (COT) Analyzer (HISTORICAL) initialized")
        if self.historical_cutoff:
            print(f"🔒 Historical cutoff set to {self.historical_cutoff}")

    def _default_config(self) -> Dict:
        return {
            "lookback_days": 180,  # ~6 months
            "positioning_proxies": {
                "gold_futures": "GC=F",     # Gold futures
                "gold_etf": "GLD",          # SPDR Gold ETF
                "miners_etf": "GDX",        # Gold miners ETF
                "dollar_index": "DX-Y.NYB"  # DXY proxy (yfinance symbol often OK)
            },
            "positioning_weights": {
                "volume_momentum": 0.40,
                "price_momentum": 0.30,
                "dollar_correlation": 0.30
            },
            "component_weight": 0.15,       # 15% inside Gold composite (suggested)
        }

    async def _fetch_positioning_proxy_data(self, cutoff_str: Optional[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch proxy datasets up to (and including) cutoff_str using yfinance daily bars.
        """
        if cutoff_str is None:
            cutoff_str = self.historical_cutoff
        print(f"📊 Fetching positioning proxy data for {cutoff_str or 'current'}")

        proxy_data: Dict[str, pd.DataFrame] = {}
        # Date range setup
        end_date = pd.to_datetime(cutoff_str) if cutoff_str else pd.Timestamp.today().normalize()
        start_date = end_date - pd.Timedelta(days=int(self.config["lookback_days"]))
        if cutoff_str:
            print(f"  📅 Historical range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")

        for proxy_name, ticker in self.config["positioning_proxies"].items():
            try:
                print(f"  📊 Fetching {proxy_name} ({ticker})...")
                t = yf.Ticker(ticker)
                if cutoff_str:
                    hist = t.history(
                        start=start_date.strftime("%Y-%m-%d"),
                        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        interval="1d",
                        auto_adjust=True,
                    )
                else:
                    hist = t.history(period=f"{self.config['lookback_days']}d", interval="1d", auto_adjust=True)

                if not hist.empty:
                    # Standardize columns we use
                    cols = [c for c in hist.columns if c in ("Close", "Volume")]
                    proxy_data[proxy_name] = hist[cols].copy()
                    print(f"    ✅ {proxy_name}: {len(hist)} days")
                else:
                    print(f"    ⚠️ {proxy_name}: No data")
            except Exception as e:
                print(f"    ❌ {proxy_name}: {e}")
                continue

        print(f"✅ Fetched {len(proxy_data)} positioning proxy datasets")
        return proxy_data

    # ---------- component analyses ----------
    def _analyze_volume_momentum_positioning(self, proxy_data: Dict[str, pd.DataFrame]) -> Dict:
        if "gold_futures" not in proxy_data:
            return {"volume_sentiment": 0.0, "trend": "no_data"}

        gold_data = proxy_data["gold_futures"]
        if "Volume" not in gold_data.columns or (gold_data["Volume"].fillna(0).sum() == 0):
            return {"volume_sentiment": 0.0, "trend": "no_volume_data"}

        volume = gold_data["Volume"].fillna(0.0)
        prices = gold_data["Close"].fillna(method="ffill").fillna(method="bfill")

        vol_ma_20 = volume.rolling(20).mean()
        vol_ma_50 = volume.rolling(50).mean()

        current_vol = float(volume.iloc[-1])
        avg_vol_20 = float(vol_ma_20.iloc[-1]) if not vol_ma_20.empty else current_vol
        avg_vol_50 = float(vol_ma_50.iloc[-1]) if not vol_ma_50.empty else current_vol

        volume_ratio_20 = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
        volume_trend = (avg_vol_20 - avg_vol_50) / avg_vol_50 if avg_vol_50 > 0 else 0.0

        price_change_20d = 0.0
        if len(prices) >= 21 and prices.iloc[-21] != 0:
            price_change_20d = float((prices.iloc[-1] - prices.iloc[-21]) / prices.iloc[-21])

        if volume_ratio_20 > 1.3 and price_change_20d > 0.02:
            volume_sentiment = 0.6
        elif volume_ratio_20 > 1.3 and price_change_20d < -0.02:
            volume_sentiment = -0.6
        elif volume_trend > 0.2 and price_change_20d > 0:
            volume_sentiment = 0.3
        elif volume_trend < -0.2:
            volume_sentiment = -0.2
        else:
            volume_sentiment = 0.0

        return {
            "current_volume_ratio": float(volume_ratio_20),
            "volume_trend": float(volume_trend),
            "volume_sentiment": float(volume_sentiment),
            "price_volume_alignment": "positive" if (price_change_20d > 0 and volume_trend > 0) else "negative",
        }

    def _analyze_price_momentum_positioning(self, proxy_data: Dict[str, pd.DataFrame]) -> Dict:
        momentum = []

        def _mom(df: pd.Series) -> float:
            if len(df) >= 21 and df.iloc[-21] != 0:
                return float((df.iloc[-1] - df.iloc[-21]) / df.iloc[-21])
            return 0.0

        if "gold_futures" in proxy_data:
            momentum.append(_mom(proxy_data["gold_futures"]["Close"].dropna()))

        if "gold_etf" in proxy_data:
            momentum.append(_mom(proxy_data["gold_etf"]["Close"].dropna()))

        if "miners_etf" in proxy_data:
            m = _mom(proxy_data["miners_etf"]["Close"].dropna()) * 0.5  # soften miners
            momentum.append(m)

        if not momentum:
            return {"momentum_sentiment": 0.0, "trend": "no_data"}

        avg_momentum = float(np.mean(momentum))
        momentum_sentiment = float(np.tanh(avg_momentum * 15.0))

        trend_strength = "strong" if abs(avg_momentum) > 0.05 else ("moderate" if abs(avg_momentum) > 0.02 else "weak")
        return {
            "average_momentum_pct": avg_momentum * 100.0,
            "momentum_sentiment": momentum_sentiment,
            "instruments_analyzed": len(momentum),
            "trend_strength": trend_strength,
        }

    def _analyze_dollar_correlation_positioning(self, proxy_data: Dict[str, pd.DataFrame]) -> Dict:
        if "dollar_index" not in proxy_data or "gold_futures" not in proxy_data:
            return {"correlation_sentiment": 0.0, "relationship": "insufficient_data"}

        dxy_close = proxy_data["dollar_index"]["Close"].dropna()
        gold_close = proxy_data["gold_futures"]["Close"].dropna()
        common = dxy_close.index.intersection(gold_close.index)
        if len(common) < 30:
            return {"correlation_sentiment": 0.0, "relationship": "insufficient_overlap"}

        dxy_ret = dxy_close.loc[common].pct_change().dropna()
        gold_ret = gold_close.loc[common].pct_change().dropna()
        common2 = dxy_ret.index.intersection(gold_ret.index)
        if len(common2) < 20:
            return {"correlation_sentiment": 0.0, "relationship": "insufficient_data"}

        corr = float(dxy_ret.loc[common2].corr(gold_ret.loc[common2]))
        recent_change = 0.0
        if len(dxy_close) >= 21 and dxy_close.iloc[-21] != 0:
            recent_change = float((dxy_close.iloc[-1] - dxy_close.iloc[-21]) / dxy_close.iloc[-21])

        if recent_change < -0.02:
            correlation_sentiment = 0.5
        elif recent_change > 0.02:
            correlation_sentiment = -0.5
        else:
            correlation_sentiment = float(np.clip(-recent_change * 10.0, -1.0, 1.0))

        relationship = "inverse" if corr < -0.3 else ("positive" if corr > 0.3 else "weak")
        return {
            "gold_usd_correlation": corr,
            "recent_usd_change_pct": recent_change * 100.0,
            "correlation_sentiment": float(np.clip(correlation_sentiment, -1.0, 1.0)),
            "relationship": relationship,
        }

    def _composite(self, analyses: Dict) -> Dict:
        w = self.config["positioning_weights"]
        vs = float(analyses["volume_analysis"].get("volume_sentiment", 0.0))
        ms = float(analyses["momentum_analysis"].get("momentum_sentiment", 0.0))
        cs = float(analyses["correlation_analysis"].get("correlation_sentiment", 0.0))

        composite = float(np.clip(vs * w["volume_momentum"] + ms * w["price_momentum"] + cs * w["dollar_correlation"], -1.0, 1.0))

        # naive confidence: count non-zero components (max 0.8)
        avail = sum(int(abs(x) > 1e-6) for x in (vs, ms, cs))
        confidence = min(0.8, 0.5 + 0.1 * avail)

        return {
            "score": composite,
            "confidence": float(confidence),
            "components": {"volume_positioning": vs, "momentum_positioning": ms, "dollar_correlation": cs},
        }

    # -------- public: aligned with BTC analyzers --------
    async def analyze_gold_cot_positioning(self, cutoff_str: Optional[str] = None) -> Dict:
        """
        Main entry. Returns dict with at least: {"score": [-1,1], "confidence": [0,1], "component_weight": float}
        """
        cutoff = cutoff_str or self.historical_cutoff
        print(f"📊 Analyzing Professional Positioning for Gold - {cutoff or 'current'}...")
        t0 = datetime.now()

        try:
            proxy_data = await self._fetch_positioning_proxy_data(cutoff)
            analyses = {
                "volume_analysis": self._analyze_volume_momentum_positioning(proxy_data),
                "momentum_analysis": self._analyze_price_momentum_positioning(proxy_data),
                "correlation_analysis": self._analyze_dollar_correlation_positioning(proxy_data),
            }
            comp = self._composite(analyses)
            took = (datetime.now() - t0).total_seconds()

            out = {
                "score": comp["score"],
                "confidence": comp["confidence"],
                "component_weight": float(self.config["component_weight"]),
                "weighted_contribution": comp["score"] * float(self.config["component_weight"]),
                "components": comp["components"],
                "market_context": {
                    "data_sources": len(proxy_data),
                    "lookback_days": self.config["lookback_days"],
                    "backtest_date": cutoff or "real_time",
                },
                "kpi_name": "Gold_Professional_Positioning_COT",
                "execution_time_seconds": took,
                "analysis_timestamp": datetime.now().isoformat(),
            }

            print(f"✅ Professional Positioning completed: {out['score']:+.3f}  (conf {out['confidence']:.0%})")
            return out

        except Exception as e:
            print(f"❌ Professional positioning analysis failed: {e}")
            return {
                "score": 0.0,
                "confidence": 0.4,
                "component_weight": float(self.config["component_weight"]) * 0.5,
                "error": f"Positioning analysis exception: {e}",
                "kpi_name": "Gold_Professional_Positioning_COT",
                "analysis_timestamp": datetime.now().isoformat(),
            }


# quick test
if __name__ == "__main__":
    import asyncio

    async def _t():
        az = ProfessionalPositioningAnalyzer(historical_cutoff="2023-03-31")
        res = await az.analyze_gold_cot_positioning("2023-03-31")
        print(res)

    asyncio.run(_t())
