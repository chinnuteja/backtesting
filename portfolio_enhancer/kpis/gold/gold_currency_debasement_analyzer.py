# portfolio_enhancer/gold_currency_debasement_analyzer.py
# HISTORICAL VERSION — Currency Debasement KPI for Gold
# Aligned with our BTC-style interfaces: accepts historical_cutoff and returns {"score": ...}

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class CurrencyDebasementAnalyzer:
    """
    USD debasement proxy for Gold.
    Suggested component weight: 25% in Gold composite.
    """

    def __init__(self, historical_cutoff: Optional[str] = None, config: Dict = None):
        self.config = config or self._default_config()
        self.historical_cutoff = pd.to_datetime(historical_cutoff).strftime("%Y-%m-%d") if historical_cutoff else None
        self.cache: Dict = {}
        print("Currency Debasement Analyzer (HISTORICAL) initialized")
        if self.historical_cutoff:
            print(f"🔒 Historical cutoff set to {self.historical_cutoff}")

    def _default_config(self) -> Dict:
        return {
            "lookback_days": 365,  # ~1y
            "currency_indicators": {
                "dxy_index": "DX-Y.NYB",   # Dollar Index (best-effort in yfinance)
                "eur_usd": "EURUSD=X",     # EURUSD (↑ = USD weaker)
                "gbp_usd": "GBPUSD=X",     # GBPUSD (↑ = USD weaker)
                "usd_jpy": "JPY=X",        # USDJPY proxy; NOTE: Yahoo's JPY=X is USDJPY (↑ = USD stronger)
                "treasury_10y": "^TNX",    # 10Y nominal
                "money_supply_proxy": "^GSPC",  # SPX as loose liquidity proxy
            },
            "debasement_weights": {
                "dxy_weakness": 0.40,
                "currency_cross_weakness": 0.25,
                "real_rates_decline": 0.20,
                "liquidity_expansion": 0.15,
            },
            "component_weight": 0.25,  # in Gold composite
        }

    async def _fetch_currency_data(self, cutoff_str: Optional[str]) -> Dict[str, pd.DataFrame]:
        print(f"💵 Fetching currency debasement data for {cutoff_str or 'current'}")
        data: Dict[str, pd.DataFrame] = {}

        end_date = pd.to_datetime(cutoff_str) if cutoff_str else pd.Timestamp.today().normalize()
        start_date = end_date - pd.Timedelta(days=int(self.config["lookback_days"]))
        if cutoff_str:
            print(f"  📅 Historical range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")

        for name, ticker in self.config["currency_indicators"].items():
            try:
                print(f"  📊 Fetching {name} ({ticker})...")
                t = yf.Ticker(ticker)
                hist = t.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="1d",
                )
                if not hist.empty:
                    data[name] = hist[["Close"]].copy()
                    print(f"    ✅ {name}: {len(hist)} days")
                else:
                    print(f"    ⚠️ {name}: No data")
            except Exception as e:
                print(f"    ❌ {name}: {e}")
                continue

        print(f"✅ Fetched {len(data)} currency datasets")
        return data

    # ---------- components ----------
    def _dxy_weakness(self, data: Dict[str, pd.DataFrame]) -> Dict:
        if "dxy_index" not in data or data["dxy_index"].empty:
            return {"dxy_sentiment": 0.0, "trend": "no_data"}
        dxy = data["dxy_index"]["Close"].dropna()
        cur = float(dxy.iloc[-1])

        ch_1m = float((dxy.iloc[-1] - dxy.iloc[-21]) / dxy.iloc[-21]) if len(dxy) >= 22 else 0.0
        ch_3m = float((dxy.iloc[-1] - dxy.iloc[-63]) / dxy.iloc[-63]) if len(dxy) >= 64 else 0.0
        ch_6m = float((dxy.iloc[-1] - dxy.iloc[-126]) / dxy.iloc[-126]) if len(dxy) >= 127 else 0.0

        weighted = ch_1m * 0.5 + ch_3m * 0.3 + ch_6m * 0.2
        # USD weakness (DXY down) => bullish gold => positive sentiment
        dxy_sentiment = float(np.clip(-weighted * 8.0, -1.0, 1.0))

        if weighted < -0.03:
            trend = "significant_weakness_bullish_gold"
        elif weighted < -0.01:
            trend = "moderate_weakness_bullish_gold"
        elif weighted > 0.03:
            trend = "significant_strength_bearish_gold"
        elif weighted > 0.01:
            trend = "moderate_strength_bearish_gold"
        else:
            trend = "stable_neutral"

        return {
            "current_dxy_level": cur,
            "dxy_change_1m_pct": ch_1m * 100.0,
            "dxy_change_3m_pct": ch_3m * 100.0,
            "weighted_dxy_change_pct": weighted * 100.0,
            "dxy_sentiment": dxy_sentiment,
            "trend": trend,
        }

    def _currency_cross_weakness(self, data: Dict[str, pd.DataFrame]) -> Dict:
        pairs = []
        # EURUSD, GBPUSD: ↑ => USD weaker (positive); JPY=X (USDJPY): ↑ => USD stronger (negative)
        for key in ("eur_usd", "gbp_usd", "usd_jpy"):
            if key in data and not data[key].empty:
                s = data[key]["Close"].dropna()
                if len(s) >= 64 and s.iloc[-63] != 0:
                    ch = float((s.iloc[-1] - s.iloc[-63]) / s.iloc[-63])
                    if key == "usd_jpy":
                        ch = -ch  # invert (↑ USDJPY => USD stronger => bearish gold)
                    pairs.append(ch)

        if not pairs:
            return {"cross_sentiment": 0.0, "pairs_analyzed": 0}

        avg = float(np.mean(pairs))
        sent = float(np.clip(avg * 5.0, -1.0, 1.0))
        return {
            "average_usd_change_pct": avg * 100.0,
            "cross_sentiment": sent,
            "pairs_analyzed": len(pairs),
            "individual_changes_pct": [float(x * 100.0) for x in pairs],
        }

    def _real_rates_decline(self, data: Dict[str, pd.DataFrame]) -> Dict:
        if "treasury_10y" not in data or data["treasury_10y"].empty:
            return {"real_rate_sentiment": 0.0, "trend": "no_data"}

        y = data["treasury_10y"]["Close"].dropna()
        cur = float(y.iloc[-1])
        ch_3m = float(y.iloc[-1] - y.iloc[-63]) if len(y) >= 64 else 0.0

        # coarse inflation assumption; sign-only proxy
        est_infl = 3.5
        est_real = cur - est_infl

        if est_real < -2.0:
            s = 0.7
        elif est_real < 0.0:
            s = 0.4
        elif est_real > 2.0:
            s = -0.5
        else:
            s = -est_real * 0.2

        if ch_3m < -0.5:
            s += 0.2
        elif ch_3m > 0.5:
            s -= 0.2

        return {
            "current_yield_pct": cur,
            "yield_change_3m": ch_3m,
            "estimated_real_rate_pct": est_real,
            "real_rate_sentiment": float(np.clip(s, -1.0, 1.0)),
        }

    def _liquidity_expansion(self, data: Dict[str, pd.DataFrame]) -> Dict:
        if "money_supply_proxy" not in data or data["money_supply_proxy"].empty:
            return {"liquidity_sentiment": 0.0, "trend": "no_data"}

        spx = data["money_supply_proxy"]["Close"].dropna()
        r_1m = float((spx.iloc[-1] - spx.iloc[-21]) / spx.iloc[-21]) if len(spx) >= 22 else 0.0
        r_3m = float((spx.iloc[-1] - spx.iloc[-63]) / spx.iloc[-63]) if len(spx) >= 64 else 0.0
        w = r_1m * 0.6 + r_3m * 0.4

        if w > 0.10:
            s = 0.4
        elif w > 0.05:
            s = 0.2
        elif w < -0.10:
            s = -0.3
        elif w < -0.05:
            s = -0.1
        else:
            s = w * 2.0

        return {
            "sp500_return_1m_pct": r_1m * 100.0,
            "sp500_return_3m_pct": r_3m * 100.0,
            "weighted_return_pct": w * 100.0,
            "liquidity_sentiment": float(s),
        }

    def _composite(self, blocks: Dict) -> Dict:
        w = self.config["debasement_weights"]
        dxy = float(blocks["dxy_analysis"].get("dxy_sentiment", 0.0))
        crx = float(blocks["currency_cross_analysis"].get("cross_sentiment", 0.0))
        rrs = float(blocks["real_rates_analysis"].get("real_rate_sentiment", 0.0))
        liq = float(blocks["liquidity_analysis"].get("liquidity_sentiment", 0.0))

        comp = float(np.clip(dxy * w["dxy_weakness"] + crx * w["currency_cross_weakness"]
                             + rrs * w["real_rates_decline"] + liq * w["liquidity_expansion"], -1.0, 1.0))

        avail = sum(int(abs(x) > 1e-6) for x in (dxy, crx, rrs, liq))
        conf = 0.6 + (avail / 4.0) * 0.3  # 0.6 .. 0.9
        return {
            "score": comp,
            "confidence": float(np.clip(conf, 0.0, 1.0)),
            "components": {
                "dxy_weakness": dxy, "currency_cross_weakness": crx,
                "real_rates_decline": rrs, "liquidity_expansion": liq
            }
        }

    # -------- public entry --------
    async def analyze_gold_currency_debasement(self, cutoff_str: Optional[str] = None) -> Dict:
        cutoff = cutoff_str or self.historical_cutoff
        print(f"💵 Analyzing Currency Debasement for Gold — {cutoff or 'current'}...")
        t0 = datetime.now()
        try:
            data = await self._fetch_currency_data(cutoff)
            blocks = {
                "dxy_analysis": self._dxy_weakness(data),
                "currency_cross_analysis": self._currency_cross_weakness(data),
                "real_rates_analysis": self._real_rates_decline(data),
                "liquidity_analysis": self._liquidity_expansion(data),
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
                    "data_sources": len(data),
                    "lookback_days": self.config["lookback_days"],
                    "backtest_date": cutoff or "real_time",
                    "debasement_summary": self._summary(blocks),
                },
                "kpi_name": "Gold_Currency_Debasement",
                "execution_time_seconds": took,
                "analysis_timestamp": datetime.now().isoformat(),
            }
            print(f"✅ Currency Debasement completed: {out['score']:+.3f}  (conf {out['confidence']:.0%})")
            print(f"   ↳ {out['market_context']['debasement_summary']}")
            return out
        except Exception as e:
            print(f"❌ Currency debasement analysis failed: {e}")
            return {
                "score": 0.0,
                "confidence": 0.4,
                "component_weight": float(self.config["component_weight"]) * 0.5,
                "error": f"Debasement analysis exception: {e}",
                "kpi_name": "Gold_Currency_Debasement",
                "analysis_timestamp": datetime.now().isoformat(),
            }

    def _summary(self, blocks: Dict) -> str:
        dxy_trend = blocks.get("dxy_analysis", {}).get("trend", "unknown")
        real_rate = blocks.get("real_rates_analysis", {}).get("estimated_real_rate_pct", 0.0)
        cross = float(blocks.get("currency_cross_analysis", {}).get("cross_sentiment", 0.0))
        if "significant_weakness" in dxy_trend and real_rate < -1.0:
            return "🟢 Strong debasement: weak DXY & negative real rates"
        if "weakness" in dxy_trend and cross > 0.2:
            return "🟢 Broad USD weakness vs majors"
        if "strength" in dxy_trend and real_rate > 2.0:
            return "🔴 USD strength with positive real rates"
        if real_rate < -1.0:
            return "🟡 Negative real rates but mixed USD"
        return "🟡 Neutral debasement environment"


# quick test
if __name__ == "__main__":
    import asyncio
    async def _t():
        az = CurrencyDebasementAnalyzer(historical_cutoff="2023-11-30")
        res = await az.analyze_gold_currency_debasement("2023-11-30")
        print(res)
    asyncio.run(_t())
