# portfolio_enhancer/kpis/bitcoin/btc_sentiment.py
# Combine BTC KPIs into one composite sentiment, backtest-safe, with horizon-based weights.

from __future__ import annotations
import asyncio
from typing import Dict, Optional

import pandas as pd

from .btc_funding_basis import BitcoinFundingBasisAnalyzer
from .btc_orderflow import BitcoinOrderflowAnalyzer
from .btc_micro_momentum import BitcoinMicroMomentumAnalyzer
from ...weights_config import BTC_HORIZON_WEIGHTS, BTC_PERIOD_WEIGHT_SCHEDULE


def _to_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s).tz_localize(None)


def _normalize(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in w.values())
    if s <= 0:
        n = len(w) or 1
        return {k: 1.0 / n for k in w.keys()}
    return {k: max(0.0, float(v)) / s for k, v in w.items()}


def _pick_period_override(backtest_ts: Optional[pd.Timestamp]) -> Optional[Dict[str, float]]:
    """Optional: if you later add date-range overrides inside 2023."""
    if not BTC_PERIOD_WEIGHT_SCHEDULE or backtest_ts is None:
        return None
    for item in BTC_PERIOD_WEIGHT_SCHEDULE:
        start = _to_ts(item.get("start"))
        end = _to_ts(item.get("end"))
        if start is None or end is None:
            continue
        if start <= backtest_ts <= end:
            w = item.get("weights") or {}
            usable = {k: float(v) for k, v in w.items() if k in {"funding_basis", "orderflow", "micro_momentum"}}
            if usable:
                return _normalize(usable)
    return None


def _horizon_weights(h: str) -> Dict[str, float]:
    base = BTC_HORIZON_WEIGHTS.get(h, BTC_HORIZON_WEIGHTS.get('M', {'funding_basis': .3, 'orderflow': .4, 'micro_momentum': .3}))
    return _normalize(base)


async def analyze_btc_sentiment(
    backtest_date: Optional[str],
    horizon: str,  # 'M' | 'Q' | 'H' | 'Y'
    historical_cutoff: str = "2022-12-31",
) -> Dict:
    """
    backtest_date: date *within 2023* that marks the prediction point (we only use data <= cutoff)
    horizon: 'M' (monthly), 'Q' (quarterly), 'H' (half-year), 'Y' (year)
    """
    ts = _to_ts(backtest_date) or pd.Timestamp("2023-01-31")

    # >>> IMPORTANT: unfreeze KPIs by passing the decision date as cutoff <<<
    cutoff_str = (ts.date().isoformat())

    # instantiate KPI analyzers (now cut off by decision date)
    funding = BitcoinFundingBasisAnalyzer(historical_cutoff=cutoff_str)
    orderfl = BitcoinOrderflowAnalyzer(historical_cutoff=cutoff_str)
    micro   = BitcoinMicroMomentumAnalyzer(historical_cutoff=cutoff_str)

    # parallel run (also pass backtest_date to the calls themselves)
    f_task = funding.analyze_btc_funding_basis(backtest_date)
    o_task = orderfl.analyze_btc_orderflow(backtest_date)
    m_task = micro.analyze_btc_micro_momentum(backtest_date)
    f_res, o_res, m_res = await asyncio.gather(f_task, o_task, m_task)

    # base horizon weights
    weights = _horizon_weights(horizon)
    # optional date overrides (if you set them)
    override = _pick_period_override(ts)
    if override:
        weights = override

    # safety normalize
    weights = _normalize(weights)

    # scores & confidences
    f_score = float(f_res.get("component_sentiment", 0.0)); f_conf = float(f_res.get("component_confidence", 0.6))
    o_score = float(o_res.get("component_sentiment", 0.0)); o_conf = float(o_res.get("component_confidence", 0.7))
    m_score = float(m_res.get("component_sentiment", 0.0)); m_conf = float(m_res.get("component_confidence", 0.65))

    composite = (
        f_score * weights["funding_basis"] +
        o_score * weights["orderflow"] +
        m_score * weights["micro_momentum"]
    )
    composite_conf = (
        f_conf * weights["funding_basis"] +
        o_conf * weights["orderflow"] +
        m_conf * weights["micro_momentum"]
    )

    return {
        "asset": "Bitcoin",
        "as_of": ts.date().isoformat(),
        "horizon": horizon,
        "weights_used": weights,
        "composite_sentiment": float(max(-1.0, min(1.0, composite))),
        "composite_confidence": float(max(0.0, min(1.0, composite_conf))),
        "components": {
            "funding_basis": f_res,
            "orderflow": o_res,
            "micro_momentum": m_res
        },
        "meta": {
            "historical_cutoff": cutoff_str,
            "weighting_strategy": "horizon" + ("+period_override" if override else ""),
        }
    }


def run_btc_sentiment(backtest_date: Optional[str], horizon: str, historical_cutoff: str = "2022-12-31") -> Dict:
    return asyncio.run(analyze_btc_sentiment(backtest_date, horizon, historical_cutoff))
