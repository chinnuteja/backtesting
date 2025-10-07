# -*- coding: utf-8 -*-
"""
NIFTY Composite KPI (historical/backtest safe)

Combines:
- Technicals (function): nifty_technical_kpi              [30%]
- FII/DII flows (class): FIIDIIFlowsAnalyzer               [35%]
- Market sentiment/VIX (class): MarketSentimentAnalyzer    [20%]
- RBI/Rate policy proxy (class): RBIInterestRatesAnalyzer  [10%]
- Global factors (class): GlobalFactorsAnalyzer            [ 5%]

Implementation note:
- We LAZY-IMPORT analyzer modules and use getattr to avoid hard symbol
  imports at module load time. This removes the “cannot import name …”
  failure and breaks circular/timing issues on Windows.
"""

from __future__ import annotations
import asyncio
from typing import Dict, Any

# weights for the composite
W = {
    "technical": 0.30,
    "flows":     0.35,
    "market":    0.20,
    "rbi":       0.10,
    "global":    0.05,
}

def _safe_get(module, name: str):
    """Return attribute if present else None."""
    return getattr(module, name, None)

async def _run_technical(asof: str, horizon: str) -> Dict[str, Any]:
    # function-based KPI
    from . import nifty_technical_analyzer as tech_mod
    func = _safe_get(tech_mod, "nifty_technical_kpi")
    if func is None:
        return {"composite_sentiment": 0.0, "confidence": 0.4, "error": "missing: nifty_technical_kpi"}
    return await func(asof, horizon)

async def _run_flows(asof: str) -> Dict[str, Any]:
    # class-based KPI (FII/DII)
    from . import fii_dii_flows_analyzer as flows_mod
    Cls = _safe_get(flows_mod, "FIIDIIFlowsAnalyzer")
    if Cls is None:
        return {"composite_sentiment": 0.0, "confidence": 0.4, "error": "missing: FIIDIIFlowsAnalyzer"}
    analyzer = Cls()
    return await analyzer.analyze_fii_dii_sentiment(asof)

async def _run_market(asof: str) -> Dict[str, Any]:
    from . import market_sentiment_analyzer as mkt_mod
    Cls = _safe_get(mkt_mod, "MarketSentimentAnalyzer")
    if Cls is None:
        return {"composite_sentiment": 0.0, "confidence": 0.4, "error": "missing: MarketSentimentAnalyzer"}
    analyzer = Cls()
    return await analyzer.analyze_market_sentiment(asof)

async def _run_rbi(asof: str) -> Dict[str, Any]:
    from . import rbi_policy_analyzer as rbi_mod
    Cls = _safe_get(rbi_mod, "RBIInterestRatesAnalyzer")
    if Cls is None:
        return {"composite_sentiment": 0.0, "confidence": 0.4, "error": "missing: RBIInterestRatesAnalyzer"}
    analyzer = Cls()
    return await analyzer.analyze_rbi_policy_sentiment(asof)

async def _run_global(asof: str) -> Dict[str, Any]:
    from . import global_factors_analyzer as glob_mod
    Cls = _safe_get(glob_mod, "GlobalFactorsAnalyzer")
    if Cls is None:
        return {"composite_sentiment": 0.0, "confidence": 0.4, "error": "missing: GlobalFactorsAnalyzer"}
    analyzer = Cls()
    return await analyzer.analyze_global_factors_sentiment(asof)

def _score(d: Dict[str, Any]) -> float:
    return float(d.get("component_sentiment") or d.get("composite_sentiment") or 0.0)

def _conf(d: Dict[str, Any]) -> float:
    return float(d.get("component_confidence") or d.get("confidence") or 0.5)

async def analyze_nifty_composite(asof: str, horizon: str = "M") -> Dict[str, Any]:
    """
    Return a unified NIFTY composite KPI dict (used by backtester).
    Output keys:
      - composite_sentiment (float in [-1,1])
      - component_confidence (float 0..1)
      - component_breakdown (dict of individual KPI scores)
    """
    # run sub-KPIs concurrently where safe
    tech_coro   = _run_technical(asof, horizon)
    flows_coro  = _run_flows(asof)
    market_coro = _run_market(asof)
    rbi_coro    = _run_rbi(asof)
    glob_coro   = _run_global(asof)

    tech, flows, market, rbi, glob = await asyncio.gather(
        tech_coro, flows_coro, market_coro, rbi_coro, glob_coro, return_exceptions=False
    )

    s_tech   = _score(tech)
    s_flows  = _score(flows)
    s_market = _score(market)
    s_rbi    = _score(rbi)
    s_glob   = _score(glob)

    comp = (
        s_tech   * W["technical"] +
        s_flows  * W["flows"]     +
        s_market * W["market"]    +
        s_rbi    * W["rbi"]       +
        s_glob   * W["global"]
    )

    # simple confidence aggregation (weighted average of confidences)
    conf = (
        _conf(tech)   * W["technical"] +
        _conf(flows)  * W["flows"]     +
        _conf(market) * W["market"]    +
        _conf(rbi)    * W["rbi"]       +
        _conf(glob)   * W["global"]
    )

    return {
        "kpi": "NIFTY_COMPOSITE",
        "asof": asof,
        "horizon": horizon,
        "composite_sentiment": float(max(-1.0, min(1.0, comp))),
        "component_confidence": float(max(0.0, min(1.0, conf))),
        "component_breakdown": {
            "technical": s_tech,
            "flows": s_flows,
            "market": s_market,
            "rbi": s_rbi,
            "global": s_glob,
        },
        "raw_components": {
            "technical": tech,
            "flows": flows,
            "market": market,
            "rbi": rbi,
            "global": glob,
        },
    }
