# portfolio_enhancer/kpis/gold/gold_kpi_composite.py
# Gold composite KPI: combines COT Positioning, Real Rates, Currency Debasement, and Technical Momentum.
# Backtest-safe: each sub-KPI only sees data up to the passed decision date.

from __future__ import annotations
import asyncio
from typing import Optional, Dict, Iterable, List, Callable
import pandas as pd

# Robust imports that tolerate filename changes
try:
    from portfolio_enhancer.kpis.gold.gold_cot_positioning_analyzer import ProfessionalPositioningAnalyzer
    from portfolio_enhancer.kpis.gold.gold_real_interest_rate_sentiment import RealRateSentimentScorer
    from portfolio_enhancer.kpis.gold.gold_currency_debasement_analyzer import CurrencyDebasementAnalyzer
    from portfolio_enhancer.kpis.gold.gold_momentum_analyzer import GoldMomentumAnalyzer
except ModuleNotFoundError:
    try:
        from .gold_cot_positioning_analyzer import ProfessionalPositioningAnalyzer
        from .gold_real_interest_rate_sentiment import RealRateSentimentScorer
        from .gold_currency_debasement_analyzer import CurrencyDebasementAnalyzer
        from .gold_momentum_analyzer import GoldMomentumAnalyzer
    except ModuleNotFoundError:
        # Final legacy fallbacks
        from .cot_positioning_analyzer import ProfessionalPositioningAnalyzer  # type: ignore
        from .real_interest_rate_sentiment import RealRateSentimentScorer     # type: ignore
        from .currency_debasement_analyzer import CurrencyDebasementAnalyzer  # type: ignore
        from .gold_momentum_analyzer import GoldMomentumAnalyzer               # type: ignore


def _to_ts(s: Optional[str | pd.Timestamp]) -> pd.Timestamp:
    if s is None:
        return pd.Timestamp("2023-01-31")
    if isinstance(s, pd.Timestamp):
        return s.tz_localize(None)
    return pd.to_datetime(s).tz_localize(None)


def _extract_score(res: Dict, pref: str = "component_sentiment") -> float:
    if pref in res:
        return float(res.get(pref, 0.0))
    return float(res.get("sentiment", 0.0))


def _extract_conf(res: Dict, pref: str = "component_confidence") -> float:
    if pref in res:
        return float(res.get(pref, 0.0))
    return float(res.get("confidence", 0.0))


def _find_method(obj: object, preferred: Iterable[str], keywords: Iterable[str] | None = None) -> Optional[Callable]:
    # exact preferred names first
    for name in preferred:
        if hasattr(obj, name):
            cand = getattr(obj, name)
            if callable(cand):
                return cand
    # keyword scan fallback (case-insensitive)
    if keywords:
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            lower = attr.lower()
            if all(k in lower for k in keywords):
                cand = getattr(obj, attr)
                if callable(cand):
                    return cand
    return None


def _call_with_cutoff(fn: Callable, cutoff_str: str):
    # Try kwarg then positional
    try:
        return fn(backtest_date=cutoff_str)
    except TypeError:
        return fn(cutoff_str)


async def gold_kpi_composite(asof: Optional[str | pd.Timestamp],
                             horizon: Optional[str] = None) -> Dict:
    """
    asof: decision date / historical cutoff (YYYY-MM-DD or Timestamp)
    horizon: optional ('M' | 'Q' | 'H' | 'Y'), informational only
    """
    asof_ts = _to_ts(asof)
    asof_str = asof_ts.date().isoformat()

    # Instantiate analyzers (we pass the cutoff when calling the methods)
    cot = ProfessionalPositioningAnalyzer()
    rr  = RealRateSentimentScorer()
    deb = CurrencyDebasementAnalyzer()
    mom = GoldMomentumAnalyzer()

    print(f"🔒 Historical cutoff set to {asof_str}")

    # Resolve methods robustly (exact names first, then keyword scans)
    cot_fn = _find_method(
        cot,
        preferred=("analyze_gold_cot_positioning", "analyze_professional_positioning", "analyze_cot_positioning"),
        keywords=("cot", "position"),
    )
    if cot_fn is None:
        raise AttributeError("COT analyzer method not found on ProfessionalPositioningAnalyzer")

    rr_fn = _find_method(
        rr,
        preferred=("analyze_gold_real_rates_sentiment", "analyze_real_rates_sentiment"),
        keywords=("real", "rate"),
    )
    if rr_fn is None:
        raise AttributeError("Real-rates analyzer method not found on RealRateSentimentScorer")

    deb_fn = _find_method(
        deb,
        preferred=("analyze_gold_currency_debasement", "analyze_currency_debasement"),
        keywords=("debas", "currency"),
    )
    if deb_fn is None:
        raise AttributeError("Debasement analyzer method not found on CurrencyDebasementAnalyzer")

    mom_fn = _find_method(
        mom,
        preferred=("analyze_gold_momentum", "analyze_momentum"),
        keywords=("moment",),  # matches 'momentum'
    )
    if mom_fn is None:
        raise AttributeError("Momentum analyzer method not found on GoldMomentumAnalyzer")

    # Launch all in parallel
    cot_task = _call_with_cutoff(cot_fn, asof_str)
    rr_task  = _call_with_cutoff(rr_fn,  asof_str)
    deb_task = _call_with_cutoff(deb_fn, asof_str)
    mom_task = _call_with_cutoff(mom_fn, asof_str)

    cot_res, rr_res, deb_res, mom_res = await asyncio.gather(
        cot_task, rr_task, deb_task, mom_task
    )

    # Extract standardized scores & confidences
    cot_s  = _extract_score(cot_res)
    rr_s   = _extract_score(rr_res)
    deb_s  = _extract_score(deb_res)
    mom_s  = _extract_score(mom_res)

    cot_c  = _extract_conf(cot_res) or 0.70
    rr_c   = _extract_conf(rr_res)  or 0.75
    deb_c  = _extract_conf(deb_res) or 0.80
    mom_c  = _extract_conf(mom_res) or 0.75

    # Fixed gold weights
    weights = {
        "cot": 0.15,
        "real_rates": 0.25,
        "debasement": 0.25,
        "momentum": 0.25,
        # spare 0.10 previously mentioned stays unused (kept for clarity)
    }
    wsum = sum(weights.values()) or 1.0

    composite = (
        cot_s * weights["cot"] +
        rr_s  * weights["real_rates"] +
        deb_s * weights["debasement"] +
        mom_s * weights["momentum"]
    )
    composite_conf = (
        cot_c * weights["cot"] +
        rr_c  * weights["real_rates"] +
        deb_c * weights["debasement"] +
        mom_c * weights["momentum"]
    ) / wsum

    # Friendly logs
    print(
        f"🥇 Gold KPI composite (asof {asof_str}): {composite:+.3f} | "
        f"components: cot={cot_s:+.3f}, real={rr_s:+.3f}, debase={deb_s:+.3f}, mom={mom_s:+.3f}"
    )
    print(
        f"🥇 Gold KPI composite (asof {asof_str}): {composite:+.3f} | "
        f"cot={cot_s:+.3f}, real_rates={rr_s:+.3f}, debasement={deb_s:+.3f}, momentum={mom_s:+.3f}"
    )

    return {
        "asset": "Gold",
        "as_of": asof_str,
        "horizon": horizon or "",
        "composite_sentiment": float(max(-1.0, min(1.0, composite))),
        "composite": float(composite),
        "composite_confidence": float(max(0.0, min(1.0, composite_conf))),
        "weights_used": weights,
        "components": {
            "cot_positioning": cot_res,
            "real_rates": rr_res,
            "currency_debasement": deb_res,
            "momentum": mom_res,
        },
        "meta": {"weighting_strategy": "fixed_gold_kpi_weights"},
    }
