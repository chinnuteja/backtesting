import asyncio
from typing import Dict

from .reit_accumulation_flow import REITAccumulationFlowAnalyzer
from .reit_technical_momentum import REITTechnicalMomentumAnalyzer
from .reit_yield_spread import REITYieldSpreadAnalyzer


def _score(x) -> float:
    try:
        if isinstance(x, dict):
            for k in ("component_sentiment", "score", "value", "sentiment"):
                if k in x:
                    return float(x[k])
        return float(x)
    except Exception:
        return 0.0


async def reit_kpi_composite(backtest_date: str, horizon: str = "M") -> Dict:
    """
    Rolling historical composite for REITs.
    Components:
      - Accumulation Flow (30%)
      - Technical Momentum (35%)
      - Yield Spread (35%)
    """
    print("REIT KPI Composite (HISTORICAL) initialized")
    print(f"🔒 Historical cutoff set to {backtest_date}")

    acc = REITAccumulationFlowAnalyzer()
    mom = REITTechnicalMomentumAnalyzer()
    yld = REITYieldSpreadAnalyzer()

    acc_task = acc.analyze_reit_accumulation_flow(backtest_date)
    mom_task = mom.analyze_reit_technical_momentum(backtest_date)
    yld_task = yld.analyze_reit_yield_spread(backtest_date)

    acc_res, mom_res, yld_res = await asyncio.gather(acc_task, mom_task, yld_task)

    weights = {"accumulation_flow": 0.30, "technical_momentum": 0.35, "yield_spread": 0.35}
    s_acc = _score(acc_res)
    s_mom = _score(mom_res)
    s_yld = _score(yld_res)

    composite = float(0.30 * s_acc + 0.35 * s_mom + 0.35 * s_yld)

    components = {
        "accumulation_flow": acc_res,
        "technical_momentum": mom_res,
        "yield_spread": yld_res,
    }

    print(
        f"🏢 REIT KPI composite (asof {backtest_date}): {composite:+0.3f} | "
        f"components: acc={s_acc:+0.3f}, tech={s_mom:+0.3f}, yield={s_yld:+0.3f}"
    )

    return {
        "asset": "REITs",
        "decision_date": backtest_date,
        "horizon": horizon,
        "composite_sentiment": composite,
        "weights": weights,
        "components": components,
    }
