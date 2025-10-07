# portfolio_enhancer/weights_config.py
# Central place to specify KPI weights by asset and horizon for the 2023 backtest.
# Horizons: 'M' (Monthly), 'Q' (Quarterly), 'H' (Half-Year), 'Y' (Year)

from typing import Dict

# For now we define only BTC (you can add 'gold', 'equities', 'reits' later).
# Each KPI key must match what btc_sentiment.py expects:
#   'funding_basis' | 'orderflow' | 'micro_momentum'
BTC_HORIZON_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Example defaults — tweak these as you like
    'M': {'funding_basis': 0.25, 'orderflow': 0.45, 'micro_momentum': 0.30},
    'Q': {'funding_basis': 0.30, 'orderflow': 0.40, 'micro_momentum': 0.30},
    'H': {'funding_basis': 0.35, 'orderflow': 0.35, 'micro_momentum': 0.30},
    'Y': {'funding_basis': 0.40, 'orderflow': 0.30, 'micro_momentum': 0.30},
}

# OPTIONAL: sub-period overrides inside 2023 (leave empty now; you can add later)
# Example shape:
# BTC_PERIOD_WEIGHT_SCHEDULE = [
#   {"start": "2023-01-01", "end": "2023-03-31",
#    "weights": {"funding_basis": 0.20, "orderflow": 0.45, "micro_momentum": 0.35}},
# ]
BTC_PERIOD_WEIGHT_SCHEDULE = []
