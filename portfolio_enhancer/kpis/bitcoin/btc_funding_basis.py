# btc_funding_basis.py
# HISTORICAL VERSION for backtesting - Bitcoin Funding/Basis Analysis
# Updated: strict historical cutoff, EWMA smoothing, math safety, clearer confidence

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import asyncio
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


def _to_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s).tz_localize(None)


def _ewma(series: pd.Series, halflife_days: float = 365.0) -> pd.Series:
    """Exponentially-weighted smoothing (half-life ~ 1y, calendar days)."""
    if series is None or series.empty:
        return series
    # Use business days approximation: 252 trading ~ 365 calendar; both work fine for smoothing.
    return series.ewm(halflife=halflife_days, min_periods=5, adjust=False).mean()


def _safe_clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        return float(np.clip(x, lo, hi))
    except Exception:
        return 0.0


class BitcoinFundingBasisAnalyzer:
    """
    HISTORICAL VERSION - Bitcoin Funding/Basis Analysis
    - Strict backtest: if backtesting 2023, data is clamped to <= 2022-12-31 (no lookahead)
    - Uses price-based proxies (GBTC, MSTR) for “funding/basis” style signals
    """

    def __init__(self, config: Dict = None, historical_cutoff: Optional[str] = "2022-12-31"):
        self.config = config or self._default_config()
        self.cache = {}
        self.historical_cutoff = _to_ts(historical_cutoff)
        print("Bitcoin Funding/Basis Analyzer (HISTORICAL) initialized")
        print("⚠️ Using price-based proxy for funding rates (real funding data unavailable historically)")
        if self.historical_cutoff is not None:
            print(f"🔒 Historical cutoff set to {self.historical_cutoff.date()}")

    def _default_config(self) -> Dict:
        return {
            'lookback_days': 60,  # 2 months for funding analysis
            'ewma_halflife_days': 365.0,  # ~1y smoothing of noisy sub-signals
            'bitcoin_instruments': {
                'spot': 'BTC-USD',      # Spot Bitcoin
                'grayscale': 'GBTC',    # Grayscale Bitcoin Trust (basis proxy)
                'futures_proxy': 'MSTR' # MicroStrategy as a rough proxy
            },
            'funding_weights': {
                'basis_analysis': 0.50,        # Price basis between instruments
                'volatility_funding': 0.30,    # Volatility-implied funding
                'momentum_funding': 0.20       # Momentum-based funding proxy
            },
            'basis_thresholds': {
                'high_contango': 0.05,   # >5% premium → expensive funding (bearish)
                'moderate_contango': 0.02,
                'fair_value': 0.005,     # within ±0.5% ~ neutral
                'backwardation': -0.02   # < -2% discount → cheap funding (bullish)
            }
        }

    # ===== DATA FETCH (with strict cutoff) =====
    async def fetch_bitcoin_instruments_data(self, backtest_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch Bitcoin instrument data for basis analysis.
        Applies strict historical cutoff to prevent lookahead.
        """
        bt_ts = _to_ts(backtest_date)
        # Determine effective end (min of backtest_date and cutoff)
        if self.historical_cutoff is not None:
            if bt_ts is None:
                effective_end = self.historical_cutoff
            else:
                effective_end = min(bt_ts, self.historical_cutoff)
        else:
            effective_end = bt_ts or pd.Timestamp.utcnow().tz_localize(None)

        lookback = int(self.config['lookback_days'])
        effective_start = effective_end - pd.Timedelta(days=lookback)

        print(f"₿ Fetching Bitcoin instruments data | "
              f"range: {effective_start.date()} → {effective_end.date()} "
              f"(requested={backtest_date or 'current'}, cutoff={self.historical_cutoff.date() if self.historical_cutoff else 'none'})")

        instruments_data: Dict[str, pd.DataFrame] = {}

        for instrument_name, ticker in self.config['bitcoin_instruments'].items():
            try:
                print(f"  📊 Fetching {instrument_name} ({ticker})...")
                hist = yf.Ticker(ticker).history(
                    start=effective_start.strftime('%Y-%m-%d'),
                    end=(effective_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),  # yfinance end is exclusive
                    interval="1d",
                )
                hist = hist.tz_localize(None)
                if not hist.empty:
                    instruments_data[instrument_name] = hist
                    print(f"    ✅ {instrument_name}: {len(hist)} rows")
                else:
                    print(f"    ⚠️ {instrument_name}: no data in range")
            except Exception as e:
                print(f"    ❌ {instrument_name}: {e}")

        print(f"✅ Fetched {len(instruments_data)} instrument datasets")
        return instruments_data

    # ===== COMPONENTS =====
    def analyze_basis_between_instruments(self, instruments_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze premium/discount (basis) of proxies vs spot."""
        if 'spot' not in instruments_data or instruments_data['spot'].empty:
            return {'basis_sentiment': 0.0, 'basis_regime': 'no_spot_data', 'basis_analyses': {}}

        spot_close = instruments_data['spot']['Close'].astype(float)

        basis_analyses = {}
        primary_instrument = None

        for name, df in instruments_data.items():
            if name == 'spot' or df.empty:
                continue

            other = df['Close'].astype(float)
            common = spot_close.index.intersection(other.index)
            if len(common) < 10:
                continue

            s_al = spot_close.loc[common]
            o_al = other.loc[common]

            # GBTC special handling: convert to BTC-equivalent via approx ratio in 2023
            if name == 'grayscale':
                btc_per_share = 0.00091  # approx 2023
                # Avoid division by zero:
                denom = btc_per_share if btc_per_share != 0 else 1e-6
                gbtc_equiv = o_al / denom
                basis = (gbtc_equiv - s_al) / s_al
            else:
                # Rough proxy basis: proxy premium/discount to spot
                basis = (o_al - s_al) / s_al

            # Smooth the basis to avoid noise spikes
            basis_smooth = _ewma(basis, self.config['ewma_halflife_days'])
            current_basis = float(basis_smooth.iloc[-1]) if len(basis_smooth) else 0.0

            # Rolling average reference
            if len(basis_smooth) >= 20:
                avg_basis = float(basis_smooth.rolling(20).mean().iloc[-1])
            else:
                avg_basis = float(current_basis)

            trend = (
                'widening' if current_basis > avg_basis * 1.1 else
                'narrowing' if current_basis < avg_basis * 0.9 else
                'stable'
            )

            basis_analyses[name] = {
                'current_basis_pct': current_basis * 100.0,
                'average_basis_pct': avg_basis * 100.0,
                'basis_trend': trend
            }

        thresholds = self.config['basis_thresholds']

        if basis_analyses:
            primary_instrument = 'grayscale' if 'grayscale' in basis_analyses else list(basis_analyses.keys())[0]
            pb = float(basis_analyses[primary_instrument]['current_basis_pct'] / 100.0)

            if pb > thresholds['high_contango']:
                basis_sentiment = -0.6
                basis_regime = 'high_contango_expensive_funding'
            elif pb > thresholds['moderate_contango']:
                basis_sentiment = -0.3
                basis_regime = 'moderate_contango'
            elif pb < thresholds['backwardation']:
                basis_sentiment = +0.6
                basis_regime = 'backwardation_cheap_funding'
            elif abs(pb) < thresholds['fair_value']:
                basis_sentiment = +0.1
                basis_regime = 'fair_value'
            else:
                basis_sentiment = 0.0
                basis_regime = 'neutral'
        else:
            basis_sentiment = 0.0
            basis_regime = 'insufficient_data'

        return {
            'basis_sentiment': _safe_clip(basis_sentiment),
            'basis_regime': basis_regime,
            'basis_analyses': basis_analyses,
            'primary_instrument': primary_instrument
        }

    def analyze_volatility_implied_funding(self, instruments_data: Dict[str, pd.DataFrame]) -> Dict:
        """Use realized vol as funding stress proxy; smooth with EWMA."""
        if 'spot' not in instruments_data or instruments_data['spot'].empty:
            return {'vol_funding_sentiment': 0.0, 'vol_regime': 'no_data'}

        close = instruments_data['spot']['Close'].astype(float)
        rets = close.pct_change().dropna()
        if len(rets) < 20:
            return {'vol_funding_sentiment': 0.0, 'vol_regime': 'insufficient_data'}

        vol_20 = rets.rolling(20).std() * np.sqrt(365.0)
        vol_20 = _ewma(vol_20, self.config['ewma_halflife_days'])
        if vol_20.empty or np.isnan(vol_20.iloc[-1]):
            return {'vol_funding_sentiment': 0.0, 'vol_regime': 'calc_error'}

        current_vol = float(vol_20.iloc[-1])
        # Percentile back-of-envelope
        vol_percentile = float((vol_20 <= current_vol).mean()) if len(vol_20) > 40 else 0.5

        if current_vol > 1.0:
            sent = -0.4
            reg = 'high_volatility_funding_stress'
        elif current_vol > 0.6:
            sent = -0.2
            reg = 'elevated_volatility'
        elif current_vol < 0.3:
            sent = +0.3
            reg = 'low_volatility_stable_funding'
        else:
            sent = 0.0
            reg = 'normal_volatility'

        # Tail adjustments by percentile
        if vol_percentile > 0.9:
            sent -= 0.2
        elif vol_percentile < 0.1:
            sent += 0.2

        return {
            'current_volatility_pct': current_vol * 100.0,
            'volatility_percentile': vol_percentile,
            'vol_funding_sentiment': _safe_clip(sent),
            'vol_regime': reg
        }

    def analyze_momentum_funding_proxy(self, instruments_data: Dict[str, pd.DataFrame]) -> Dict:
        """Momentum as a funding proxy; smoothed with EWMA."""
        if 'spot' not in instruments_data or instruments_data['spot'].empty:
            return {'momentum_funding_sentiment': 0.0, 'momentum_regime': 'no_data'}

        close = instruments_data['spot']['Close'].astype(float)
        if len(close) < 15:
            return {'momentum_funding_sentiment': 0.0, 'momentum_regime': 'insufficient_data'}

        mom7 = (close.iloc[-1] - close.iloc[-8]) / close.iloc[-8] if len(close) >= 8 else 0.0
        mom14 = (close.iloc[-1] - close.iloc[-15]) / close.iloc[-15] if len(close) >= 15 else 0.0
        comp = 0.6 * mom7 + 0.4 * mom14

        # Smooth composite momentum over time series form
        # Build a small series for smoothing if enough data
        mom_series = pd.Series(close.pct_change(7)).fillna(0.0)
        mom_smooth = _ewma(mom_series, self.config['ewma_halflife_days'])
        if not mom_smooth.empty and not np.isnan(mom_smooth.iloc[-1]):
            comp = float(0.5 * comp + 0.5 * mom_smooth.iloc[-1])

        if comp > 0.15:
            sent = -0.2
            reg = 'strong_upward_momentum_funding_warning'
        elif comp > 0.05:
            sent = +0.2
            reg = 'moderate_upward_momentum'
        elif comp < -0.15:
            sent = +0.3
            reg = 'strong_downward_momentum_cheap_funding'
        elif comp < -0.05:
            sent = -0.1
            reg = 'moderate_downward_momentum'
        else:
            sent = 0.0
            reg = 'neutral_momentum'

        return {
            'momentum_7d_pct': mom7 * 100.0,
            'momentum_14d_pct': mom14 * 100.0,
            'composite_momentum_pct': comp * 100.0,
            'momentum_funding_sentiment': _safe_clip(sent),
            'momentum_regime': reg
        }

    def calculate_funding_basis_composite(self, funding_analyses: Dict) -> Dict:
        """Weighted composite for the funding/basis block."""
        w = self.config['funding_weights']

        # Extract safely; if missing, treat as 0 with lower confidence.
        basis = float(funding_analyses.get('basis_analysis', {}).get('basis_sentiment', 0.0))
        vol   = float(funding_analyses.get('volatility_funding', {}).get('vol_funding_sentiment', 0.0))
        mom   = float(funding_analyses.get('momentum_funding', {}).get('momentum_funding_sentiment', 0.0))

        composite = basis * w['basis_analysis'] + vol * w['volatility_funding'] + mom * w['momentum_funding']
        composite = _safe_clip(composite)

        available = 0
        for k in ('basis_analysis', 'volatility_funding', 'momentum_funding'):
            if k in funding_analyses and any('sentiment' in key for key in funding_analyses[k].keys()):
                available += 1
        total = 3
        confidence = 0.55 + 0.15 * (available / total)  # 0.55 → 0.70

        return {
            'composite_score': composite,
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'component_scores': {
                'basis_analysis': basis,
                'volatility_funding': vol,
                'momentum_funding': mom
            }
        }

    # ===== MAIN =====
    async def analyze_btc_funding_basis(self, backtest_date: str = None) -> Dict:
        """Main entry with strict historical cutoff and smoothing."""
        print(f"₿ Analyzing Bitcoin Funding/Basis for {backtest_date or 'current'}...")
        t0 = datetime.now()

        try:
            instruments_data = await self.fetch_bitcoin_instruments_data(backtest_date)

            funding_analyses = {
                'basis_analysis': self.analyze_basis_between_instruments(instruments_data),
                'volatility_funding': self.analyze_volatility_implied_funding(instruments_data),
                'momentum_funding': self.analyze_momentum_funding_proxy(instruments_data),
            }

            sentiment = self.calculate_funding_basis_composite(funding_analyses)
            elapsed = (datetime.now() - t0).total_seconds()

            result = {
                'component_sentiment': sentiment['composite_score'],
                'component_confidence': sentiment['confidence'],
                'component_weight': 0.30,  # 30% of BTC composite
                'weighted_contribution': sentiment['composite_score'] * 0.30,
                'funding_basis_analysis': funding_analyses,
                'market_context': {
                    'instruments_analyzed': len(instruments_data),
                    'lookback_period_days': self.config['lookback_days'],
                    'analysis_method': 'price_basis_proxy',
                    'funding_summary': self._create_funding_summary(funding_analyses),
                    'backtest_date': backtest_date or 'real_time',
                },
                'component_scores': sentiment['component_scores'],
                'execution_time_seconds': elapsed,
                'kpi_name': 'Bitcoin_Funding_Basis',
                'analysis_timestamp': datetime.now().isoformat(),
            }

            print(f"✅ Bitcoin Funding/Basis Analysis completed: {result['component_sentiment']:.3f}")
            print(f"₿ {result['market_context']['funding_summary']}")
            return result

        except Exception as e:
            print(f"❌ Bitcoin funding/basis analysis failed: {e}")
            return self._create_error_result(f"BTC funding/basis exception: {e}")

    def _create_funding_summary(self, funding_analyses: Dict) -> str:
        basis_regime = funding_analyses.get('basis_analysis', {}).get('basis_regime', 'unknown')
        vol_regime   = funding_analyses.get('volatility_funding', {}).get('vol_regime', 'unknown')
        momentum_reg = funding_analyses.get('momentum_funding', {}).get('momentum_regime', 'unknown')

        if 'expensive_funding' in basis_regime and 'funding_stress' in vol_regime:
            return "🔴 Expensive funding + elevated vol"
        if 'cheap_funding' in basis_regime or 'cheap_funding' in momentum_reg:
            return "🟢 Cheap funding conditions"
        if 'high_volatility' in vol_regime:
            return "🟡 Volatility-induced funding stress"
        if 'fair_value' in basis_regime:
            return "🟡 Basis near fair value"
        return "🟡 Mixed funding signals"

    def _create_error_result(self, error_message: str) -> Dict:
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.15,  # halve on error
            'error': error_message,
            'kpi_name': 'Bitcoin_Funding_Basis',
            'analysis_timestamp': datetime.now().isoformat(),
        }


# Optional ad-hoc test
async def test_btc_funding_basis_analyzer():
    analyzer = BitcoinFundingBasisAnalyzer(historical_cutoff="2022-12-31")
    for date in ["2023-03-31", "2023-09-30", "2023-12-31"]:
        out = await analyzer.analyze_btc_funding_basis(date)
        print(date, out.get("component_sentiment"))

if __name__ == "__main__":
    asyncio.run(test_btc_funding_basis_analyzer())
