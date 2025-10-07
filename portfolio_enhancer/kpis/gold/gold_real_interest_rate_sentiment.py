# real_interest_rate_sentiment.py
# HISTORICAL VERSION for backtesting - Real Interest Rates Impact on Gold
# Updated: add TIPS ETF (TIP/SCHP) proxy path when DFII isn't available

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import asyncio
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


class RealRateSentimentScorer:
    """
    HISTORICAL VERSION - Real Interest Rates Sentiment Analysis for Gold
    - Backtesting-safe via backtest_date cutoff
    - Tries DFII (10Y TIPS real yield). If unavailable on Yahoo, falls back to TIP/SCHP ETFs for trend and uses
      nominal 10Y minus simple inflation expectation for level.
    - 25% weight in Gold composite sentiment
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.cache = {}
        print("Real Interest Rate Sentiment Scorer (HISTORICAL) initialized")

    def _default_config(self) -> Dict:
        return {
            'lookback_days': 730,  # 2 years for rates analysis
            'rate_data_sources': {
                'treasury_10y': '^TNX',   # US 10Y Treasury (nominal %)
                'treasury_5y': '^FVX',    # US 5Y Treasury
                'treasury_2y': '^IRX',    # US 13W T-bill (proxy short)
                # 'tips_10y': 'DFII',     # Often fails on Yahoo. We'll handle via ETF proxies below.
            },
            # Additional proxies we will try explicitly:
            'tips_etf_candidates': ['TIP', 'SCHP'],  # TIPS ETFs (price up ≈ real yields down)
            'real_rate_weights': {
                'current_real_rate_level': 0.40,
                'real_rate_trend': 0.35,
                'inflation_expectations': 0.25
            },
            'rate_thresholds': {
                'very_negative': -3.0,
                'negative': -1.0,
                'neutral': 1.0,
                'positive': 2.5,
                'very_positive': 4.0
            }
        }

    async def fetch_rate_data(self, backtest_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch interest rate & proxy data for real rates calculation (historically bounded)
        """
        print(f"📈 Fetching interest rate data for {backtest_date or 'current'}")

        rate_data = {}
        if backtest_date:
            end_date = pd.to_datetime(backtest_date)
            start_date = end_date - pd.Timedelta(days=self.config['lookback_days'])
            print(f"  📅 Historical range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        else:
            end_date = None
            start_date = None

        # Core rates (Yahoo)
        for rate_name, ticker in self.config['rate_data_sources'].items():
            try:
                print(f"  📊 Fetching {rate_name} ({ticker})...")
                t = yf.Ticker(ticker)
                if backtest_date:
                    hist = t.history(start=start_date.strftime('%Y-%m-%d'),
                                     end=end_date.strftime('%Y-%m-%d'),
                                     interval="1d")
                else:
                    hist = t.history(period=f"{self.config['lookback_days']}d", interval="1d")
                if not hist.empty:
                    rate_data[rate_name] = hist
                    print(f"    ✅ {rate_name}: {len(hist)} days")
                else:
                    print(f"    ⚠️ {rate_name}: No data")
            except Exception as e:
                print(f"    ❌ {rate_name}: {e}")

        # TIPS ETF proxies
        for etf in self.config['tips_etf_candidates']:
            try:
                print(f"  📊 Fetching tips_proxy_etf ({etf})...")
                t = yf.Ticker(etf)
                if backtest_date:
                    hist = t.history(start=start_date.strftime('%Y-%m-%d'),
                                     end=end_date.strftime('%Y-%m-%d'),
                                     interval="1d")
                else:
                    hist = t.history(period=f"{self.config['lookback_days']}d", interval="1d")
                if not hist.empty:
                    rate_data.setdefault('tips_proxy', {})[etf] = hist
                    print(f"    ✅ tips_proxy_etf {etf}: {len(hist)} days")
                else:
                    print(f"    ⚠️ tips_proxy_etf {etf}: No data")
            except Exception as e:
                print(f"    ❌ tips_proxy_etf {etf}: {e}")

        print(f"✅ Fetched {len(rate_data)} rate/proxy datasets")
        return rate_data

    def _estimate_inflation_from_nominal(self, nominal_10y: float) -> float:
        # Simple heuristic as before
        if nominal_10y > 4.0:
            return 2.5
        elif nominal_10y > 2.0:
            return 2.0
        else:
            return 1.5

    def _tips_proxy_trend_pp(self, tips_proxy: Dict[str, pd.DataFrame]) -> float:
        """
        Map 3m TIP/SCHP price change to an approximate *percentage points* change in real yields.
        Price ↑ ~ yields ↓. Use scale ≈ 10, i.e., 3% price rise ≈ -0.30 pp real-yield change.
        """
        if not tips_proxy:
            return 0.0
        # Pick the first available proxy with enough history
        for sym, df in tips_proxy.items():
            closes = df.get('Close')
            if closes is None or len(closes) < 63:
                continue
            ret_3m = (closes.iloc[-1] - closes.iloc[-63]) / closes.iloc[-63]
            return float(-ret_3m * 10.0)
        return 0.0

    def calculate_real_interest_rates(self, rate_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate real interest rates:
        - Level: nominal 10Y minus simple inflation expectation (as before).
        - Trend: use TIP/SCHP price trend (if present) mapped into pp; otherwise use nominal-change proxy.
        """
        # Nominal 10Y path (for level)
        if 'treasury_10y' in rate_data and not rate_data['treasury_10y'].empty:
            nom = rate_data['treasury_10y']['Close']
            current_nominal = float(nom.iloc[-1])

            est_infl = self._estimate_inflation_from_nominal(current_nominal)
            est_real_level = current_nominal - est_infl

            # Trend via TIP/SCHP if available
            tips_proxy = rate_data.get('tips_proxy', {})
            tips_trend_pp = self._tips_proxy_trend_pp(tips_proxy)

            # Also compute nominal-change-based trend for fallback/blend
            if len(nom) >= 63:
                past_nom = float(nom.iloc[-63])
            else:
                past_nom = current_nominal
            nominal_trend_pp = (current_nominal - past_nom)  # in percentage points over ~3m

            # Blend: if we have TIP/SCHP, use 70% TIP-based, 30% nominal-based
            if tips_trend_pp != 0.0:
                real_trend_pp = 0.7 * tips_trend_pp + 0.3 * (-nominal_trend_pp)  # nominal ↑ => real ↑, so negative for gold
                method = 'nominal_minus_inflation_with_TIPS_proxy_trend'
                quality = 'medium_high'
            else:
                real_trend_pp = -nominal_trend_pp
                method = 'nominal_minus_inflation_estimate'
                quality = 'medium'

            print(f"  📊 Estimated real rates: level {est_real_level:.2f}% | trend(3m) {real_trend_pp:+.2f} pp")
            return {
                'current_real_rate': float(est_real_level),
                'real_rate_trend_3m': float(real_trend_pp),
                'nominal_rate': float(current_nominal),
                'estimated_inflation': float(est_infl),
                'calculation_method': method,
                'data_quality': quality
            }

        # No rates at all
        print("  ⚠️ No nominal rate data available for real-rate estimate")
        return {
            'current_real_rate': 0.0,
            'real_rate_trend_3m': 0.0,
            'calculation_method': 'no_data',
            'data_quality': 'low'
        }

    def analyze_real_rate_level_sentiment(self, real_rate_data: Dict) -> Dict:
        current_real_rate = real_rate_data['current_real_rate']
        th = self.config['rate_thresholds']
        if current_real_rate <= th['very_negative']:
            s, d = 0.8, 'very_negative_very_bullish'
        elif current_real_rate <= th['negative']:
            s, d = 0.6, 'negative_bullish'
        elif current_real_rate <= th['neutral']:
            s, d = 0.2, 'low_positive_mildly_bullish'
        elif current_real_rate <= th['positive']:
            s, d = -0.3, 'positive_bearish'
        else:
            s, d = -0.7, 'very_positive_very_bearish'
        return {
            'current_real_rate_pct': float(current_real_rate),
            'level_sentiment': float(s),
            'level_description': d,
            'interpretation': self._interpret_real_rate_level(current_real_rate)
        }

    def analyze_real_rate_trend_sentiment(self, real_rate_data: Dict) -> Dict:
        t = float(real_rate_data.get('real_rate_trend_3m', 0.0))
        if t < -0.5:
            s, d = 0.6, 'falling_significantly_bullish'
        elif t < -0.2:
            s, d = 0.3, 'falling_moderately_bullish'
        elif t > 0.5:
            s, d = -0.6, 'rising_significantly_bearish'
        elif t > 0.2:
            s, d = -0.3, 'rising_moderately_bearish'
        else:
            s, d = 0.0, 'stable_neutral'
        return {
            'real_rate_trend_3m': float(t),
            'trend_sentiment': float(s),
            'trend_description': d
        }

    def analyze_inflation_expectations(self, rate_data: Dict[str, pd.DataFrame]) -> Dict:
        if 'treasury_2y' in rate_data and 'treasury_10y' in rate_data:
            y2 = float(rate_data['treasury_2y']['Close'].iloc[-1])
            y10 = float(rate_data['treasury_10y']['Close'].iloc[-1])
            spread = y10 - y2
            if spread > 2.0:
                s, d = 0.4, 'rising_inflation_expectations'
            elif spread > 0.5:
                s, d = 0.2, 'stable_inflation_expectations'
            elif spread < -0.5:
                s, d = -0.3, 'falling_inflation_expectations'
            else:
                s, d = 0.0, 'uncertain_inflation_expectations'
            return {'yield_curve_spread': float(spread), 'inflation_sentiment': float(s), 'inflation_description': d}
        return {'inflation_sentiment': 0.0, 'inflation_description': 'insufficient_data'}

    def _interpret_real_rate_level(self, real_rate: float) -> str:
        if real_rate <= -2.0:
            return f"🟢 VERY BULLISH for Gold - Deeply negative real rates ({real_rate:.1f}%)"
        elif real_rate <= -1.0:
            return f"🟢 BULLISH for Gold - Negative real rates ({real_rate:.1f}%)"
        elif real_rate <= 1.0:
            return f"🟡 NEUTRAL for Gold - Low positive real rates ({real_rate:.1f}%)"
        elif real_rate <= 2.5:
            return f"🔴 BEARISH for Gold - Positive real rates ({real_rate:.1f}%)"
        else:
            return f"🔴 VERY BEARISH for Gold - High positive real rates ({real_rate:.1f}%)"

    def calculate_real_rates_sentiment_composite(self, rate_analyses: Dict) -> Dict:
        w = self.config['real_rate_weights']
        level = rate_analyses['level_analysis']['level_sentiment']
        trend = rate_analyses['trend_analysis']['trend_sentiment']
        infl  = rate_analyses['inflation_analysis']['inflation_sentiment']
        composite = level * w['current_real_rate_level'] + trend * w['real_rate_trend'] + infl * w['inflation_expectations']
        dq = rate_analyses.get('real_rate_data', {}).get('data_quality', 'low')
        if dq == 'medium_high':
            conf = 0.80
        elif dq == 'medium':
            conf = 0.75
        elif dq == 'high':
            conf = 0.85  # keep compatibility if DFII ever works
        else:
            conf = 0.50
        return {
            'composite_score': float(np.clip(composite, -1.0, 1.0)),
            'confidence': float(conf),
            'component_scores': {
                'real_rate_level': float(level),
                'real_rate_trend': float(trend),
                'inflation_expectations': float(infl)
            }
        }

    async def analyze_real_rates_sentiment(self, backtest_date: str = None) -> Dict:
        print(f"📈 Analyzing Real Interest Rates Sentiment for Gold — {backtest_date or 'current'}...")
        start_time = datetime.now()
        try:
            rate_data = await self.fetch_rate_data(backtest_date)
            real_rate_data = self.calculate_real_interest_rates(rate_data)
            analyses = {
                'real_rate_data': real_rate_data,
                'level_analysis': self.analyze_real_rate_level_sentiment(real_rate_data),
                'trend_analysis': self.analyze_real_rate_trend_sentiment(real_rate_data),
                'inflation_analysis': self.analyze_inflation_expectations(rate_data),
            }
            sentiment = self.calculate_real_rates_sentiment_composite(analyses)
            exec_time = (datetime.now() - start_time).total_seconds()
            result = {
                'component_sentiment': sentiment['composite_score'],
                'component_confidence': sentiment['confidence'],
                'component_weight': 0.25,
                'weighted_contribution': sentiment['composite_score'] * 0.25,
                'real_rates_analysis': {
                    'current_real_rate_pct': real_rate_data['current_real_rate'],
                    'real_rate_trend_3m': real_rate_data.get('real_rate_trend_3m', 0.0),
                    'calculation_method': real_rate_data['calculation_method'],
                    'data_quality': real_rate_data['data_quality']
                },
                'market_context': {
                    'rate_data_sources': len(rate_data),
                    'lookback_period_days': self.config['lookback_days'],
                    'primary_interpretation': analyses['level_analysis']['interpretation'],
                    'backtest_date': backtest_date or 'real_time'
                },
                'component_scores': sentiment['component_scores'],
                'execution_time_seconds': exec_time,
                'kpi_name': 'Real_Interest_Rates',
                'analysis_timestamp': datetime.now().isoformat()
            }
            print(f"✅ Real Rates completed: {result['component_sentiment']:+.3f}  (conf {result['component_confidence']*100:.0f}%)")
            print(f"   ↳ {analyses['level_analysis']['interpretation']}")
            return result
        except Exception as e:
            print(f"❌ Real rates analysis failed: {e}")
            return self._create_error_result(f"Real rates exception: {e}")

    def _create_error_result(self, error_message: str) -> Dict:
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.25 * 0.5,
            'error': error_message,
            'kpi_name': 'Real_Interest_Rates',
            'analysis_timestamp': datetime.now().isoformat()
        }


# Test function
async def test_real_rates_analyzer():
    analyzer = RealRateSentimentScorer()
    for date in ["2023-01-31", "2023-05-31", "2023-09-30"]:
        try:
            r = await analyzer.analyze_real_rates_sentiment(date)
            print(f"\n📅 {date}:")
            print(f"   Sentiment: {r.get('component_sentiment', 0):+.3f}")
            print(f"   Confidence: {r.get('component_confidence', 0):.1%}")
            d = r.get('real_rates_analysis', {})
            print(f"   Real Rate (est): {d.get('current_real_rate_pct', 0):.2f}%")
            print(f"   Trend(3m): {d.get('real_rate_trend_3m', 0):+.2f} pp")
            print(f"   Method: {d.get('calculation_method', 'unknown')} | Quality: {d.get('data_quality', 'low')}")
        except Exception as e:
            print(f"❌ {date} failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_real_rates_analyzer())
