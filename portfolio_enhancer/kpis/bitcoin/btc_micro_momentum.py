# btc_micro_momentum.py
# HISTORICAL VERSION for backtesting - Bitcoin Micro Momentum Analysis
# Updated: strict historical cutoff (no-lookahead), EWMA smoothing, safer math

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


def _ewma(series: pd.Series, halflife_days: float = 365.0, min_periods: int = 5) -> pd.Series:
    if series is None or series.empty:
        return series
    return series.ewm(halflife=halflife_days, min_periods=min_periods, adjust=False).mean()


def _safe_clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        return float(np.clip(x, lo, hi))
    except Exception:
        return 0.0


class BitcoinMicroMomentumAnalyzer:
    """
    HISTORICAL VERSION - Bitcoin Micro Momentum Analysis (short-term momentum, vol, volume, persistence)
    - Strict historical cutoff to avoid lookahead for 2023 backtests
    - 35% weight in Bitcoin composite sentiment
    """

    def __init__(self, config: Dict = None, historical_cutoff: Optional[str] = "2022-12-31"):
        self.config = config or self._default_config()
        self.cache = {}
        self.historical_cutoff = _to_ts(historical_cutoff)
        print("Bitcoin Micro Momentum Analyzer (HISTORICAL) initialized")
        if self.historical_cutoff is not None:
            print(f"🔒 Historical cutoff set to {self.historical_cutoff.date()} (prevents lookahead)")

    def _default_config(self) -> Dict:
        return {
            'lookback_days': 30,  # ≈1 month
            'ewma_halflife_days': 365.0,  # ~1y smoothing to reduce noise
            # Note: '^CME_BTC1!' may not resolve via yfinance; kept as best-effort, we fallback gracefully
            'bitcoin_tickers': ['BTC-USD', 'GBTC', '^CME_BTC1!'],
            'momentum_weights': {
                'short_term_momentum': 0.40,
                'intraday_volatility': 0.25,
                'volume_momentum': 0.20,
                'trend_persistence': 0.15
            },
            'momentum_periods': [1, 3, 7, 14],
            'volatility_lookback': 20
        }

    # ===== DATA FETCH (with strict cutoff) =====
    async def fetch_bitcoin_data(self, backtest_date: str = None) -> pd.DataFrame:
        """
        Fetch Bitcoin price data, clamped to <= historical_cutoff (no lookahead).
        """
        bt_ts = _to_ts(backtest_date)
        if self.historical_cutoff is not None:
            effective_end = min(bt_ts, self.historical_cutoff) if bt_ts is not None else self.historical_cutoff
        else:
            effective_end = bt_ts or pd.Timestamp.utcnow().tz_localize(None)

        lookback = int(self.config['lookback_days'])
        effective_start = effective_end - pd.Timedelta(days=lookback)

        print(f"₿ Fetching Bitcoin price data | "
              f"range: {effective_start.date()} → {effective_end.date()} "
              f"(requested={backtest_date or 'current'}, cutoff={self.historical_cutoff.date() if self.historical_cutoff else 'none'})")

        # Try multiple sources; first one that returns adequate rows wins
        for ticker in self.config['bitcoin_tickers']:
            try:
                print(f"  📊 Trying {ticker}...")
                hist = yf.Ticker(ticker).history(
                    start=effective_start.strftime('%Y-%m-%d'),
                    end=(effective_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),  # yfinance end is exclusive
                    interval="1d"
                )
                hist = hist.tz_localize(None)
                if not hist.empty and len(hist) >= 10:
                    print(f"    ✅ Successfully fetched {len(hist)} rows from {ticker}")
                    return hist
                else:
                    print(f"    ⚠️ Insufficient data from {ticker}: {len(hist)} rows")
            except Exception as e:
                print(f"    ❌ Failed to fetch from {ticker}: {e}")

        raise ValueError(f"Unable to fetch Bitcoin data for {backtest_date or 'current date'}")

    # ===== COMPONENTS =====
    def calculate_short_term_momentum(self, btc_data: pd.DataFrame) -> Dict:
        """Short-term momentum across multiple periods; lightly smoothed."""
        closes = btc_data['Close'].astype(float)
        if closes.empty:
            return {'momentum_sentiment': 0.0, 'trend_strength': 'weak', 'trend_direction': 'neutral',
                    'individual_momentums': {}, 'composite_momentum_pct': 0.0}

        current_price = float(closes.iloc[-1])
        momentum_scores: Dict[str, float] = {}

        for period in self.config['momentum_periods']:
            if len(closes) >= period + 1:
                past = float(closes.iloc[-(period + 1)])
                if past != 0:
                    momentum_scores[f'{period}d_momentum'] = (current_price - past) / past
                else:
                    momentum_scores[f'{period}d_momentum'] = 0.0
            else:
                momentum_scores[f'{period}d_momentum'] = 0.0

        m1 = momentum_scores.get('1d_momentum', 0.0)
        m3 = momentum_scores.get('3d_momentum', 0.0)
        m7 = momentum_scores.get('7d_momentum', 0.0)
        m14 = momentum_scores.get('14d_momentum', 0.0)

        composite = 0.40 * m1 + 0.30 * m3 + 0.20 * m7 + 0.10 * m14

        # Add a touch of smoothing: 3d change series EWMA blended in
        chg_3d = closes.pct_change(3).fillna(0.0)
        chg_ewm = _ewma(chg_3d, self.config['ewma_halflife_days'])
        if chg_ewm is not None and not chg_ewm.empty and not np.isnan(chg_ewm.iloc[-1]):
            composite = 0.7 * composite + 0.3 * float(chg_ewm.iloc[-1])

        momentum_sent = float(np.tanh(composite * 10.0))  # squash to [-1,1]

        trend_strength = 'strong' if abs(composite) > 0.05 else ('moderate' if abs(composite) > 0.02 else 'weak')
        trend_direction = 'bullish' if composite > 0.01 else ('bearish' if composite < -0.01 else 'neutral')

        return {
            'individual_momentums': {k: float(v) for k, v in momentum_scores.items()},
            'composite_momentum_pct': composite * 100.0,
            'momentum_sentiment': _safe_clip(momentum_sent),
            'trend_strength': trend_strength,
            'trend_direction': trend_direction
        }

    def calculate_intraday_volatility_pattern(self, btc_data: pd.DataFrame) -> Dict:
        """Daily high-low/close as intraday volatility proxy."""
        if len(btc_data) < 5:
            return {'volatility_sentiment': 0.0, 'pattern': 'insufficient_data',
                    'current_daily_range_pct': 0.0, 'volatility_ratio': 1.0, 'volatility_trend': 0.0}

        high = btc_data['High'].astype(float)
        low = btc_data['Low'].astype(float)
        close = btc_data['Close'].astype(float)

        daily_range = (high - low) / close.replace(0, np.nan)
        daily_range = daily_range.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Smooth ranges a touch
        dr_smooth = _ewma(daily_range, self.config['ewma_halflife_days'])
        current_range = float(dr_smooth.iloc[-1]) if not dr_smooth.empty else float(daily_range.iloc[-1])

        vol_lb = int(self.config['volatility_lookback'])
        avg_range = float(daily_range.rolling(vol_lb).mean().iloc[-1]) if len(daily_range) >= vol_lb else float(
            daily_range.mean())

        volatility_ratio = current_range / avg_range if avg_range > 0 else 1.0

        # Vol trend: last 5 vs prior 10 (if available)
        recent = float(daily_range.tail(5).mean())
        older = float(daily_range.tail(15).head(10).mean()) if len(daily_range) >= 15 else recent
        vol_trend = (recent - older) / (older + 1e-9)

        if volatility_ratio > 2.0:
            vol_sent = -0.4
        elif volatility_ratio > 1.5:
            # direction-aware
            if len(close) >= 2 and close.iloc[-2] != 0:
                price_chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
            else:
                price_chg = 0.0
            vol_sent = 0.3 if price_chg > 0 else -0.3
        elif volatility_ratio < 0.5:
            vol_sent = 0.2
        else:
            vol_sent = 0.0

        pattern = 'expanding' if vol_trend > 0.1 else ('contracting' if vol_trend < -0.1 else 'stable')

        return {
            'current_daily_range_pct': current_range * 100.0,
            'volatility_ratio': float(volatility_ratio),
            'volatility_trend': float(vol_trend),
            'volatility_sentiment': _safe_clip(vol_sent),
            'pattern': pattern
        }

    def calculate_volume_momentum(self, btc_data: pd.DataFrame) -> Dict:
        """Volume momentum relative to 10d mean, direction-aware."""
        if 'Volume' not in btc_data.columns or float(btc_data['Volume'].sum()) == 0.0:
            return {'volume_sentiment': 0.0, 'volume_trend': 'unavailable',
                    'current_volume_ratio': 1.0, 'price_volume_alignment': 'neutral'}

        volume = btc_data['Volume'].astype(float)
        close = btc_data['Close'].astype(float)

        vol_ma10 = volume.rolling(10).mean()
        current_volume = float(volume.iloc[-1])
        avg_volume = float(vol_ma10.iloc[-1]) if not vol_ma10.empty and not np.isnan(vol_ma10.iloc[-1]) else current_volume
        vol_ratio = current_volume / (avg_volume + 1e-9)

        price_chg = (close.iloc[-1] - close.iloc[-2]) / (close.iloc[-2] + 1e-9) if len(close) >= 2 else 0.0

        if vol_ratio > 2.0 and price_chg > 0.02:
            vol_sent = 0.6
        elif vol_ratio > 2.0 and price_chg < -0.02:
            vol_sent = -0.6
        elif vol_ratio > 1.3:
            vol_sent = 0.2 if price_chg > 0 else -0.2
        else:
            vol_sent = 0.0

        vol_trend = 'high' if vol_ratio > 1.5 else ('normal' if vol_ratio > 0.8 else 'low')
        align = 'positive' if (price_chg > 0 and vol_ratio > 1.2) else ('negative' if (price_chg < 0 and vol_ratio > 1.2) else 'neutral')

        return {
            'current_volume_ratio': float(vol_ratio),
            'volume_sentiment': _safe_clip(vol_sent),
            'volume_trend': vol_trend,
            'price_volume_alignment': align
        }

    def calculate_trend_persistence(self, btc_data: pd.DataFrame) -> Dict:
        """Consecutive up/down day persistence with sensible thresholds."""
        closes = btc_data['Close'].astype(float)
        if len(closes) < 10:
            return {'persistence_sentiment': 0.0, 'trend_quality': 'insufficient_data',
                    'consecutive_positive_days': 0, 'consecutive_negative_days': 0}

        changes = closes.pct_change().dropna()
        recent = changes.tail(7)  # last week

        # Count consecutive move from end (with 1% threshold)
        pos_cnt = 0
        neg_cnt = 0

        # First step: determine direction of last "significant" day
        direction = 0
        if abs(recent.iloc[-1]) > 0.01:
            direction = 1 if recent.iloc[-1] > 0 else -1

        if direction == 1:
            pos_cnt = 1
            for ch in reversed(recent.iloc[:-1]):
                if ch > 0:
                    pos_cnt += 1
                else:
                    break
        elif direction == -1:
            neg_cnt = 1
            for ch in reversed(recent.iloc[:-1]):
                if ch < 0:
                    neg_cnt += 1
                else:
                    break

        if pos_cnt >= 3:
            sent = 0.4
            qual = 'strong_upward_persistence'
        elif neg_cnt >= 3:
            sent = -0.4
            qual = 'strong_downward_persistence'
        elif pos_cnt == 2:
            sent = 0.2
            qual = 'moderate_upward_persistence'
        elif neg_cnt == 2:
            sent = -0.2
            qual = 'moderate_downward_persistence'
        else:
            sent = 0.0
            qual = 'no_clear_persistence'

        return {
            'consecutive_positive_days': int(pos_cnt),
            'consecutive_negative_days': int(neg_cnt),
            'persistence_sentiment': _safe_clip(sent),
            'trend_quality': qual
        }

    def calculate_bitcoin_micro_momentum_composite(self, momentum_analyses: Dict) -> Dict:
        """Weighted composite for micro-momentum block."""
        w = self.config['momentum_weights']

        short_mom = float(momentum_analyses.get('short_term_momentum', {}).get('momentum_sentiment', 0.0))
        vol_sent  = float(momentum_analyses.get('volatility_analysis', {}).get('volatility_sentiment', 0.0))
        vol_mom   = float(momentum_analyses.get('volume_analysis', {}).get('volume_sentiment', 0.0))
        persist   = float(momentum_analyses.get('trend_persistence', {}).get('persistence_sentiment', 0.0))

        composite = (
            short_mom * w['short_term_momentum'] +
            vol_sent  * w['intraday_volatility'] +
            vol_mom   * w['volume_momentum'] +
            persist   * w['trend_persistence']
        )
        composite = _safe_clip(composite)

        # Confidence: agreement among signals → higher confidence
        signals = [short_mom, vol_sent, vol_mom, persist]
        pos = sum(1 for s in signals if s > 0.10)
        neg = sum(1 for s in signals if s < -0.10)
        tot = len(signals)

        if pos >= 0.75 * tot or neg >= 0.75 * tot:
            conf = 0.85
        elif pos >= 0.5 * tot or neg >= 0.5 * tot:
            conf = 0.75
        else:
            conf = 0.65

        return {
            'composite_score': composite,
            'confidence': float(conf),
            'component_scores': {
                'short_term_momentum': short_mom,
                'volatility_pattern': vol_sent,
                'volume_momentum': vol_mom,
                'trend_persistence': persist
            }
        }

    # ===== MAIN =====
    async def analyze_btc_micro_momentum(self, backtest_date: str = None) -> Dict:
        """
        Main Bitcoin micro momentum analysis with strict historical cutoff.
        """
        print(f"₿ Analyzing Bitcoin Micro Momentum for {backtest_date or 'current'}...")
        t0 = datetime.now()

        try:
            btc_data = await self.fetch_bitcoin_data(backtest_date)

            momentum_analyses = {
                'short_term_momentum': self.calculate_short_term_momentum(btc_data),
                'volatility_analysis': self.calculate_intraday_volatility_pattern(btc_data),
                'volume_analysis': self.calculate_volume_momentum(btc_data),
                'trend_persistence': self.calculate_trend_persistence(btc_data)
            }

            sentiment = self.calculate_bitcoin_micro_momentum_composite(momentum_analyses)
            elapsed = (datetime.now() - t0).total_seconds()

            result = {
                'component_sentiment': sentiment['composite_score'],
                'component_confidence': sentiment['confidence'],
                'component_weight': 0.35,  # 35% of BTC composite
                'weighted_contribution': sentiment['composite_score'] * 0.35,
                'micro_momentum_analysis': momentum_analyses,
                'market_context': {
                    'current_price': float(btc_data['Close'].astype(float).iloc[-1]),
                    'daily_change_pct': float(((btc_data['Close'].iloc[-1] - btc_data['Close'].iloc[-2]) /
                                               (btc_data['Close'].iloc[-2] + 1e-9) * 100)) if len(btc_data) >= 2 else 0.0,
                    'data_points': int(len(btc_data)),
                    'lookback_period_days': int(self.config['lookback_days']),
                    'note': 'Daily data proxy for micro momentum (intraday not available historically)',
                    'backtest_date': backtest_date or 'real_time'
                },
                'component_scores': sentiment['component_scores'],
                'momentum_summary': self._create_momentum_summary(momentum_analyses),
                'execution_time_seconds': elapsed,
                'kpi_name': 'Bitcoin_Micro_Momentum',
                'analysis_timestamp': datetime.now().isoformat()
            }

            print(f"✅ Bitcoin Micro Momentum Analysis completed: {result['component_sentiment']:.3f}")
            print(f"₿ {result['momentum_summary']}")
            return result

        except Exception as e:
            print(f"❌ Bitcoin micro momentum analysis failed: {e}")
            return self._create_error_result(f"BTC momentum exception: {e}")

    def _create_momentum_summary(self, momentum_analyses: Dict) -> str:
        short_m = momentum_analyses.get('short_term_momentum', {})
        vol     = momentum_analyses.get('volatility_analysis', {})
        persist = momentum_analyses.get('trend_persistence', {})

        direction = short_m.get('trend_direction', 'neutral')
        strength  = short_m.get('trend_strength', 'weak')
        pattern   = vol.get('pattern', 'unknown')
        quality   = persist.get('trend_quality', 'unknown')

        if direction == 'bullish' and strength == 'strong':
            return "🟢 Strong bullish micro momentum" + (" with trend persistence" if 'upward_persistence' in quality else "")
        if direction == 'bearish' and strength == 'strong':
            return "🔴 Strong bearish micro momentum" + (" with trend persistence" if 'downward_persistence' in quality else "")
        if 'expanding' in pattern and direction == 'bullish':
            return "🟡 Bullish momentum with expanding volatility"
        if 'expanding' in pattern and direction == 'bearish':
            return "🟡 Bearish momentum with expanding volatility"
        return "🟡 Mixed micro momentum signals"

    def _create_error_result(self, error_message: str) -> Dict:
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.5,
            'component_weight': 0.175,  # halve on error
            'error': error_message,
            'kpi_name': 'Bitcoin_Micro_Momentum',
            'analysis_timestamp': datetime.now().isoformat()
        }


# Optional ad-hoc test
async def test_btc_micro_momentum_analyzer():
    analyzer = BitcoinMicroMomentumAnalyzer(historical_cutoff="2022-12-31")
    for date in ["2023-01-31", "2023-06-30", "2023-12-31"]:
        result = await analyzer.analyze_btc_micro_momentum(date)
        print(f"{date}  sentiment={result.get('component_sentiment'):.3f}  "
              f"conf={result.get('component_confidence'):.2f}  points={result.get('market_context',{}).get('data_points')}")

if __name__ == "__main__":
    asyncio.run(test_btc_micro_momentum_analyzer())
