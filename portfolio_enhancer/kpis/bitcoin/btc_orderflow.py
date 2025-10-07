# btc_orderflow.py
# HISTORICAL VERSION for backtesting - Bitcoin Orderflow CVD Analysis
# Updated: strict historical cutoff, EWMA smoothing, safer math

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
    if series is None or series.empty:
        return series
    return series.ewm(halflife=halflife_days, min_periods=5, adjust=False).mean()


def _safe_clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        return float(np.clip(x, lo, hi))
    except Exception:
        return 0.0


class BitcoinOrderflowAnalyzer:
    """
    HISTORICAL VERSION - Bitcoin Orderflow CVD Analysis (proxy)
    - Strict backtest: clamp all data to <= historical_cutoff
    - 35% weight in BTC composite
    """

    def __init__(self, config: Dict = None, historical_cutoff: Optional[str] = "2022-12-31"):
        self.config = config or self._default_config()
        self.historical_cutoff = _to_ts(historical_cutoff)
        print("Bitcoin Orderflow Analyzer (HISTORICAL) initialized")
        print("⚠️ Using volume proxy - real orderflow data unavailable historically")
        if self.historical_cutoff is not None:
            print(f"🔒 Historical cutoff set to {self.historical_cutoff.date()}")

    def _default_config(self) -> Dict:
        return {
            'lookback_days': 45,
            'ewma_halflife_days': 365.0,
            'bitcoin_tickers': ['BTC-USD', 'GBTC'],  # prefer BTC-USD, fallback GBTC
            'orderflow_weights': {
                'volume_delta_proxy': 0.50,
                'price_volume_momentum': 0.30,
                'buying_pressure_proxy': 0.20
            }
        }

    # ===== DATA FETCH (with strict cutoff) =====
    async def fetch_bitcoin_data(self, backtest_date: str = None) -> pd.DataFrame:
        """Fetch BTC series with no-lookahead."""
        bt_ts = _to_ts(backtest_date)
        if self.historical_cutoff is not None:
            if bt_ts is None:
                effective_end = self.historical_cutoff
            else:
                effective_end = min(bt_ts, self.historical_cutoff)
        else:
            effective_end = bt_ts or pd.Timestamp.utcnow().tz_localize(None)

        lookback = int(self.config['lookback_days'])
        effective_start = effective_end - pd.Timedelta(days=lookback)

        print(f"₿ Fetching Bitcoin orderflow data | "
              f"range: {effective_start.date()} → {effective_end.date()} "
              f"(requested={backtest_date or 'current'}, cutoff={self.historical_cutoff.date() if self.historical_cutoff else 'none'})")

        for ticker in self.config['bitcoin_tickers']:
            try:
                hist = yf.Ticker(ticker).history(
                    start=effective_start.strftime('%Y-%m-%d'),
                    end=(effective_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                    interval="1d"
                )
                hist = hist.tz_localize(None)
                if not hist.empty and len(hist) >= 10:
                    print(f"✅ Fetched {len(hist)} rows from {ticker}")
                    return hist
                else:
                    print(f"⚠️ No data returned for {ticker}")
            except Exception as e:
                print(f"❌ {ticker}: {e}")

        raise ValueError("Unable to fetch Bitcoin data for orderflow proxy")

    # ===== COMPONENTS =====
    def calculate_volume_delta_proxy(self, btc_data: pd.DataFrame) -> Dict:
        """Proxy for CVD using up/down volume; smoothed."""
        if 'Volume' not in btc_data.columns or btc_data['Volume'].sum() == 0:
            return {'volume_delta_sentiment': 0.0, 'trend': 'no_volume_data'}

        close = btc_data['Close'].astype(float)
        vol = btc_data['Volume'].astype(float)

        price_chg = close.pct_change().fillna(0.0)
        up_vol = vol.where(price_chg > 0, 0.0)
        dn_vol = vol.where(price_chg < 0, 0.0)

        cvd = (up_vol - dn_vol).cumsum()
        cvd = _ewma(cvd, self.config['ewma_halflife_days'])

        if cvd is None or cvd.empty:
            return {'volume_delta_sentiment': 0.0, 'trend': 'insufficient_data'}

        # 1-week change relative to scale
        if len(cvd) >= 8 and abs(cvd.iloc[-8]) > 0:
            cvd_trend = float((cvd.iloc[-1] - cvd.iloc[-8]) / (abs(cvd.iloc[-8]) + 1e-9))
        else:
            cvd_trend = 0.0

        if cvd_trend > 0.1:
            sent = +0.5
        elif cvd_trend > 0.05:
            sent = +0.2
        elif cvd_trend < -0.1:
            sent = -0.5
        elif cvd_trend < -0.05:
            sent = -0.2
        else:
            sent = 0.0

        recent_buy = float(up_vol.tail(7).sum())
        recent_dn  = float(dn_vol.tail(7).sum())
        denom = recent_buy + recent_dn
        ratio = float(recent_buy / denom) if denom > 0 else 0.5

        return {
            'cvd_trend': cvd_trend,
            'volume_delta_sentiment': _safe_clip(sent),
            'recent_buying_ratio': ratio
        }

    def calculate_price_volume_momentum(self, btc_data: pd.DataFrame) -> Dict:
        """Combine price and volume momentum; smoothed."""
        if 'Volume' not in btc_data.columns:
            return {'pv_momentum_sentiment': 0.0}

        close = btc_data['Close'].astype(float)
        vol = btc_data['Volume'].astype(float)

        if len(close) >= 8:
            pm7 = float((close.iloc[-1] - close.iloc[-8]) / (close.iloc[-8] + 1e-9))
        else:
            pm7 = 0.0

        vol_ma = vol.rolling(14).mean()
        vratio = float(vol.iloc[-1] / (vol_ma.iloc[-1] + 1e-9)) if len(vol_ma) and vol_ma.iloc[-1] > 0 else 1.0

        # Smooth price-change series a bit
        pm_series = close.pct_change(7).fillna(0.0)
        pm_smooth = _ewma(pm_series, self.config['ewma_halflife_days'])
        if not pm_smooth.empty and not np.isnan(pm_smooth.iloc[-1]):
            pm7 = 0.5 * pm7 + 0.5 * float(pm_smooth.iloc[-1])

        if pm7 > 0.05 and vratio > 1.3:
            sent = +0.6
        elif pm7 < -0.05 and vratio > 1.3:
            sent = -0.6
        elif abs(pm7) > 0.02 and vratio > 1.1:
            sent = +0.3 if pm7 > 0 else -0.3
        else:
            sent = 0.0

        return {
            'price_momentum_7d_pct': pm7 * 100.0,
            'volume_momentum_ratio': vratio,
            'pv_momentum_sentiment': _safe_clip(sent)
        }

    def calculate_buying_pressure_proxy(self, btc_data: pd.DataFrame) -> Dict:
        """Close-in-range buying pressure; smoothed via rolling mean."""
        hi = btc_data['High'].astype(float)
        lo = btc_data['Low'].astype(float)
        cl = btc_data['Close'].astype(float)

        rng = (hi - lo).replace(0, np.nan)
        pos = ((cl - lo) / rng).clip(0.0, 1.0).fillna(0.5)
        # Smooth 14d
        avg_pos = float(pos.rolling(14).mean().iloc[-1]) if len(pos) >= 14 else 0.5

        if avg_pos > 0.7:
            sent = +0.4
            reg = 'strong_buying'
        elif avg_pos > 0.6:
            sent = +0.2
            reg = 'buying'
        elif avg_pos < 0.3:
            sent = -0.4
            reg = 'strong_selling'
        elif avg_pos < 0.4:
            sent = -0.2
            reg = 'selling'
        else:
            sent = 0.0
            reg = 'neutral'

        return {
            'avg_daily_position': avg_pos,
            'buying_pressure_sentiment': _safe_clip(sent),
            'pressure_regime': reg
        }

    def calculate_orderflow_composite(self, orderflow_analyses: Dict) -> Dict:
        """Weighted composite for orderflow proxy."""
        w = self.config['orderflow_weights']

        vol_delta = float(orderflow_analyses.get('volume_delta', {}).get('volume_delta_sentiment', 0.0))
        pvm      = float(orderflow_analyses.get('pv_momentum', {}).get('pv_momentum_sentiment', 0.0))
        buy_pr   = float(orderflow_analyses.get('buying_pressure', {}).get('buying_pressure_sentiment', 0.0))

        composite = vol_delta * w['volume_delta_proxy'] + pvm * w['price_volume_momentum'] + buy_pr * w['buying_pressure_proxy']
        composite = _safe_clip(composite)

        # Fixed moderate confidence for proxy approach
        confidence = 0.70

        return {
            'composite_score': composite,
            'confidence': confidence,
            'component_scores': {
                'volume_delta_proxy': vol_delta,
                'price_volume_momentum': pvm,
                'buying_pressure_proxy': buy_pr
            }
        }

    # ===== MAIN =====
    async def analyze_btc_orderflow(self, backtest_date: str = None) -> Dict:
        print(f"₿ Analyzing Bitcoin Orderflow for {backtest_date or 'current'}...")
        t0 = datetime.now()
        try:
            btc = await self.fetch_bitcoin_data(backtest_date)

            analyses = {
                'volume_delta': self.calculate_volume_delta_proxy(btc),
                'pv_momentum': self.calculate_price_volume_momentum(btc),
                'buying_pressure': self.calculate_buying_pressure_proxy(btc),
            }
            sentiment = self.calculate_orderflow_composite(analyses)
            elapsed = (datetime.now() - t0).total_seconds()

            result = {
                'component_sentiment': sentiment['composite_score'],
                'component_confidence': sentiment['confidence'],
                'component_weight': 0.35,
                'weighted_contribution': sentiment['composite_score'] * 0.35,
                'orderflow_analysis': analyses,
                'market_context': {
                    'current_price': float(btc['Close'].astype(float).iloc[-1]),
                    'data_points': int(len(btc)),
                    'analysis_method': 'volume_proxy_orderflow',
                    'backtest_date': backtest_date or 'real_time'
                },
                'component_scores': sentiment['component_scores'],
                'execution_time_seconds': elapsed,
                'kpi_name': 'Bitcoin_Orderflow_CVD',
                'analysis_timestamp': datetime.now().isoformat()
            }
            print(f"✅ Bitcoin Orderflow Analysis completed: {result['component_sentiment']:.3f}")
            return result

        except Exception as e:
            print(f"❌ Bitcoin orderflow analysis failed: {e}")
            return {
                'component_sentiment': 0.0,
                'component_confidence': 0.4,
                'component_weight': 0.175,  # halve on error
                'error': str(e),
                'kpi_name': 'Bitcoin_Orderflow_CVD',
                'analysis_timestamp': datetime.now().isoformat()
            }


# Optional ad-hoc test
async def test_btc_orderflow_analyzer():
    analyzer = BitcoinOrderflowAnalyzer(historical_cutoff="2022-12-31")
    for date in ["2023-01-31", "2023-06-30", "2023-12-31"]:
        out = await analyzer.analyze_btc_orderflow(date)
        print(date, out.get("component_sentiment"))

if __name__ == "__main__":
    asyncio.run(test_btc_orderflow_analyzer())
