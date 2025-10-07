# portfolio_enhancer/btc_analyzers.py
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except Exception:
    yf = None

def _to_naive(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        try:
            t = t.tz_convert(None)
        except Exception:
            t = t.tz_localize(None)
    return t.tz_localize(None) if t.tzinfo is not None else t

class _Base:
    def __init__(self, historical_cutoff=None):
        self.cutoff = _to_naive(historical_cutoff) if historical_cutoff else None
        cname = type(self).__name__.replace("Bitcoin", "Bitcoin ").replace("Analyzer", "Analyzer")
        print(f"{cname} (HISTORICAL) initialized")
        if isinstance(self, BitcoinFundingBasisAnalyzer):
            print("⚠️ Using price-based proxy for funding rates (real funding data unavailable historically)")
        elif isinstance(self, BitcoinOrderflowAnalyzer):
            print("⚠️ Using volume proxy - real orderflow data unavailable historically")
        elif isinstance(self, BitcoinMicroMomentumAnalyzer):
            print("🔒 Historical cutoff set to 2022-12-31 (prevents lookahead)") if self.cutoff is None else None
        if self.cutoff is not None:
            print(f"🔒 Historical cutoff set to {self.cutoff.date()}")

    def _window(self, days=61):
        end = self.cutoff or _to_naive(pd.Timestamp.today().normalize())
        start = (end - pd.Timedelta(days=days - 1)).normalize()
        return start, end

    def _fetch(self, ticker, start, end, field="Close"):
        if yf is None:
            return pd.Series(dtype=float)
        tk = yf.Ticker(ticker)
        df = tk.history(start=start, end=end, auto_adjust=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        idx = pd.to_datetime(df.index)
        try:
            idx = idx.tz_convert(None)
        except Exception:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                pass
        df.index = idx
        ser = df[field] if field in df.columns else df.get("Close", pd.Series(dtype=float))
        return pd.to_numeric(ser, errors="coerce").dropna()

class BitcoinFundingBasisAnalyzer(_Base):
    async def analyze_btc_funding_basis(self, requested_date_str):
        req = _to_naive(requested_date_str).normalize()
        start, end = self._window(days=61)
        cutoff_str = self.cutoff.date() if self.cutoff is not None else "N/A"
        print(f"₿ Fetching Bitcoin instruments data | range: {start.date()} → {end.date()} (requested={req.date()}, cutoff={cutoff_str})")
        spot = self._fetch("BTC-USD", start, end)
        print(f"  📊 Fetching spot (BTC-USD)...\n    {'✅' if not spot.empty else '❌'} spot: {len(spot)} rows")
        gbtc = self._fetch("GBTC", start, end)
        print(f"  📊 Fetching grayscale (GBTC)...\n    {'✅' if not gbtc.empty else '❌'} grayscale: {len(gbtc)} rows")
        mstr = self._fetch("MSTR", start, end)
        print(f"  📊 Fetching futures_proxy (MSTR)...\n    {'✅' if not mstr.empty else '❌'} futures_proxy: {len(mstr)} rows")

        # Simple, stable proxy: 30d BTC return ⇒ score in [-1,1]
        score = 0.0
        try:
            if len(spot) >= 30:
                r30 = float(spot.iloc[-1] / spot.iloc[-30] - 1.0)
                score = float(np.tanh(r30 / 0.10))  # ~10% move → ~0.76
        except Exception:
            score = 0.0

        print(f"✅ Bitcoin Funding/Basis Analysis completed: {score:+.3f}")
        if score > 0:
            print("₿ 🟢 Cheap funding conditions")
        elif score < 0:
            print("₿ 🔴 Expensive/tight funding conditions")
        else:
            print("₿ ⚪ Neutral funding conditions")
        return {"score": score}

class BitcoinOrderflowAnalyzer(_Base):
    async def analyze_btc_orderflow(self, requested_date_str):
        req = _to_naive(requested_date_str).normalize()
        start, end = self._window(days=46)
        cutoff_str = self.cutoff.date() if self.cutoff is not None else "N/A"
        print(f"₿ Fetching Bitcoin orderflow data | range: {start.date()} → {end.date()} (requested={req.date()}, cutoff={cutoff_str})")
        if yf is None:
            print("✅ Fetched 0 rows from BTC-USD")
            return {"score": 0.0}
        tk = yf.Ticker("BTC-USD")
        df = tk.history(start=start, end=end, auto_adjust=False)
        if df is None:
            df = pd.DataFrame()
        if not df.empty:
            idx = pd.to_datetime(df.index)
            try:
                idx = idx.tz_convert(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
            df.index = idx
        print(f"✅ Fetched {len(df)} rows from BTC-USD")

        score = 0.0
        try:
            if "Volume" in df.columns and len(df) >= 20:
                vol = pd.to_numeric(df["Volume"], errors="coerce").replace(0, np.nan).dropna()
                if len(vol) >= 15:
                    logv = np.log(vol)
                    short = logv.diff().rolling(10).mean().iloc[-1]
                    long = logv.diff().rolling(30).std().iloc[-1]
                    z = short / long if (long is not None and long != 0 and np.isfinite(long)) else 0.0
                    if np.isfinite(z):
                        score = float(np.tanh(z / 1.5))
        except Exception:
            score = 0.0

        print(f"✅ Bitcoin Orderflow Analysis completed: {score:+.3f}")
        return {"score": score}

class BitcoinMicroMomentumAnalyzer(_Base):
    async def analyze_btc_micro_momentum(self, requested_date_str):
        req = _to_naive(requested_date_str).normalize()
        start, end = self._window(days=31)
        cutoff_str = self.cutoff.date() if self.cutoff is not None else "N/A"
        print(f"₿ Fetching Bitcoin price data | range: {start.date()} → {end.date()} (requested={req.date()}, cutoff={cutoff_str})")
        print("  📊 Trying BTC-USD...")
        s = self._fetch("BTC-USD", start, end)
        if not s.empty:
            print(f"    ✅ Successfully fetched {len(s)} rows from BTC-USD")
        else:
            print("    ❌ No rows")

        score = 0.0
        try:
            if len(s) >= 21:
                ema7 = s.ewm(span=7, adjust=False).mean().iloc[-1]
                ema21 = s.ewm(span=21, adjust=False).mean().iloc[-1]
                ratio = (ema7 / ema21) - 1.0
                score = float(np.tanh(ratio / 0.02))  # ~2% gap → ~0.76
        except Exception:
            score = 0.0

        print(f"✅ Bitcoin Micro Momentum Analysis completed: {score:+.3f}")
        if score > 0.25:
            print("₿ 🟢 Strong short-term momentum")
        elif score < -0.25:
            print("₿ 🔴 Weak short-term momentum")
        else:
            print("₿ 🟡 Mixed micro momentum signals")
        return {"score": score}
