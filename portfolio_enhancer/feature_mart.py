# -*- coding: utf-8 -*-
# portfolio_enhancer/feature_mart.py
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import yfinance as yf

# ========= Core assets (same as backtester) =========
ASSETS = {
    "Equities": "^NSEI",   # Nifty 50
    "Gold":     "GLD",     # SPDR Gold Shares
    "REITs":    "VNQ",     # US REIT proxy
    "Bitcoin":  "BTC-USD",
}

# ========= Macros / Aux series =========
# We try primary first, then fallback if needed.
AUX_PRIMARY = {
    "VIX":   "^VIX",
    "VIX3M": "^VIX3M",   # VIX 3-month (term structure)
    "TNX":   "^TNX",     # US 10y yield %
    "DXY":   "DX-Y.NYB", # US Dollar index (Yahoo alt)
    "FVX":   "^FVX",     # 5Y yield
    "TYX":   "^TYX",     # 30Y yield
}
AUX_FALLBACK = {
    "DXY": "DXY"  # if DX-Y.NYB fails
}

# ========= KPI modules (composite only) =========
from .kpis.nifty import analyze_nifty_composite
from .kpis.gold.gold_kpi_composite import gold_kpi_composite
from .kpis.reit import reit_kpi_composite
from .kpis.bitcoin.btc_sentiment import analyze_btc_sentiment

# ========= Small helpers =========
def _to_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_localize(None) if x.tzinfo is not None else x
    return pd.Timestamp(x).tz_localize(None)

def _yf_hist(ticker: str, start: str, end: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)), auto_adjust=True)
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    return df

def _pct_chg(s: pd.Series, periods: int) -> pd.Series:
    return (s / s.shift(periods) - 1.0)

def _safe_last(s: Optional[pd.Series], asof: pd.Timestamp) -> Optional[float]:
    if s is None or s.empty: return None
    sub = s.loc[:asof].dropna()
    return float(sub.iloc[-1]) if len(sub) else None

def _classify_regime(vix_val: Optional[float]) -> str:
    if vix_val is None or np.isnan(vix_val): return "neutral"
    if vix_val >= 25.0: return "risk_off"
    if vix_val <= 16.0: return "risk_on"
    return "neutral"

async def _silent(factory, timeout: float = 60.0) -> dict:
    """Run a KPI function while silencing prints; return dict or neutral."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = factory()
        if asyncio.iscoroutine(res):
            with contextlib.redirect_stdout(buf):
                res = await asyncio.wait_for(res, timeout=timeout)
        if isinstance(res, dict): return res
        return {"composite_sentiment": float(res) if res is not None else 0.0}
    except Exception:
        return {"composite_sentiment": 0.0}

async def kpi_for_asset(asset: str, asof: str, H: str = "M") -> float:
    if asset == "Equities":
        for fac in (lambda: analyze_nifty_composite(asof, H),
                    lambda: analyze_nifty_composite(asof)):
            out = await _silent(fac)
            if out is not None: return float(out.get("composite_sentiment", 0.0) or 0.0)
        return 0.0
    if asset == "Gold":
        for fac in (lambda: gold_kpi_composite(asof, H),
                    lambda: gold_kpi_composite(asof)):
            out = await _silent(fac)
            if out is not None: return float(out.get("composite_sentiment", 0.0) or 0.0)
        return 0.0
    if asset == "REITs":
        out = await _silent(lambda: reit_kpi_composite(asof, H))
        return float(out.get("composite_sentiment", 0.0) or 0.0)
    if asset == "Bitcoin":
        out = await _silent(lambda: analyze_btc_sentiment(backtest_date=asof, horizon=H, historical_cutoff=asof))
        return float(out.get("composite_sentiment", 0.0) or 0.0)
    return 0.0

# ========= Technical indicators =========
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def bollinger_position(series: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + k * sd
    lower = ma - k * sd
    rng = (upper - lower)
    pos = (series - lower) / rng
    return pos.clip(0, 1)

def drawdown(series: pd.Series) -> pd.Series:
    cummax = series.cummax()
    return (series / cummax) - 1.0

def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    mu = returns.rolling(window).mean()
    sig = returns.rolling(window).std()
    return (mu / sig.replace(0, np.nan)) * np.sqrt(252.0)

# ========= Period building =========
def month_ends(start: str, end: str) -> List[pd.Timestamp]:
    idx = pd.date_range(_to_ts(start), _to_ts(end), freq="M")
    return [d.tz_localize(None) for d in idx]

def week_ends(start: str, end: str) -> List[pd.Timestamp]:
    idx = pd.date_range(_to_ts(start), _to_ts(end), freq="W-FRI")
    return [d.tz_localize(None) for d in idx]

def pairs_from_period_ends(pends: List[pd.Timestamp]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for i in range(1, len(pends)):
        decision = pends[i-1]
        period_end = pends[i]
        out.append((decision, period_end))
    return out

def realized_forward(series_ret: pd.Series, start_excl: pd.Timestamp, end_incl: pd.Timestamp) -> float:
    sub = series_ret.loc[(series_ret.index > start_excl) & (series_ret.index <= end_incl)]
    if sub.empty: return 0.0
    return float((1.0 + sub.fillna(0.0)).prod() - 1.0)

# ========= Feature building =========
@dataclass
class Config:
    start: str
    end: str
    freqs: List[str]  # ["monthly"] or ["monthly","weekly"]
    include_kpis: bool = True
    out: str = "Training_Panel.csv"
    to_parquet: bool = False

def load_prices_and_returns(start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    close = {}
    for a, tk in ASSETS.items():
        df = _yf_hist(tk, start, end)
        if df.empty:
            raise RuntimeError(f"No data for {a} ({tk}) between {start} and {end}")
        close[a] = df["Close"].astype(float)
    px = pd.DataFrame(close).sort_index()
    rets = px.pct_change().rename(columns=lambda c: f"{c}_ret")
    return px, rets

def load_aux(start: str, end: str) -> pd.DataFrame:
    out = {}
    for name, tk in AUX_PRIMARY.items():
        try:
            df = _yf_hist(tk, start, end)
            if not df.empty:
                out[name] = df["Close"].astype(float)
            else:
                raise RuntimeError("empty")
        except Exception:
            fb = AUX_FALLBACK.get(name)
            if fb:
                try:
                    df = _yf_hist(fb, start, end)
                    if not df.empty: out[name] = df["Close"].astype(float)
                except Exception:
                    pass
    aux = pd.DataFrame(out).sort_index()
    # simple changes
    for c in aux.columns:
        aux[f"{c}_chg1"] = aux[c].pct_change()
        aux[f"{c}_chg5"] = aux[c].pct_change(5)
    # term structure and yield curve slope if available
    if "VIX" in aux and "VIX3M" in aux:
        aux["vix_ts_ratio"] = aux["VIX"] / aux["VIX3M"].replace(0, np.nan)
    if "TYX" in aux and "FVX" in aux:
        aux["yc_slope"] = aux["TYX"] - aux["FVX"]
    return aux

def build_single_asset_features(px: pd.Series, ret: pd.Series) -> pd.DataFrame:
    """Daily indicator panel for ONE asset."""
    df = pd.DataFrame({"price": px, "ret": ret})
    # Momentum (price-based)
    for k in [5, 10, 21, 63, 126, 252]:
        df[f"mom_{k}"] = _pct_chg(df["price"], k)
    # simple lag returns
    for k in [1, 5, 21]:
        df[f"ret_{k}"] = df["ret"].rolling(k).apply(lambda x: (1+x).prod()-1, raw=True)
    # Realized vol
    for k in [21, 63, 126]:
        df[f"vol_{k}"] = df["ret"].rolling(k).std() * np.sqrt(252.0)
    # RSI, MACD, Bollinger
    df["rsi_14"] = rsi(df["price"], 14)
    macd_line, macd_sig, macd_hist = macd(df["price"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist
    df["bb_pos"] = bollinger_position(df["price"], 20, 2.0)
    # Drawdown & rolling sharpe
    df["cum"] = (1.0 + df["ret"].fillna(0)).cumprod()
    df["dd_raw"] = drawdown(df["cum"])
    df["dd_21max"] = df["dd_raw"].rolling(21).min()
    df["roll_sharpe_63"] = rolling_sharpe(df["ret"], 63)
    df = df.drop(columns=["cum"])
    return df

def feature_at_date(F: pd.DataFrame, dt: pd.Timestamp, cols: Iterable[str]) -> Dict[str, float]:
    """Take last available values on/before dt for requested cols."""
    out = {}
    snap = F.loc[:dt].iloc[-1:] if not F.loc[:dt].empty else pd.DataFrame(columns=F.columns)
    for c in cols:
        v = np.nan
        if not snap.empty and c in F.columns:
            v = snap.iloc[0][c]
        out[c] = float(v) if pd.notna(v) else np.nan
    return out

async def build_panel(config: Config) -> pd.DataFrame:
    px, rets = load_prices_and_returns(config.start, config.end)
    aux = load_aux(config.start, config.end)

    # Precompute daily features per asset
    per_asset_features: Dict[str, pd.DataFrame] = {}
    feat_cols: Dict[str, List[str]] = {}
    for a in ASSETS.keys():
        F = build_single_asset_features(px[a], rets[f"{a}_ret"])
        per_asset_features[a] = F
        cols = [c for c in F.columns if c not in ("price","ret")]
        feat_cols[a] = cols

    # Decision calendars
    decisions: List[Tuple[str, List[Tuple[pd.Timestamp,pd.Timestamp]]]] = []
    if "monthly" in [f.lower() for f in config.freqs]:
        me = month_ends(config.start, config.end)
        decisions.append(("M", pairs_from_period_ends(me)))
    if "weekly" in [f.lower() for f in config.freqs]:
        we = week_ends(config.start, config.end)
        decisions.append(("W", pairs_from_period_ends(we)))

    rows: List[Dict] = []
    for H, pairs in decisions:
        for dd, pe in pairs:
            # Regime & macro snapshot
            vix_val = _safe_last(aux.get("VIX"), dd)
            regime  = _classify_regime(vix_val)
            macro_fields = ["VIX","VIX3M","vix_ts_ratio","TNX","DXY","FVX","TYX","yc_slope",
                            "VIX_chg1","VIX_chg5","TNX_chg1","TNX_chg5","DXY_chg1","DXY_chg5"]
            # ensure change cols exist
            for base in ["VIX","TNX","DXY"]:
                if f"{base}_chg1" not in aux.columns and base in aux.columns:
                    aux[f"{base}_chg1"] = aux[base].pct_change()
                if f"{base}_chg5" not in aux.columns and base in aux.columns:
                    aux[f"{base}_chg5"] = aux[base].pct_change(5)

            macro_vals = feature_at_date(aux, dd, [c for c in macro_fields if c in aux.columns])

            # KPI (optional) — expensive, so do per asset and cache if needed
            kpi_cache: Dict[str, float] = {}
            if config.include_kpis:
                asof = dd.date().isoformat()
                for a in ASSETS.keys():
                    kpi_cache[a] = await kpi_for_asset(a, asof, H)
            else:
                for a in ASSETS.keys(): kpi_cache[a] = np.nan

            # Asset rows
            for a in ASSETS.keys():
                F = per_asset_features[a]
                cols = feat_cols[a]
                aset = feature_at_date(F, dd, cols)

                # forward target
                fwd = realized_forward(rets[f"{a}_ret"], dd, pe)

                row = {
                    "date_decision": dd.date().isoformat(),
                    "date_period_end": pe.date().isoformat(),
                    "horizon": H,
                    "asset": a,
                    "target_fwd_ret": fwd,
                    "regime": regime,
                    "kpi_composite": float(kpi_cache.get(a, np.nan)),
                }
                # flatten features with asset prefix
                for k, v in aset.items():
                    row[f"{a}.{k}"] = v
                # add macro snapshot
                for k, v in macro_vals.items():
                    row[k] = v
                rows.append(row)

    df = pd.DataFrame(rows).sort_values(["date_decision","asset"]).reset_index(drop=True)
    return df

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a large, leak-free feature/target panel for ML training.")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str,   required=True)
    p.add_argument("--freq", nargs="+", choices=["monthly","weekly"], default=["monthly"],
                   help="Decision frequency (monthly and/or weekly).")
    p.add_argument("--skip-kpis", action="store_true", help="Skip KPI composites to speed up.")
    p.add_argument("--out", type=str, default="Training_Panel.csv")
    p.add_argument("--parquet", action="store_true", help="Write Parquet instead of CSV.")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        start=args.start, end=args.end,
        freqs=args.freq,
        include_kpis=not args.skip_kpis,
        out=args.out,
        to_parquet=args.parquet
    )
    df = asyncio.run(build_panel(cfg))
    if cfg.to_parquet:
        df.to_parquet(cfg.out, index=False)
    else:
        df.to_csv(cfg.out, index=False)
    # quick summary
    n_months = (df.loc[df["horizon"]=="M","date_decision"].nunique()) if "horizon" in df.columns else 0
    n_weeks  = (df.loc[df["horizon"]=="W","date_decision"].nunique()) if "horizon" in df.columns else 0
    print(f"✅ Wrote {len(df):,} rows to {cfg.out}  "
          f"(assets={len(ASSETS)}, months={n_months}, weeks={n_weeks}, kpis={'on' if cfg.include_kpis else 'off'})")

if __name__ == "__main__":
    main()
