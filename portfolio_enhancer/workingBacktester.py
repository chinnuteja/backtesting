from __future__ import annotations

import asyncio
from typing import Dict, Optional, Tuple, List
import contextlib, io

import numpy as np
import pandas as pd
import yfinance as yf

# --- External KPI modules (async) ---
from .kpis.gold.gold_kpi_composite import gold_kpi_composite
from .kpis.bitcoin.btc_sentiment import analyze_btc_sentiment
from .kpis.reit import reit_kpi_composite
from .kpis.nifty.nifty_composite import compute_nifty_composite  # NEW ✅

# =========================
# Global config
# =========================
START_DATE = "2015-01-01"
END_DATE   = "2023-12-31"

TICKERS = {
    "Equities": "^NSEI",   # NIFTY 50
    "Gold":     "GLD",
    "REITs":    "VNQ",
    "Bitcoin":  "BTC-USD",
}

FLOORS = {"Equities": 0.05, "Gold": 0.05, "REITs": 0.05, "Bitcoin": 0.05}
CAPS   = {"Equities": 0.60, "Gold": 0.65, "REITs": 0.50, "Bitcoin": 0.20}
TURNOVER_LIMIT = 0.30  # L1 per decision
EWMA_SPAN = 63

# ---- Output controls ----
PRINT_MODE = "summary"           # "full" | "summary" | "quiet"
SHOW_KPI_SUMMARY = True
SILENCE_KPI_LOGS = True
PRINT_LAST_N_MONTHS = 0          # 0 = print all

# KPI safety
KPI_TIMEOUT_SEC = 60

np.set_printoptions(suppress=True, linewidth=120)

# =========================
# Utilities
# =========================
def _to_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x.tz_localize(None) if x.tzinfo is not None else x
    return pd.Timestamp(x).tz_localize(None)

def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:+.2f}%"

def _fmt_w_short(w: Dict[str, float]) -> str:
    return "E={:.1f}% G={:.1f}% R={:.1f}% B={:.1f}%".format(
        100*w["Equities"], 100*w["Gold"], 100*w["REITs"], 100*w["Bitcoin"]
    )

def _hletter(horizon_key: str) -> str:
    return {"monthly": "M", "quarterly": "Q", "half": "H", "yearly": "Y"}.get(horizon_key, "M")

def _should_print_period(i: int, total: int) -> bool:
    if PRINT_MODE == "quiet":
        return False
    if PRINT_LAST_N_MONTHS and (total - i) > PRINT_LAST_N_MONTHS:
        return False
    return True

async def _await_quietly(coro):
    if not SILENCE_KPI_LOGS:
        return await coro
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return await coro

async def _safe_kpi(label: str, coro) -> Dict:
    try:
        return await asyncio.wait_for(_await_quietly(coro), timeout=KPI_TIMEOUT_SEC)
    except Exception as e:
        if PRINT_MODE != "quiet":
            print(f"[WARN] {label} KPI failed: {e!s} — using neutral 0.0")
        return {"composite_sentiment": 0.0, "components": {}}

# =========================
# Data loading
# =========================
def _fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    if PRINT_MODE == "full":
        print(f"Fetching {ticker} {start} → {end}")
    tkr = yf.Ticker(ticker)
    # yfinance end is exclusive -> add +1 day
    hist = tkr.history(start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1), auto_adjust=True)
    if isinstance(hist.index, pd.DatetimeIndex):
        hist.index = hist.index.tz_localize(None)
    return hist

def load_returns_fresh() -> pd.DataFrame:
    series = []
    for name, ticker in TICKERS.items():
        hist = _fetch_history(ticker, START_DATE, END_DATE)
        s = hist["Close"].pct_change().dropna().rename(name)
        series.append(s)
        if PRINT_MODE == "full":
            print(f"✅ fetched {name} ({ticker}): {len(hist):,} rows")
    df = pd.concat(series, axis=1).sort_index()
    df.index = df.index.tz_localize(None)
    if PRINT_MODE == "full":
        print(f"Loaded returns: {df.shape}")
    return df

# =========================
# Estimators
# =========================
def _slice_training(returns: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    end = _to_ts(decision_date) - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=3)
    return returns.loc[(returns.index >= start) & (returns.index <= end)]

def _ewma_mu_daily(returns: pd.DataFrame, decision_date: pd.Timestamp) -> pd.Series:
    sub = _slice_training(returns, decision_date)
    if sub.empty:
        return pd.Series(0.0, index=returns.columns)
    mu = sub.ewm(span=EWMA_SPAN, adjust=False).mean().iloc[-1]
    if PRINT_MODE == "full":
        print(f"  EWMA μ(daily): {sub.index.min().date()} → {sub.index.max().date()} (span={EWMA_SPAN})")
    return mu

def _horizon_multiplier(horizon_key: str) -> int:
    return {"monthly": 21, "quarterly": 63, "half": 126, "yearly": 252}.get(horizon_key, 21)

# =========================
# Optimizer helpers
# =========================
def _project_capped_simplex(
    w_raw: Dict[str, float],
    floors: Dict[str, float],
    caps: Dict[str, float],
) -> Dict[str, float]:
    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    lo = np.array([floors.get(k, 0.0) for k in keys], dtype=float)
    hi = np.array([caps.get(k, 1.0)   for k in keys], dtype=float)
    x0 = np.array([max(0.0, float(w_raw.get(k, 0.0))) for k in keys], dtype=float)

    if lo.sum() > 1.0:
        lo = lo / lo.sum()

    x0 = np.minimum(hi, np.maximum(lo, x0))
    if abs(float(x0.sum()) - 1.0) <= 1e-12:
        return {k: float(v) for k, v in zip(keys, x0)}

    tau_lo = (x0 - hi).min() - 1.0
    tau_hi = (x0 - lo).max() + 1.0
    for _ in range(80):
        tau = 0.5 * (tau_lo + tau_hi)
        x = np.clip(x0 - tau, lo, hi)
        s = float(x.sum())
        if s > 1.0:
            tau_lo = tau
        else:
            tau_hi = tau
    x = np.clip(x0 - 0.5 * (tau_lo + tau_hi), lo, hi)
    x = x / max(1e-12, x.sum())
    return {k: float(v) for k, v in zip(keys, x)}

def _apply_turnover(prev_w: Optional[Dict[str, float]], new_w: Dict[str, float], limit: float) -> Dict[str, float]:
    if prev_w is None:
        return new_w
    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    p = np.array([prev_w[k] for k in keys], dtype=float)
    n = np.array([new_w[k]  for k in keys], dtype=float)
    diff = n - p
    l1 = float(np.abs(diff).sum())
    if l1 <= limit + 1e-12:
        return new_w
    scale = limit / max(1e-12, l1)
    adj = p + diff * scale
    adj = adj / max(1e-12, adj.sum())
    out = {k: float(v) for k, v in zip(keys, adj)}
    return _project_capped_simplex(out, FLOORS, CAPS)

# =========================
# KPI wrappers (sequential + silenced + timeout)
# =========================
async def _btc_kpi(decision_date: pd.Timestamp, horizon_key: str) -> Dict:
    asof = decision_date.date().isoformat()
    return await _safe_kpi("BTC", analyze_btc_sentiment(
        backtest_date=asof, horizon=_hletter(horizon_key), historical_cutoff=asof
    ))

async def _gold_kpi(decision_date: pd.Timestamp, horizon_key: str) -> Dict:
    asof = decision_date.date().isoformat()
    try:
        coro = gold_kpi_composite(asof, _hletter(horizon_key))
    except TypeError:
        coro = gold_kpi_composite(asof)
    return await _safe_kpi("GOLD", coro)

async def _reit_kpi(decision_date: pd.Timestamp, horizon_key: str) -> Dict:
    asof = decision_date.date().isoformat()
    return await _safe_kpi("REIT", reit_kpi_composite(asof, _hletter(horizon_key)))

async def _nifty_kpi(decision_date: pd.Timestamp, horizon_key: str) -> Dict:
    asof = decision_date.date().isoformat()
    # compute the NIFTY composite (FII/DII + Global + VIX + Technicals + RBI)
    return await _safe_kpi("NIFTY", compute_nifty_composite(asof, _hletter(horizon_key), verbose=False))

# =========================
# Period helpers
# =========================
def _period_dates_monthly(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for m in range(1, 12 + 1):
        period_end = _to_ts(f"{year}-{m:02d}-01") + pd.offsets.MonthEnd(0)
        decision_date = period_end - pd.offsets.MonthEnd(1)
        out.append((_to_ts(decision_date), _to_ts(period_end)))
    return out

def _period_dates_quarterly(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    q_ends = [f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
    out = []
    for q_end_str in q_ends:
        period_end = _to_ts(q_end_str)
        decision_date = period_end - pd.offsets.QuarterEnd(1)
        out.append((_to_ts(decision_date), _to_ts(period_end)))
    return out

def _period_dates_halfyear(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    ends = [f"{year}-06-30", f"{year}-12-31"]
    out = []
    for e in ends:
        period_end = _to_ts(e)
        decision_date = _to_ts(f"{year-1}-12-31") if e.endswith("06-30") else _to_ts(f"{year}-06-30")
        out.append((decision_date, period_end))
    return out

def _period_dates_year(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    period_end = _to_ts(f"{year}-12-31")
    decision_date = _to_ts(f"{year-1}-12-31")
    return [(decision_date, period_end)]

def _realized_period_return(returns: pd.DataFrame, start_exclusive: pd.Timestamp, end_inclusive: pd.Timestamp) -> pd.Series:
    mask = (returns.index > start_exclusive) & (returns.index <= end_inclusive)
    sub = returns.loc[mask]
    if sub.empty:
        return pd.Series(0.0, index=returns.columns)
    gross = (1.0 + sub).prod(axis=0)
    return gross - 1.0

# =========================
# Forecasting
# =========================
async def forecast_one_period(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    period_end: pd.Timestamp,
    horizon_key: str,
    w_prev: Optional[Dict[str, float]],
    period_index: int = 1,
    period_count: int = 1,
) -> Tuple[Dict[str, float], float, Dict[str, Dict], float]:

    if PRINT_MODE == "full" and _should_print_period(period_index, period_count):
        label = "YEAR" if horizon_key == "yearly" else horizon_key.upper()[:-2]
        print(f"[{label}] Forecasting {period_end.strftime('%b %Y')}  |  using data/KPIs up to {decision_date.date()}")

    mu_daily = _ewma_mu_daily(returns, decision_date)
    mu_h = mu_daily * _horizon_multiplier(horizon_key)

    # KPIs — sequential, silent, timeout-guarded
    nifty = await _nifty_kpi(decision_date, horizon_key)  # NEW ✅
    btc   = await _btc_kpi(decision_date, horizon_key)
    gold  = await _gold_kpi(decision_date, horizon_key)
    reit  = await _reit_kpi(decision_date, horizon_key)

    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    mu_vec = mu_h.reindex(keys).fillna(0.0).to_numpy()
    if not np.isfinite(mu_vec).all():
        mu_vec = np.nan_to_num(mu_vec, nan=0.0)

    if (mu_vec > 0).any():
        w_raw = mu_vec.clip(min=0.0)
        w_raw = w_raw / w_raw.sum()
    else:
        w_raw = np.ones_like(mu_vec) / len(mu_vec)

    w0 = {k: float(v) for k, v in zip(keys, w_raw)}

    # Floors/Caps projection, then turnover control
    w_proj = _project_capped_simplex(w0, FLOORS, CAPS)
    w_final = _apply_turnover(w_prev, w_proj, TURNOVER_LIMIT)

    exp_h = float(np.dot(mu_vec, np.array([w_final[k] for k in keys])))
    realized_vec = _realized_period_return(returns, decision_date, period_end)
    realized = float(np.dot(realized_vec.reindex(keys).values, np.array([w_final[k] for k in keys])))

    if _should_print_period(period_index, period_count):
        if PRINT_MODE == "full":
            print(f"  w: {_fmt_w_short(w_final)}")
            print(f"  Expected {('monthly' if horizon_key=='monthly' else 'period')} return: {_fmt_pct(exp_h)}")
            print(f"  Realized {'month' if horizon_key=='monthly' else 'period'} return: {_fmt_pct(realized)}\n")
        elif PRINT_MODE == "summary":
            line = f"{period_end.strftime('%Y-%m')} | w: {_fmt_w_short(w_final)} | exp={_fmt_pct(exp_h)} | real={_fmt_pct(realized)}"
            print(line)
            if SHOW_KPI_SUMMARY:
                print(f"    KPI: eq={float(nifty.get('composite_sentiment',0.0)):+.3f} "
                      f"gold={float(gold.get('composite_sentiment',0.0)):+.3f} "
                      f"reit={float(reit.get('composite_sentiment',0.0)):+.3f} "
                      f"btc={float(btc.get('composite_sentiment',0.0)):+.3f}")

    kpi_pack = {"nifty": nifty, "bitcoin": btc, "gold": gold, "reit": reit}
    return w_final, exp_h, kpi_pack, realized

async def forecast_report_2023():
    returns = load_returns_fresh()

    # ---- MONTHLY ----
    if PRINT_MODE != "quiet":
        print("\n\n========== MONTHLY FORECASTS (12) ==========")
        print("(For month M, decision date = previous month end; EWMA & KPIs use data up to that date.)\n")

    monthly_periods = _period_dates_monthly(2023)
    w_prev = None
    monthly_turnovers_l1: List[float] = []
    monthly_realized: List[float] = []

    for idx, (decision_date, period_end) in enumerate(monthly_periods, start=1):
        w, _, _, realized = await forecast_one_period(
            returns, decision_date, period_end, "monthly", w_prev,
            period_index=idx, period_count=len(monthly_periods)
        )
        if w_prev is not None:
            keys = ["Equities", "Gold", "REITs", "Bitcoin"]
            l1 = float(sum(abs(w[k] - w_prev[k]) for k in keys))
            monthly_turnovers_l1.append(l1)
            if PRINT_MODE == "full" and _should_print_period(idx, len(monthly_periods)):
                print(f"  Turnover this month (L1): {100.0*l1:.2f}%  |  Limit: {100.0*TURNOVER_LIMIT:.2f}%\n")
        w_prev = w
        monthly_realized.append(realized)

    # ---- QUARTERLY ----
    if PRINT_MODE != "quiet":
        print("\n========== QUARTERLY FORECASTS (3: Q2, Q3, Q4) ==========")
        print("(Each quarter uses EWMA & KPIs up to previous quarter end; Q1 is the 2022-end decision.)\n")
    for idx, (decision_date, period_end) in enumerate(_period_dates_quarterly(2023), start=1):
        await forecast_one_period(returns, decision_date, period_end, "quarterly", w_prev, idx, 3)

    # ---- HALF-YEAR ----
    if PRINT_MODE != "quiet":
        print("\n\n========== HALF-YEAR FORECASTS (2: H1, H2) ==========")
        print("(H1 uses 2022-12-31 EWMA/KPIs; H2 uses 2023-06-30 EWMA/KPIs.)\n")
    for idx, (decision_date, period_end) in enumerate(_period_dates_halfyear(2023), start=1):
        await forecast_one_period(returns, decision_date, period_end, "half", w_prev, idx, 2)

    # ---- YEARLY ----
    if PRINT_MODE != "quiet":
        print("\n\n========== YEARLY FORECAST (1) ==========")
        print("(Y2023 uses EWMA & KPIs up to 2022-12-31.)\n")
    for idx, (decision_date, period_end) in enumerate(_period_dates_year(2023), start=1):
        await forecast_one_period(returns, decision_date, period_end, "yearly", w_prev, idx, 1)

    # ---- Final summary ----
    gross = np.prod([1.0 + r for r in monthly_realized]) - 1.0 if monthly_realized else 0.0
    ann_ret = gross
    ann_vol = float(np.std(monthly_realized, ddof=1) * np.sqrt(12.0)) if len(monthly_realized) > 1 else 0.0
    sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
    avg_turn_l1 = float(np.mean(monthly_turnovers_l1)) if monthly_turnovers_l1 else 0.0

    print("\n========== MONTHLY POLICY: 2023 REALIZED PERFORMANCE ==========")
    print(f"Total: {_fmt_pct(gross)}  |  AnnRet: {_fmt_pct(ann_ret)}  |  AnnVol: {100*ann_vol:.2f}%  |  Sharpe: {sharpe:.2f}")
    print(f"Avg monthly turnover: {100*avg_turn_l1:.2f}%")
    print("Done. (12 + 3 + 2 + 1 = 18 forecasts printed.)")


if __name__ == "__main__":
    asyncio.run(forecast_report_2023())
