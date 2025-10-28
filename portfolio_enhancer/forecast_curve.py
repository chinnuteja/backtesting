# portfolio_enhancer/forecast_curve.py

import os
import pickle
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ========= CONFIG =========

CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache_forecast")
os.makedirs(CACHE_DIR, exist_ok=True)

ASSET_TICKERS = {
    "Equities": "^NSEI",      # NIFTY50 (you can swap for SPY if needed)
    "Gold": "GLD",
    "REITs": "VNQ",
    "Bitcoin": "BTC-USD",
}

MACRO_TICKERS = {
    "VIX": "^VIX",
    "TNX": "^TNX",        # 10Y yield proxy
    "USD": "DX-Y.NYB",    # DXY index; if fails you can switch to "DXY" or "INR=X"
}

LOOKBACK_YEARS = 6        # for expanding window we still need enough history
TRAIN_MIN_POINTS = 200    # don't train if we don't at least have this many rows


# ========= SMALL UTILS =========

def _cache_path(prefix: str, key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.pkl")


def _cache_get(prefix: str, key: str):
    path = _cache_path(prefix, key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _cache_set(prefix: str, key: str, value):
    path = _cache_path(prefix, key)
    with open(path, "wb") as f:
        pickle.dump(value, f)


def _forward_fill_limit(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    # forward-fill gaps up to 'limit' days, like our main pipeline logic
    return df.ffill(limit=limit)


def download_prices(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Pull Adj Close from yfinance for all tickers. Returns wide DF [date x tickers]
    """
    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )["Adj Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index()
    return data


def load_market_and_macro(start_plot: pd.Timestamp,
                          end_plot: pd.Timestamp,
                          horizon_days: int) -> Tuple[pd.DataFrame, Dict[str, pd.Series], pd.DataFrame]:
    """
    We need:
    - enough history before start_plot (for features & training window)
    - enough future after end_plot (for realized forward return calc)

    Returns:
      prices_all: wide DF of all assets [date -> price columns per asset]
      asset_series_map: dict(asset -> price series)
      macro_df: DF with columns [VIX,TNX,USD]
    """
    hist_start = start_plot - pd.DateOffset(years=LOOKBACK_YEARS)
    hist_end = end_plot + pd.Timedelta(days=horizon_days + 5)

    cache_key = f"{hist_start}_{hist_end}_{sorted(ASSET_TICKERS.items())}_{sorted(MACRO_TICKERS.items())}"
    cached = _cache_get("mktmacro", cache_key)
    if cached is not None:
        return cached

    # 1. download assets
    asset_prices_raw = download_prices(list(ASSET_TICKERS.values()), hist_start, hist_end)
    # rename columns to friendly asset names
    rename_map = {v: k for k, v in ASSET_TICKERS.items()}
    asset_prices = asset_prices_raw.rename(columns=rename_map)

    # 2. download macro
    macro_prices_raw = download_prices(list(MACRO_TICKERS.values()), hist_start, hist_end)
    macro_rename = {v: k for k, v in MACRO_TICKERS.items()}
    macro_prices = macro_prices_raw.rename(columns=macro_rename)

    # forward fill macro up to 5 days, same style as backtester
    macro_prices = _forward_fill_limit(macro_prices, limit=5)

    # build dict per asset (clean single series)
    asset_series_map = {asset: asset_prices[asset].dropna().sort_index() for asset in ASSET_TICKERS.keys()}

    # macro as df
    macro_df = macro_prices.sort_index()

    out = (asset_prices, asset_series_map, macro_df)
    _cache_set("mktmacro", cache_key, out)
    return out


# ========= FEATURE ENGINEERING =========

def build_feature_frame(price: pd.Series, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    price: pd.Series of asset close prices
    macro_df: columns ['VIX','TNX','USD'] (or subset), indexed by date
    returns DF indexed by date with engineered features
    """

    df = pd.DataFrame(index=price.index)
    df["px"] = price
    df["ret_d"] = df["px"].pct_change()

    # technical features
    df["mom_3m"] = df["px"] / df["px"].shift(63) - 1.0
    df["mom_12m"] = df["px"] / df["px"].shift(252) - 1.0
    df["trend_accel"] = df["mom_3m"] - df["mom_12m"]

    roll_max_21 = df["px"].rolling(window=21, min_periods=21).max()
    df["drawdown_1m"] = df["px"] / roll_max_21 - 1.0

    df["trail_vol"] = df["ret_d"].rolling(window=21, min_periods=21).std()

    roll_mean_63 = df["px"].rolling(window=63, min_periods=63).mean()
    df["stretch_raw"] = df["px"] / roll_mean_63 - 1.0

    # stretch_z: zscore of stretch_raw over ~1y
    st_raw_mean = df["stretch_raw"].rolling(window=252, min_periods=100).mean()
    st_raw_std = df["stretch_raw"].rolling(window=252, min_periods=100).std()
    df["stretch_z"] = (df["stretch_raw"] - st_raw_mean) / st_raw_std

    # relative_strength style: momentum / vol
    df["relative_strength"] = df["mom_3m"] / (df["trail_vol"] + 1e-6)

    # merge macro
    # we'll ffill to align to asset dates
    mac = macro_df.reindex(df.index).ffill(limit=5)
    df["vix_level"] = mac["VIX"]
    df["rate_level"] = mac["TNX"]
    if "USD" in mac.columns:
        df["usd_level"] = mac["USD"]
    else:
        df["usd_level"] = np.nan

    df["macro_spread"] = df["rate_level"] - df["usd_level"]

    # final clean
    return df


def make_supervised_df(price: pd.Series,
                       macro_df: pd.DataFrame,
                       horizon_days: int) -> pd.DataFrame:
    """
    Build the full table for one asset & one horizon.

    Returns columns:
      - features...
      - fwd_return (target)
    Index = date (as_of)
    """

    feat = build_feature_frame(price, macro_df)

    # compute forward cumulative return over horizon_days
    # index approach: idx(t) = cumprod(1+ret) so fwd_ret = idx[t+h]/idx[t]-1
    ret_d = price.pct_change()
    idx = (1.0 + ret_d).cumprod()
    fwd_idx = idx.shift(-horizon_days)
    feat["fwd_return"] = (fwd_idx / idx) - 1.0

    return feat


# ========= WALK-FORWARD FORECASTING =========

@dataclass
class WalkForwardResult:
    pred_df: pd.DataFrame  # columns: ['as_of','pred_ret']
    real_df: pd.DataFrame  # columns: ['as_of','real_fwd_ret']
    metrics: dict          # summary stats for this asset


def expand_walk_forward_forecast(df_sup: pd.DataFrame,
                                 asset_name: str,
                                 horizon_days: int,
                                 start_plot: pd.Timestamp,
                                 end_plot: pd.Timestamp) -> WalkForwardResult:
    """
    df_sup: output of make_supervised_df()
           index=date, columns=[features..., 'fwd_return']

    We will:
      - loop through each day t in [start_plot, end_plot]
      - build training set: all rows strictly < t
      - fit GradientBoostingRegressor
      - predict fwd_return for row t
      - store realized fwd_return[t] too

    expanding window (not fixed length).
    """

    df_sup = df_sup.sort_index()

    all_dates = df_sup.index
    test_dates = all_dates[(all_dates >= start_plot) & (all_dates <= end_plot)]

    rows_pred = []
    rows_real = []

    feature_cols = [
        "mom_3m", "mom_12m", "trend_accel",
        "drawdown_1m", "trail_vol",
        "stretch_raw", "stretch_z", "relative_strength",
        "vix_level", "rate_level",
        "macro_spread",
    ]

    feature_cols = [c for c in feature_cols if c in df_sup.columns]

    for t in test_dates:
        # training mask: rows strictly before t
        train_mask = (df_sup.index < t)
        train_df = df_sup.loc[train_mask, :]

        # enforce min history so model has something non-garbage
        train_df = train_df.dropna(subset=feature_cols + ["fwd_return"])
        if len(train_df) < TRAIN_MIN_POINTS:
            continue

        # the row to predict at time t
        if t not in df_sup.index:
            continue
        row_t = df_sup.loc[[t], :]
        if row_t[feature_cols].isna().any(axis=1).iloc[0]:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df["fwd_return"].values

        # fit model
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            subsample=0.7,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # predict for t
        pred_t = model.predict(row_t[feature_cols].values)[0]
        real_t = row_t["fwd_return"].iloc[0]

        rows_pred.append({"as_of": t, "pred_ret": pred_t})
        rows_real.append({"as_of": t, "real_fwd_ret": real_t})

    pred_df = pd.DataFrame(rows_pred).sort_values("as_of").reset_index(drop=True)
    real_df = pd.DataFrame(rows_real).sort_values("as_of").reset_index(drop=True)

    # compute metrics for the whole horizon window
    metrics = compute_metrics(pred_df, real_df, asset_name)

    return WalkForwardResult(
        pred_df=pred_df,
        real_df=real_df,
        metrics=metrics,
    )


# ========= METRICS =========

def safe_corr(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < 2:
        return np.nan
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]):
        return np.nan
    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan


def compute_metrics(pred_df: pd.DataFrame,
                    real_df: pd.DataFrame,
                    asset_name: str) -> dict:
    """
    Join pred/real on as_of, compute aggregate metrics.
    """
    merged = pd.merge(pred_df, real_df, on="as_of", how="inner").dropna()

    if len(merged) == 0:
        return {
            "asset": asset_name,
            "avg_pred": np.nan,
            "avg_real": np.nan,
            "bias": np.nan,
            "MAE": np.nan,
            "MSE": np.nan,
            "R2": np.nan,
            "corr": np.nan,
            "n_points": 0,
        }

    y_pred = merged["pred_ret"].values
    y_real = merged["real_fwd_ret"].values

    avg_pred = float(np.mean(y_pred))
    avg_real = float(np.mean(y_real))
    bias = avg_pred - avg_real
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    r2 = safe_r2(y_real, y_pred)
    c = safe_corr(y_pred, y_real)

    return {
        "asset": asset_name,
        "avg_pred": avg_pred,
        "avg_real": avg_real,
        "bias": bias,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "corr": c,
        "n_points": len(merged),
    }


# ========= PRETTY PLOTTING =========

def plot_forecasts(results: Dict[str, WalkForwardResult],
                   horizon_days: int,
                   start: pd.Timestamp,
                   end: pd.Timestamp):
    """
    results: dict[asset] = WalkForwardResult

    We draw a clean 2x2 grid:
    - solid black = realized forward % return
    - dashed blue = predicted forward % return
    - title shows IC (corr)
    - tight layout, single readable legend per subplot
    - saved PNG
    """

    # consistent visual style
    plt.style.use("seaborn-v0_8-whitegrid")

    assets = list(results.keys())
    # sort to keep stable order
    assets = ["Bitcoin", "Equities", "Gold", "REITs"]
    assets = [a for a in assets if a in results]

    # create 2x2
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.flatten()

    for i, asset in enumerate(assets):
        ax = axs[i]
        wf_res = results[asset]

        pred_df = wf_res.pred_df.copy()
        real_df = wf_res.real_df.copy()

        pred_df["as_of"] = pd.to_datetime(pred_df["as_of"])
        real_df["as_of"] = pd.to_datetime(real_df["as_of"])

        # filter window
        pred_df = pred_df[(pred_df["as_of"] >= start) & (pred_df["as_of"] <= end)].sort_values("as_of")
        real_df = real_df[(real_df["as_of"] >= start) & (real_df["as_of"] <= end)].sort_values("as_of")

        # fetch corr from precomputed metrics
        ic_val = wf_res.metrics.get("corr", np.nan)
        if ic_val is None or np.isnan(ic_val):
            ic_txt = "IC: n/a"
        else:
            ic_txt = f"IC: {ic_val:.2f}"

        # realized (black solid)
        ax.plot(
            real_df["as_of"],
            real_df["real_fwd_ret"] * 100.0,
            color="black",
            linewidth=1.6,
            label=f"{asset} realized {horizon_days}d fwd %",
        )

        # predicted (blue dashed)
        ax.plot(
            pred_df["as_of"],
            pred_df["pred_ret"] * 100.0,
            color="#1f77b4",
            linewidth=1.6,
            linestyle="--",
            label=f"{asset} predicted {horizon_days}d fwd %",
        )

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_ylabel("Return %", fontsize=9)
        ax.set_title(f"{asset}  |  {horizon_days}d forward return   ({ic_txt})",
                     fontsize=11,
                     fontweight="bold")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.3)

        # legend box in upper left corner with light frame
        ax.legend(
            fontsize=8,
            frameon=True,
            fancybox=True,
            framealpha=0.8,
            loc="upper left",
        )

    # hide any unused subplots if <4 assets
    for j in range(len(assets), len(axs)):
        axs[j].axis("off")

    # x-label only on bottom row subplots
    axs[2].set_xlabel("Forecast timestamp (as_of date)", fontsize=9)
    axs[3].set_xlabel("Forecast timestamp (as_of date)", fontsize=9)

    # main figure title
    fig.suptitle(
        f"Model Forecast vs Realized Forward Return ({horizon_days}d horizon)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # save a png so you can share
    outname = f"forecast_vs_realized_{horizon_days}d_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.png"
    plt.savefig(outname, dpi=200)
    print(f"[INFO] Plot saved -> {outname}")

    plt.show()


# ========= MAIN PIPELINE =========

def run_forecast_pipeline(horizon_days: int,
                          start_plot: str,
                          end_plot: str,
                          use_cache: bool = True):
    """
    1. load prices + macro
    2. per asset build supervised DF
    3. walk-forward expanding training up to each day
    4. collect metrics
    5. return all results
    """

    start_plot_dt = pd.Timestamp(start_plot)
    end_plot_dt = pd.Timestamp(end_plot)

    cache_key = f"{horizon_days}_{start_plot_dt}_{end_plot_dt}"
    if use_cache:
        cached = _cache_get("wf_results", cache_key)
        if cached is not None:
            return cached

    print(f"[INFO] Downloading data {start_plot_dt - pd.DateOffset(years=LOOKBACK_YEARS)} -> {end_plot_dt} ...")
    prices_all, asset_series_map, macro_df = load_market_and_macro(
        start_plot_dt, end_plot_dt, horizon_days
    )

    results = {}
    print("[INFO] Building + forecasting per asset ...")

    for asset_name, px_series in asset_series_map.items():
        print(f"[INFO]   {asset_name} ...")

        # supervised dataframe (features + fwd_return target)
        df_sup = make_supervised_df(px_series, macro_df, horizon_days)

        # walk-forward expanding-window forecasting
        wf_res = expand_walk_forward_forecast(
            df_sup=df_sup,
            asset_name=asset_name,
            horizon_days=horizon_days,
            start_plot=start_plot_dt,
            end_plot=end_plot_dt,
        )

        results[asset_name] = wf_res

    if use_cache:
        _cache_set("wf_results", cache_key, results)

    return results


def print_metrics(results: Dict[str, WalkForwardResult], horizon_days: int):
    print("")
    print(f"MODEL CALIBRATION (horizon {horizon_days}d)")
    print("===================================")
    for asset, wf_res in results.items():
        m = wf_res.metrics

        # nice formatting for R2 / corr including NaN safety
        r2_txt = "nan" if (m["R2"] is None or (isinstance(m["R2"], float) and np.isnan(m["R2"]))) else f"{m['R2']:.2f}"
        corr_txt = "nan" if (m["corr"] is None or (isinstance(m["corr"], float) and np.isnan(m["corr"]))) else f"{m['corr']:.2f}"

        print(
            f"{asset}: "
            f"avg_pred={m['avg_pred']*100:.2f}% "
            f"avg_real={m['avg_real']*100:.2f}% "
            f"bias={(m['bias']*100):.2f}% "
            f"MAE={m['MAE']*100:.2f}% "
            f"MSE={m['MSE']:.4f} "
            f"R2={r2_txt} "
            f"corr={corr_txt} "
            f"n={m['n_points']}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward forecast curve + metrics + plot"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Forward return horizon in trading days (e.g. 21,63,126,252)",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Plot start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Plot end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache (force retrain)",
    )

    args = parser.parse_args()

    horizon_days = args.horizon
    start_plot = pd.Timestamp(args.start)
    end_plot = pd.Timestamp(args.end)

    # 1. run pipeline (download -> features -> walk-forward -> metrics)
    results = run_forecast_pipeline(
        horizon_days=horizon_days,
        start_plot=args.start,
        end_plot=args.end,
        use_cache=not args.no_cache,
    )

    # 2. print metrics
    print_metrics(results, horizon_days)

    # 3. plot curves (pretty 2x2 grid + PNG export)
    print("[INFO] Plotting ...")
    plot_forecasts(results, horizon_days, start_plot, end_plot)


if __name__ == "__main__":
    main()
