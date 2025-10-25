
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Callable, Any

def run_scenarios(
    n_paths: int,
    horizon_days: int,
    last_inputs: Dict[str, float],   # {"VIX": v, "TNX": t, "DXY": d, "KPI_Equities": k, ...}
    forecasters: Dict[str, Any],     # {"VIX": VIXNowcaster, "TNX": DriftNowcaster, "DXY": DriftNowcaster}
    decision_callback: Callable[[dict], tuple],
):
    """
    Runs N simulated paths for VIX/TNX/DXY and persistence for KPIs.
    decision_callback gets a per-path dict of inputs and must return (weights_dict, exp_float[, exp_assets_vec]).
    Returns:
      weights_paths: shape (n_paths, n_assets) in asset order ["Equities","Gold","REITs","Bitcoin"]
      exp_paths: shape (n_paths,)
      aux_summary: dict for diagnostics (e.g., vix_median, vix_p90, exp_assets_paths)
    """
    N = int(n_paths)
    H = int(horizon_days)
    if N <= 0 or H <= 0:
        raise ValueError("n_paths and horizon_days must be positive")

    # Simulate auxiliary inputs
    vix_paths = forecasters["VIX"].simulate(N, H, last_inputs["VIX"])
    tnx_paths = forecasters["TNX"].simulate(N, H, last_inputs["TNX"])
    dxy_paths = forecasters["DXY"].simulate(N, H, last_inputs["DXY"])

    # KPI composites: persistence with tiny noise (done inside callback for variety)

    # vectorized storage
    keys = ["Equities", "Gold", "REITs", "Bitcoin"]
    weights_paths = np.zeros((N, len(keys)), dtype=float)
    exp_paths = np.zeros(N, dtype=float)
    exp_assets_paths = None  # lazily create when callback returns vec

    for s in range(N):
        vpath = vix_paths[s]
        tpath = tnx_paths[s]
        dpath = dxy_paths[s]

        # 5-day shock window relative to start
        tnx_bps = (tpath[min(5, H-1)] - tpath[0]) * 10.0  # ^TNX *10 ~= bps
        dxy_pct = 100.0 * (dpath[min(5, H-1)] - dpath[0]) / (dpath[0] if dpath[0] != 0 else 1.0)

        callback_inputs = {
            "VIX_path": vpath,
            "TNX_path": tpath,
            "DXY_path": dpath,
            "VIX_mean": float(np.mean(vpath)),
            "TNX_5d_bps": float(tnx_bps),
            "DXY_5d_pct": float(dxy_pct),
        }

        out = decision_callback(callback_inputs)
        if isinstance(out, tuple) and len(out) == 3:
            w_s, exp_s, exp_assets_vec = out
            if exp_assets_paths is None:
                exp_assets_paths = np.zeros((N, len(keys)), dtype=float)
            exp_assets_paths[s, :] = np.asarray(exp_assets_vec, dtype=float)
        else:
            w_s, exp_s = out
        weights_paths[s, :] = [w_s["Equities"], w_s["Gold"], w_s["REITs"], w_s["Bitcoin"]]
        exp_paths[s] = float(exp_s)

    aux_summary = {
        "vix_med": float(np.median(np.mean(vix_paths, axis=1))),
        "vix_p90": float(np.quantile(np.mean(vix_paths, axis=1), 0.90)),
    }
    if exp_assets_paths is not None:
        aux_summary["exp_assets_paths"] = exp_assets_paths
    return weights_paths, exp_paths, aux_summary
