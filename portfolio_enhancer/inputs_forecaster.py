
import numpy as np
import pandas as pd

class VIXNowcaster:
    """
    AR(1) mean-reverting nowcaster for VIX with clipping to [10, 60].
    x_{t+1} = mu + phi*(x_t - mu) + sigma*eps
    """
    def __init__(self, floor: float = 10.0, ceiling: float = 60.0):
        self.mu = None
        self.phi = None
        self.sigma = None
        self.floor = float(floor)
        self.ceiling = float(ceiling)

    def fit(self, series: pd.Series) -> None:
        s = pd.Series(series).dropna().astype(float)
        if len(s) < 5:
            # Fallback: weak mean reversion around last value
            last = float(s.iloc[-1]) if len(s) else 20.0
            self.mu = last
            self.phi = 0.6
            self.sigma = 1.5
            return

        x = s.values
        mu = float(np.mean(x))
        x_c = x[:-1] - mu
        y_c = x[1:] - mu
        den = float(np.dot(x_c, x_c)) + 1e-8
        phi = float(np.dot(x_c, y_c) / den)
        phi = float(np.clip(phi, -0.95, 0.95))
        resid = y_c - phi * x_c
        sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else 1.5

        self.mu = mu
        self.phi = phi
        self.sigma = max(1e-6, sigma)

    def simulate(self, n_paths: int, n_days: int, last_value: float) -> np.ndarray:
        """
        Returns array shape (n_paths, n_days) of synthetic daily levels.
        Uses numpy's global RNG (seed upstream for determinism).
        """
        mu = self.mu if self.mu is not None else float(last_value)
        phi = self.phi if self.phi is not None else 0.6
        sigma = self.sigma if self.sigma is not None else 1.5

        out = np.empty((n_paths, n_days), dtype=float)
        out[:, 0] = np.clip(mu + phi * (last_value - mu) + sigma * np.random.normal(size=n_paths),
                            self.floor, self.ceiling)

        for t in range(1, n_days):
            eps = np.random.normal(size=n_paths)
            out[:, t] = mu + phi * (out[:, t-1] - mu) + sigma * eps
            out[:, t] = np.clip(out[:, t], self.floor, self.ceiling)

        return out


class DriftNowcaster:
    """
    Random walk with drift + Gaussian noise.
    Suitable for TNX (yields) and DXY (dollar index).
    """
    def __init__(self, floor: float | None = None, ceiling: float | None = None):
        self.drift = None  # per-day
        self.sigma = None  # per-day
        self.floor = floor
        self.ceiling = ceiling

    def fit(self, series: pd.Series) -> None:
        s = pd.Series(series).dropna().astype(float)
        if len(s) < 3:
            self.drift = 0.0
            self.sigma = 0.1
            return
        x = s.values
        dx = np.diff(x)
        self.drift = float(np.mean(dx))
        self.sigma = float(np.std(dx, ddof=1)) if len(dx) > 1 else 0.1
        self.sigma = max(1e-6, self.sigma)

    def simulate(self, n_paths: int, n_days: int, last_value: float) -> np.ndarray:
        out = np.empty((n_paths, n_days), dtype=float)
        out[:, 0] = last_value + self.drift + self.sigma * np.random.normal(size=n_paths)
        if self.floor is not None:
            out[:, 0] = np.maximum(out[:, 0], self.floor)
        if self.ceiling is not None:
            out[:, 0] = np.minimum(out[:, 0], self.ceiling)

        for t in range(1, n_days):
            step = self.drift + self.sigma * np.random.normal(size=n_paths)
            out[:, t] = out[:, t-1] + step
            if self.floor is not None or self.ceiling is not None:
                if self.floor is not None:
                    out[:, t] = np.maximum(out[:, t], self.floor)
                if self.ceiling is not None:
                    out[:, t] = np.minimum(out[:, t], self.ceiling)
        return out


def kpi_persistence_sim(last_value: float, n_paths: int, noise_std: float = 0.05) -> np.ndarray:
    """
    Return shape (n_paths,) with small noise and clipping to [-1, +1].
    If noise_std <= 0, it is pure persistence.
    """
    vals = last_value + noise_std * np.random.normal(size=n_paths)
    return np.clip(vals, -1.0, 1.0)
