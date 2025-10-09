# portfolio_enhancer/evr_charts.py
# Create clean Expected vs Realized charts for 2023 from backtester logs.
# Usage:
#   python -m portfolio_enhancer.evr_charts --log bt_2023.log --prefix EVR_2023

from __future__ import annotations
import re
import argparse
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -------------------------
# Helpers: parsing the log
# -------------------------

SECTION_PATTERNS = {
    "monthly": re.compile(r"^=+ MONTHLY FORECASTS"),
    "quarterly": re.compile(r"^=+ QUARTERLY FORECASTS"),
    "half": re.compile(r"^=+ HALF-YEAR FORECASTS"),
    "yearly": re.compile(r"^=+ YEARLY FORECAST"),
}

# Lines that look like:
# 2023-06 | w: ... | exp=+1.31% | real=+4.34%
LINE_RE = re.compile(
    r"^(?P<period>\d{4}-\d{2})\s+\|\s+w:\s+.*?\|\s+exp=(?P<exp>[+\-]?\d+\.\d+)%\s+\|\s+real=(?P<real>[+\-]?\d+\.\d+)%",
    re.IGNORECASE
)

def parse_backtester_log(path: str, year: int = 2023) -> Dict[str, pd.DataFrame]:
    """
    Parse the backtester console output into four DataFrames:
    - monthly:   columns [period, exp, real], 12 rows (Jan..Dec where available)
    - quarterly: columns [period, exp, real], 3 rows (Q2, Q3, Q4 in your logs)
    - half:      columns [period, exp, real], 2 rows (H1, H2)
    - yearly:    columns [period, exp, real], 1 row (Y2023)
    """
    section = None
    buckets = {"monthly": [], "quarterly": [], "half": [], "yearly": []}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            # Detect section changes
            for name, pat in SECTION_PATTERNS.items():
                if pat.search(line):
                    section = name
                    break

            # If we're inside a known section, parse period lines
            if section and (m := LINE_RE.match(line)):
                per = m.group("period")
                if not per.startswith(f"{year}-"):
                    continue
                exp = float(m.group("exp")) / 100.0
                real = float(m.group("real")) / 100.0
                buckets[section].append({"period": per, "exp": exp, "real": real})

    # Build DataFrames
    out = {}
    for k, rows in buckets.items():
        df = pd.DataFrame(rows)
        if df.empty:
            out[k] = df
            continue
        # Keep only 2023
        df = df[df["period"].str.startswith(f"{year}-")].copy()
        df.sort_values("period", inplace=True)
        out[k] = df.reset_index(drop=True)

    # Friendly x labels per timeline
    if not out["monthly"].empty:
        out["monthly"]["label"] = pd.to_datetime(out["monthly"]["period"] + "-01").dt.strftime("%b")
    if not out["quarterly"].empty:
        # Map 2023-03/06/09/12 -> Q1/Q2/Q3/Q4 (your logs typically have 06,09,12)
        out["quarterly"]["label"] = pd.to_datetime(out["quarterly"]["period"] + "-01").dt.quarter.map(
            {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
        )
    if not out["half"].empty:
        # Map 2023-06 -> H1; 2023-12 -> H2
        def half_label(p):
            m = int(p.split("-")[1])
            return "H1" if m <= 6 else "H2"
        out["half"]["label"] = out["half"]["period"].map(half_label)
    if not out["yearly"].empty:
        out["yearly"]["label"] = [str(year)] * len(out["yearly"])

    return out

# -------------------------
# Plotting utilities
# -------------------------

def _percent_axis(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axhline(0, color="#999", linewidth=1, alpha=0.7)

def _save(fig, fname):
    fig.tight_layout()
    fig.savefig(fname, dpi=220)
    plt.close(fig)

# ---- Monthly (12) – 3 options ----
def plot_monthly_opt1(df: pd.DataFrame, outpath: str):
    """Option 1: Grouped bars (Expected vs Realized) across 12 months."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(df))
    width = 0.38
    ax.bar([i - width/2 for i in x], df["exp"], width=width, label="Expected", alpha=0.9)
    ax.bar([i + width/2 for i in x], df["real"], width=width, label="Realized", alpha=0.9)
    ax.set_title("Expected vs Realized – Monthly (2023)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

def plot_monthly_opt2(df: pd.DataFrame, outpath: str):
    """Option 2: Dumbbell (one line per month connecting Expected to Realized)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df))
    # stem/dumbbell
    for i in x:
        ax.plot([i, i], [df["exp"].iloc[i], df["real"].iloc[i]], color="#999", linewidth=2, alpha=0.8)
    ax.scatter(x, df["exp"], s=60, label="Expected")
    ax.scatter(x, df["real"], s=60, label="Realized")
    ax.set_title("Expected vs Realized – Monthly Dumbbell (2023)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend(loc="upper left")
    _save(fig, outpath)

def plot_monthly_opt3(df: pd.DataFrame, outpath: str):
    """Option 3: Dual line chart with markers."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(df))
    ax.plot(x, df["exp"], marker="o", label="Expected")
    ax.plot(x, df["real"], marker="o", label="Realized")
    ax.set_title("Expected vs Realized – Monthly Lines (2023)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

# ---- Quarterly – 3 options ----
def plot_quarterly_opt1(df: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df["exp"], width=width, label="Expected", alpha=0.9)
    ax.bar([i + width/2 for i in x], df["real"], width=width, label="Realized", alpha=0.9)
    ax.set_title("Expected vs Realized – Quarterly (2023)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

def plot_quarterly_opt2(df: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    for i in x:
        ax.plot([i, i], [df["exp"].iloc[i], df["real"].iloc[i]], color="#999", linewidth=2)
    ax.scatter(x, df["exp"], s=70, label="Expected")
    ax.scatter(x, df["real"], s=70, label="Realized")
    ax.set_title("Expected vs Realized – Quarterly Dumbbell (2023)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

def plot_quarterly_opt3(df: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    ax.plot(x, df["exp"], marker="o", label="Expected")
    ax.plot(x, df["real"], marker="o", label="Realized")
    ax.set_title("Expected vs Realized – Quarterly Lines (2023)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

# ---- Half-year – 3 options ----
def plot_half_opt1(df: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df["exp"], width=width, label="Expected", alpha=0.9)
    ax.bar([i + width/2 for i in x], df["real"], width=width, label="Realized", alpha=0.9)
    ax.set_title("Expected vs Realized – Half-year (2023)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

def plot_half_opt2(df: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(df))
    for i in x:
        ax.plot([i, i], [df["exp"].iloc[i], df["real"].iloc[i]], color="#999", linewidth=2)
    ax.scatter(x, df["exp"], s=80, label="Expected")
    ax.scatter(x, df["real"], s=80, label="Realized")
    ax.set_title("Expected vs Realized – Half-year Dumbbell (2023)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

def plot_half_opt3(df: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(df))
    ax.plot(x, df["exp"], marker="o", label="Expected")
    ax.plot(x, df["real"], marker="o", label="Realized")
    ax.set_title("Expected vs Realized – Half-year Lines (2023)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

# ---- Yearly – 3 options (single point) ----
def plot_yearly_opt1(df: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Expected", "Realized"]
    values = [df["exp"].iloc[0], df["real"].iloc[0]]
    ax.bar(labels, values, alpha=0.9)
    ax.set_title("Expected vs Realized – Yearly (2023)")
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    _save(fig, outpath)

def plot_yearly_opt2(df: pd.DataFrame, outpath: str):
    # Dumbbell-like single category
    fig, ax = plt.subplots(figsize=(6, 5))
    x = [0]
    ax.plot([0, 0], [df["exp"].iloc[0], df["real"].iloc[0]], color="#999", linewidth=3)
    ax.scatter(x, df["exp"], s=120, label="Expected")
    ax.scatter(x, df["real"], s=120, label="Realized")
    ax.set_title("Expected vs Realized – Yearly Dumbbell (2023)")
    ax.set_xticks([0]); ax.set_xticklabels([df["label"].iloc[0]])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

def plot_yearly_opt3(df: pd.DataFrame, outpath: str):
    # Bullet-style approximation: bar for realized with expected as a marker
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([0], [df["real"].iloc[0]], alpha=0.85, label="Realized")
    ax.scatter([0], [df["exp"].iloc[0]], s=140, marker="D", label="Expected")
    ax.set_title("Expected vs Realized – Yearly Bullet (2023)")
    ax.set_xticks([0]); ax.set_xticklabels([df["label"].iloc[0]])
    ax.set_ylabel("Return (%)")
    _percent_axis(ax)
    ax.legend()
    _save(fig, outpath)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Build EVR charts (Expected vs Realized) from backtester log.")
    ap.add_argument("--log", required=True, help="Path to saved backtester output (e.g., bt_2023.log)")
    ap.add_argument("--year", type=int, default=2023, help="Year to parse (default: 2023)")
    ap.add_argument("--prefix", default="EVR_2023", help="Filename prefix for saved charts")
    args = ap.parse_args()

    data = parse_backtester_log(args.log, year=args.year)

    # Monthly charts (3)
    if not data["monthly"].empty:
        plot_monthly_opt1(data["monthly"], f"{args.prefix}_monthly_opt1.png")
        plot_monthly_opt2(data["monthly"], f"{args.prefix}_monthly_opt2.png")
        plot_monthly_opt3(data["monthly"], f"{args.prefix}_monthly_opt3.png")
    else:
        print("[INFO] No monthly data found for year", args.year)

    # Quarterly charts (3)
    if not data["quarterly"].empty:
        plot_quarterly_opt1(data["quarterly"], f"{args.prefix}_quarterly_opt1.png")
        plot_quarterly_opt2(data["quarterly"], f"{args.prefix}_quarterly_opt2.png")
        plot_quarterly_opt3(data["quarterly"], f"{args.prefix}_quarterly_opt3.png")
    else:
        print("[INFO] No quarterly data found for year", args.year)

    # Half-year charts (3)
    if not data["half"].empty:
        plot_half_opt1(data["half"], f"{args.prefix}_half_opt1.png")
        plot_half_opt2(data["half"], f"{args.prefix}_half_opt2.png")
        plot_half_opt3(data["half"], f"{args.prefix}_half_opt3.png")
    else:
        print("[INFO] No half-year data found for year", args.year)

    # Yearly charts (3)
    if not data["yearly"].empty:
        plot_yearly_opt1(data["yearly"], f"{args.prefix}_yearly_opt1.png")
        plot_yearly_opt2(data["yearly"], f"{args.prefix}_yearly_opt2.png")
        plot_yearly_opt3(data["yearly"], f"{args.prefix}_yearly_opt3.png")
    else:
        print("[INFO] No yearly data found for year", args.year)

    print(f"Charts saved with prefix '{args.prefix}_*.png'")

if __name__ == "__main__":
    main()
