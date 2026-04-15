"""
visualize_oiken_vs_reel.py — Charge réelle vs prévision Oiken
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timezone, timedelta

SCRIPT_DIR   = Path(__file__).resolve().parent
FEATURES_DIR = SCRIPT_DIR / "data" / "processed" / "features_v3"
FIGURES_DIR  = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    te = pl.read_parquet(FEATURES_DIR / "test_da_v3.parquet")

    start = datetime(2025, 9, 10, tzinfo=timezone.utc)
    end   = datetime(2025, 9, 18, tzinfo=timezone.utc)
    week  = te.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))

    ts   = week["timestamp"].to_list()
    load = week["load"].to_numpy()
    fc   = week["forecast_load"].to_numpy()

    mae_oiken = np.mean(np.abs(load - fc))
    bias = np.mean(fc - load)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(ts, load, color="#2c3e50", linewidth=1.5, label="Charge réelle", zorder=5)
    ax.plot(ts, fc, color="#e74c3c", linewidth=1.2, alpha=0.85, label="Oiken (forecast)", linestyle="--")
    for d in range(8):
        ns = start + timedelta(days=d, hours=21)
        ax.axvspan(ns, ns + timedelta(hours=9), color="#ecf0f1", alpha=0.3, zorder=0)
    ax.set_title("Charge réelle vs prévision Oiken — 10-17 septembre 2025", fontsize=14, fontweight="bold")
    ax.set_ylabel("Charge normalisée (z-score)", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.text(0.02, 0.95, f"MAE Oiken = {mae_oiken:.4f}\nBiais = {bias:+.4f}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax2 = axes[1]
    err = fc - load
    ax2.fill_between(ts, err, 0, where=(err >= 0), color="#e74c3c", alpha=0.4, label="Oiken surestime")
    ax2.fill_between(ts, err, 0, where=(err < 0), color="#3498db", alpha=0.4, label="Oiken sous-estime")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axhline(y=bias, color="#e74c3c", linewidth=1, linestyle=":", label=f"Biais moyen ({bias:+.3f})")
    ax2.set_ylabel("Erreur (Oiken − réel)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "oiken_vs_reel_sept2025.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {FIGURES_DIR / 'oiken_vs_reel_sept2025.png'}")


if __name__ == "__main__":
    main()
