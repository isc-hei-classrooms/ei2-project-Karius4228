"""
visualize_intraday.py — Visualisation Intraday multi-horizon
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
MODELS_DIR  = SCRIPT_DIR / "models_saved"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    with open(MODELS_DIR / "intraday_v3_metrics.json") as f:
        return json.load(f)


def plot_mae_by_horizon(metrics):
    horizons = list(range(1, 13))
    minutes  = [h * 15 for h in horizons]
    naive = [metrics[f"h{h}"]["naive"]["MAE"] for h in horizons]
    xgb   = [metrics[f"h{h}"]["xgboost"]["MAE"] for h in horizons]
    lgb   = [metrics[f"h{h}"]["lightgbm"]["MAE"] for h in horizons]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(minutes, naive, "o--", color="#95a5a6", linewidth=2, markersize=7, label="Naïf (charge J-1)")
    ax.plot(minutes, xgb,   "s-",  color="#3498db", linewidth=2, markersize=7, label="XGBoost")
    ax.plot(minutes, lgb,   "^-",  color="#2ecc71", linewidth=2, markersize=7, label="LightGBM")
    ax.fill_between(minutes, naive, xgb, color="#3498db", alpha=0.1)
    ax.set_xlabel("Horizon (minutes)", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title("Intraday — MAE par horizon", fontsize=14, fontweight="bold")
    ax.set_xticks(minutes)
    ax.set_xticklabels([f"{m}'" for m in minutes])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intraday_mae_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ intraday_mae_by_horizon.png")


def plot_improvement(metrics):
    horizons = list(range(1, 13))
    minutes  = [h * 15 for h in horizons]
    naive = [metrics[f"h{h}"]["naive"]["MAE"] for h in horizons]
    xgb   = [metrics[f"h{h}"]["xgboost"]["MAE"] for h in horizons]
    lgb   = [metrics[f"h{h}"]["lightgbm"]["MAE"] for h in horizons]
    xgb_pct = [100*(n-x)/n for n,x in zip(naive, xgb)]
    lgb_pct = [100*(n-l)/n for n,l in zip(naive, lgb)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(horizons))
    w = 0.35
    b1 = ax.bar(x - w/2, xgb_pct, w, color="#3498db", edgecolor="#2c3e50", linewidth=0.5, label="XGBoost")
    b2 = ax.bar(x + w/2, lgb_pct, w, color="#2ecc71", edgecolor="#2c3e50", linewidth=0.5, label="LightGBM")
    for bar, pct in zip(b1, xgb_pct):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{pct:.1f}%", ha="center", fontsize=8, color="#3498db", fontweight="bold")
    for bar, pct in zip(b2, lgb_pct):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{pct:.1f}%", ha="center", fontsize=8, color="#2ecc71", fontweight="bold")
    ax.set_xlabel("Horizon", fontsize=12)
    ax.set_ylabel("Amélioration vs naïf (%)", fontsize=12)
    ax.set_title("Intraday — Gain relatif par horizon", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{m}'" for m in minutes])
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intraday_improvement_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ intraday_improvement_by_horizon.png")


def plot_rmse(metrics):
    horizons = list(range(1, 13))
    minutes  = [h * 15 for h in horizons]
    naive = [metrics[f"h{h}"]["naive"]["RMSE"] for h in horizons]
    xgb   = [metrics[f"h{h}"]["xgboost"]["RMSE"] for h in horizons]
    lgb   = [metrics[f"h{h}"]["lightgbm"]["RMSE"] for h in horizons]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(minutes, naive, "o--", color="#95a5a6", linewidth=2, markersize=7, label="Naïf")
    ax.plot(minutes, xgb,   "s-",  color="#3498db", linewidth=2, markersize=7, label="XGBoost")
    ax.plot(minutes, lgb,   "^-",  color="#2ecc71", linewidth=2, markersize=7, label="LightGBM")
    ax.fill_between(minutes, naive, xgb, color="#3498db", alpha=0.1)
    ax.set_xlabel("Horizon (minutes)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Intraday — RMSE par horizon", fontsize=14, fontweight="bold")
    ax.set_xticks(minutes)
    ax.set_xticklabels([f"{m}'" for m in minutes])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intraday_rmse_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ intraday_rmse_by_horizon.png")


def plot_comparison(metrics):
    da_models = {"Naïf J-1": 0.3347, "Oiken": 0.2182, "XGBoost": 0.2188, "LightGBM": 0.2147}
    horizons = list(range(1, 13))
    id_avg = {
        "Naïf J-1": np.mean([metrics[f"h{h}"]["naive"]["MAE"] for h in horizons]),
        "XGBoost":  np.mean([metrics[f"h{h}"]["xgboost"]["MAE"] for h in horizons]),
        "LightGBM": np.mean([metrics[f"h{h}"]["lightgbm"]["MAE"] for h in horizons]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, data, colors in [
        (axes[0], "Day-Ahead (nettoyé)", da_models, ["#95a5a6","#e74c3c","#3498db","#2ecc71"]),
        (axes[1], "Intraday (moyenne)", id_avg, ["#95a5a6","#3498db","#2ecc71"]),
    ]:
        names, vals = list(data.keys()), list(data.values())
        bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="#2c3e50", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("MAE")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals)*1.15)

    fig.suptitle("Comparaison globale — DA vs Intraday", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "da_vs_intraday_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ da_vs_intraday_comparison.png")


def main():
    print("Chargement métriques Intraday...")
    metrics = load_metrics()
    print("\n── MAE par horizon ──")
    plot_mae_by_horizon(metrics)
    print("\n── Amélioration relative ──")
    plot_improvement(metrics)
    print("\n── RMSE par horizon ──")
    plot_rmse(metrics)
    print("\n── Comparaison DA vs Intraday ──")
    plot_comparison(metrics)
    print(f"\n✓ Figures dans {FIGURES_DIR}")


if __name__ == "__main__":
    main()
