"""
visualize_intraday.py — Visualisation Intraday multi-horizon
─────────────────────────────────────────────────────────────
Produit 3 figures :
  1. MAE par horizon (Naïf vs XGBoost vs LightGBM)
  2. Amélioration relative vs naïf par horizon
  3. Courbes de prédiction sur une journée type

Prérequis :
  - models_saved/intraday_v3_metrics.json
  - data/processed/features_v3/test_intraday_v3.parquet
  - models_saved/xgb_id_v3_h1..h12.joblib + lgb_id_v3_h1..h12.joblib

Auteur : Marius Fabbri
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
MODELS_DIR  = SCRIPT_DIR / "models_saved"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    with open(MODELS_DIR / "intraday_v3_metrics.json") as f:
        return json.load(f)


def plot_mae_by_horizon(metrics):
    """Figure 1 : MAE par horizon pour les 3 modèles."""
    horizons = list(range(1, 13))
    minutes = [h * 15 for h in horizons]

    naive_mae = [metrics[f"h{h}"]["naive"]["MAE"] for h in horizons]
    xgb_mae   = [metrics[f"h{h}"]["xgboost"]["MAE"] for h in horizons]
    lgb_mae   = [metrics[f"h{h}"]["lightgbm"]["MAE"] for h in horizons]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(minutes, naive_mae, "o--", color="#95a5a6", linewidth=2, markersize=7, label="Naïf (charge J-1)")
    ax.plot(minutes, xgb_mae,   "s-",  color="#3498db", linewidth=2, markersize=7, label="XGBoost")
    ax.plot(minutes, lgb_mae,   "^-",  color="#2ecc71", linewidth=2, markersize=7, label="LightGBM")

    # Zone entre naïf et modèles
    ax.fill_between(minutes, naive_mae, xgb_mae, color="#3498db", alpha=0.1)

    ax.set_xlabel("Horizon de prédiction (minutes)", fontsize=12)
    ax.set_ylabel("MAE (charge normalisée)", fontsize=12)
    ax.set_title("Intraday — MAE par horizon de prédiction", fontsize=14, fontweight="bold")
    ax.set_xticks(minutes)
    ax.set_xticklabels([f"{m}'" for m in minutes])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.annotate(f"Δ = {naive_mae[0] - xgb_mae[0]:.3f}",
                xy=(minutes[0], (naive_mae[0] + xgb_mae[0]) / 2),
                fontsize=9, color="#3498db", ha="right", va="center",
                xytext=(-15, 0), textcoords="offset points")
    ax.annotate(f"Δ = {naive_mae[-1] - xgb_mae[-1]:.3f}",
                xy=(minutes[-1], (naive_mae[-1] + xgb_mae[-1]) / 2),
                fontsize=9, color="#3498db", ha="left", va="center",
                xytext=(10, 0), textcoords="offset points")

    plt.tight_layout()
    path = FIGURES_DIR / "intraday_mae_by_horizon.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_improvement_by_horizon(metrics):
    """Figure 2 : Amélioration relative (%) vs naïf par horizon."""
    horizons = list(range(1, 13))
    minutes = [h * 15 for h in horizons]

    naive_mae = [metrics[f"h{h}"]["naive"]["MAE"] for h in horizons]
    xgb_mae   = [metrics[f"h{h}"]["xgboost"]["MAE"] for h in horizons]
    lgb_mae   = [metrics[f"h{h}"]["lightgbm"]["MAE"] for h in horizons]

    xgb_pct = [100 * (n - x) / n for n, x in zip(naive_mae, xgb_mae)]
    lgb_pct = [100 * (n - l) / n for n, l in zip(naive_mae, lgb_mae)]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(horizons))
    width = 0.35

    bars1 = ax.bar(x - width/2, xgb_pct, width, color="#3498db", edgecolor="#2c3e50",
                   linewidth=0.5, label="XGBoost")
    bars2 = ax.bar(x + width/2, lgb_pct, width, color="#2ecc71", edgecolor="#2c3e50",
                   linewidth=0.5, label="LightGBM")

    # Valeurs sur les barres
    for bar, pct in zip(bars1, xgb_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=8, color="#3498db", fontweight="bold")
    for bar, pct in zip(bars2, lgb_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=8, color="#2ecc71", fontweight="bold")

    ax.set_xlabel("Horizon de prédiction", fontsize=12)
    ax.set_ylabel("Amélioration vs naïf (%)", fontsize=12)
    ax.set_title("Intraday — Gain relatif par horizon", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{m}'" for m in minutes])
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "intraday_improvement_by_horizon.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_summary_comparison(metrics):
    """Figure 3 : Tableau récapitulatif DA + Intraday."""
    # DA results (from model_da_v3.py output)
    da_models = {
        "Naïf J-1": 0.3334,
        "Oiken":    0.2307,
        "XGBoost":  0.2178,
        "LightGBM": 0.2136,
    }

    # Intraday moyennes
    horizons = list(range(1, 13))
    id_avg = {
        "Naïf J-1": np.mean([metrics[f"h{h}"]["naive"]["MAE"] for h in horizons]),
        "XGBoost":  np.mean([metrics[f"h{h}"]["xgboost"]["MAE"] for h in horizons]),
        "LightGBM": np.mean([metrics[f"h{h}"]["lightgbm"]["MAE"] for h in horizons]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # DA
    ax = axes[0]
    models = list(da_models.keys())
    maes = list(da_models.values())
    colors = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(models, maes, color=colors, edgecolor="#2c3e50", linewidth=0.5)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{mae:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("MAE")
    ax.set_title("Day-Ahead", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(maes) * 1.15)

    # Intraday
    ax2 = axes[1]
    models2 = list(id_avg.keys())
    maes2 = list(id_avg.values())
    colors2 = ["#95a5a6", "#3498db", "#2ecc71"]
    bars2 = ax2.bar(models2, maes2, color=colors2, edgecolor="#2c3e50", linewidth=0.5)
    for bar, mae in zip(bars2, maes2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{mae:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("MAE (moyenne 12 horizons)")
    ax2.set_title("Intraday (moyenne)", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, max(maes2) * 1.15)

    fig.suptitle("Comparaison globale — Day-Ahead vs Intraday", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "da_vs_intraday_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_rmse_by_horizon(metrics):
    """Figure 4 : RMSE par horizon."""
    horizons = list(range(1, 13))
    minutes = [h * 15 for h in horizons]

    naive_rmse = [metrics[f"h{h}"]["naive"]["RMSE"] for h in horizons]
    xgb_rmse   = [metrics[f"h{h}"]["xgboost"]["RMSE"] for h in horizons]
    lgb_rmse   = [metrics[f"h{h}"]["lightgbm"]["RMSE"] for h in horizons]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(minutes, naive_rmse, "o--", color="#95a5a6", linewidth=2, markersize=7, label="Naïf")
    ax.plot(minutes, xgb_rmse,   "s-",  color="#3498db", linewidth=2, markersize=7, label="XGBoost")
    ax.plot(minutes, lgb_rmse,   "^-",  color="#2ecc71", linewidth=2, markersize=7, label="LightGBM")
    ax.fill_between(minutes, naive_rmse, xgb_rmse, color="#3498db", alpha=0.1)

    ax.set_xlabel("Horizon de prédiction (minutes)", fontsize=12)
    ax.set_ylabel("RMSE (charge normalisée)", fontsize=12)
    ax.set_title("Intraday — RMSE par horizon de prédiction", fontsize=14, fontweight="bold")
    ax.set_xticks(minutes)
    ax.set_xticklabels([f"{m}'" for m in minutes])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "intraday_rmse_by_horizon.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def main():
    print("Chargement des métriques Intraday...")
    metrics = load_metrics()

    print("\n── MAE par horizon ──")
    plot_mae_by_horizon(metrics)

    print("\n── Amélioration relative ──")
    plot_improvement_by_horizon(metrics)

    print("\n── RMSE par horizon ──")
    plot_rmse_by_horizon(metrics)

    print("\n── Comparaison DA vs Intraday ──")
    plot_summary_comparison(metrics)

    print(f"\n✓ Toutes les figures sauvegardées dans {FIGURES_DIR}")


if __name__ == "__main__":
    main()