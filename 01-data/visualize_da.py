"""
visualize_da.py — Visualisation Charge réelle vs Oiken vs Modèles
─────────────────────────────────────────────────────────────────
Produit des graphiques comparatifs sur des semaines types.

Prérequis : exécuter model_da_v3.py d'abord (génère da_v3_predictions.joblib)

Auteur : Marius Fabbri
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timezone

MODELS_DIR = Path(__file__).resolve().parent / "models_saved"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions():
    """Charge les prédictions sauvegardées par model_da_v3.py."""
    return joblib.load(MODELS_DIR / "da_v3_predictions.joblib")


def plot_week(data, start_date, end_date, title, filename):
    """
    Trace une semaine : charge réelle vs Oiken vs XGBoost vs LightGBM.
    """
    timestamps = data["timestamps"]
    indices = [i for i, ts in enumerate(timestamps) if start_date <= ts < end_date]

    if not indices:
        print(f"  Aucune donnée pour {start_date} → {end_date}")
        return

    ts = [timestamps[i] for i in indices]
    y_real = [data["y_test"][i] for i in indices]
    y_oiken = [data["y_oiken"][i] for i in indices]
    y_xgb = [data["y_xgb"][i] for i in indices]
    y_lgb = [data["y_lgb"][i] for i in indices]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

    # ── Courbes de prédiction ──
    ax = axes[0]
    ax.plot(ts, y_real, color="#2c3e50", linewidth=1.5, label="Charge réelle", zorder=5)
    ax.plot(ts, y_oiken, color="#e74c3c", linewidth=1.2, alpha=0.8, label="Oiken (forecast)", linestyle="--")
    ax.plot(ts, y_xgb, color="#3498db", linewidth=1.2, alpha=0.8, label="XGBoost")
    ax.plot(ts, y_lgb, color="#2ecc71", linewidth=1.2, alpha=0.8, label="LightGBM")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Charge normalisée (z-score)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator())

    # ── Erreurs absolues ──
    ax2 = axes[1]
    err_oiken = np.abs(np.array(y_real) - np.array(y_oiken))
    err_xgb = np.abs(np.array(y_real) - np.array(y_xgb))
    err_lgb = np.abs(np.array(y_real) - np.array(y_lgb))

    ax2.fill_between(ts, err_oiken, color="#e74c3c", alpha=0.3, label=f"Oiken (MAE={np.mean(err_oiken):.3f})")
    ax2.plot(ts, err_xgb, color="#3498db", linewidth=0.8, alpha=0.8, label=f"XGBoost (MAE={np.mean(err_xgb):.3f})")
    ax2.plot(ts, err_lgb, color="#2ecc71", linewidth=0.8, alpha=0.8, label=f"LightGBM (MAE={np.mean(err_lgb):.3f})")

    ax2.set_ylabel("Erreur absolue")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())

    plt.tight_layout()
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {path}")


def plot_feature_importance(data, top_n=15):
    """Bar chart des top features XGBoost."""
    imp = data.get("xgb_importance", {})
    if not imp:
        print("  Pas de feature importance disponible")
        return

    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])[:top_n]
    names = [x[0] for x in sorted_imp][::-1]
    values = [x[1] for x in sorted_imp][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(names, values, color="#3498db", edgecolor="#2c3e50", linewidth=0.5)
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top 15 Features — XGBoost Day-Ahead", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "da_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {path}")


def plot_results_table(data):
    """Bar chart comparatif des MAE."""
    results = data["results"]
    models = list(results.keys())
    maes = [results[m]["MAE"] for m in models]

    colors = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71"][:len(models)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, maes, color=colors, edgecolor="#2c3e50", linewidth=0.5)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{mae:.4f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("MAE (charge normalisée)")
    ax.set_title("Comparaison Day-Ahead — MAE sur test set", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "da_mae_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {path}")


def main():
    print("Chargement des prédictions...")
    data = load_predictions()

    print("\n── Semaine hiver (janvier 2025) ──")
    plot_week(data,
              datetime(2025, 1, 13, tzinfo=timezone.utc),
              datetime(2025, 1, 20, tzinfo=timezone.utc),
              "Day-Ahead — Semaine hiver (13-19 jan 2025)",
              "da_predictions_winter.png")

    print("\n── Semaine été (juin 2025) ──")
    plot_week(data,
              datetime(2025, 6, 2, tzinfo=timezone.utc),
              datetime(2025, 6, 9, tzinfo=timezone.utc),
              "Day-Ahead — Semaine été (2-8 juin 2025)",
              "da_predictions_summer.png")

    print("\n── Semaine printemps (avril 2025) ──")
    plot_week(data,
              datetime(2025, 4, 7, tzinfo=timezone.utc),
              datetime(2025, 4, 14, tzinfo=timezone.utc),
              "Day-Ahead — Semaine printemps (7-13 avril 2025)",
              "da_predictions_spring.png")

    print("\n── Feature importance ──")
    plot_feature_importance(data)

    print("\n── Comparaison MAE ──")
    plot_results_table(data)

    print("\n✓ Toutes les figures sauvegardées dans figures/")


if __name__ == "__main__":
    main()
