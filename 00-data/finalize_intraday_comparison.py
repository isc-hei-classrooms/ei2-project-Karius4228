"""
finalize_intraday_comparison.py — Tableau comparatif final Intraday
Auteur : Marius Fabbri

Charge tous les modèles sauvegardés (XGB h1..12, LGB h1..12, RF h1..12),
recalcule les métriques sur le test set, produit le tableau comparatif final
et sauvegarde compare_intraday_metrics.json.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import FEATURES_DIR, MODELS_DIR
from features_intraday import build_nwp_for_horizon, get_feature_columns, get_nwp_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HORIZON_MAX = 12
NWP_COLS    = get_nwp_columns()

# Mapping nom → préfixe fichier joblib
MODELS = {
    "XGBoost"      : "compare_xgboost_id_h",
    "LightGBM"     : "compare_lightgbm_id_h",
    "Random Forest": "compare_random_forest_id_h",
}

def build_X_for_horizon(df, base_features, h):
    df_h = build_nwp_for_horizon(df, h)
    feature_cols_h = [f"{c}_h{h}" if c in NWP_COLS else c for c in base_features]
    return df_h.select(feature_cols_h).to_numpy()

def compute_metrics(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mask = np.abs(yt) > 0.1
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.sum() > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n": int(valid.sum())}

def print_comparison(all_results, results_naive):
    models = list(all_results.keys())
    log.info("\n" + "═"*80)
    log.info("  TABLEAU COMPARATIF FINAL — TEST SET INTRADAY")
    log.info("═"*80)

    header = f"  {'Horizon':<10}"
    for name in ["Naïf"] + models:
        header += f"  {name+' MAE':>14}"
    log.info(header)
    log.info("  " + "─"*76)

    for h in range(1, HORIZON_MAX + 1):
        row = f"  t+{h:<8}"
        row += f"  {results_naive[h]['MAE']:>14.4f}"
        for name in models:
            row += f"  {all_results[name][h]['MAE']:>14.4f}"
        log.info(row)

    log.info("  " + "─"*76)

    # Moyennes
    naive_avg = float(np.mean([results_naive[h]["MAE"] for h in range(1, 13)]))
    row_avg   = f"  {'Moyenne':<10}  {naive_avg:>14.4f}"
    row_delta = f"  {'vs Naïf':<10}  {'—':>14}"
    for name in models:
        avg   = all_results[name]["avg"]["MAE"]
        delta = (1 - avg / naive_avg) * 100
        row_avg   += f"  {avg:>14.4f}"
        row_delta += f"  {delta:>+13.1f}%"
    log.info(row_avg)
    log.info(row_delta)
    log.info("═"*80)

def run():
    log.info("═"*60)
    log.info("FINALISATION COMPARAISON INTRADAY")
    log.info("═"*60)

    df_test = pl.read_parquet(FEATURES_DIR / "test_intraday_v2.parquet")
    df_train = pl.read_parquet(FEATURES_DIR / "train_intraday_v2.parquet")
    base_features = get_feature_columns(df_train)

    # Baseline naïve
    y_lag = df_test["load_lag_96"].to_numpy()
    results_naive = {}
    for h in range(1, HORIZON_MAX + 1):
        y_true = df_test[f"target_{h}"].to_numpy()
        results_naive[h] = compute_metrics(y_true, y_lag)
    log.info(f"Baseline naïve MAE moy. : {np.mean([r['MAE'] for r in results_naive.values()]):.4f}")

    # Chargement et évaluation de chaque modèle
    all_results = {}

    for model_name, prefix in MODELS.items():
        log.info(f"\n── Évaluation {model_name} ──────────────────────────────")
        results = {}
        missing = []

        for h in range(1, HORIZON_MAX + 1):
            path = MODELS_DIR / f"{prefix}{h}.joblib"
            if not path.exists():
                missing.append(h)
                log.warning(f"  ⚠ Manquant : {path.name}")
                continue

            model  = joblib.load(path)
            X_test = build_X_for_horizon(df_test, base_features, h)
            y_test = df_test[f"target_{h}"].to_numpy()
            y_pred = model.predict(X_test)
            m      = compute_metrics(y_test, y_pred)
            results[h] = m

            naive_mae = results_naive[h]["MAE"]
            delta     = (1 - m["MAE"] / naive_mae) * 100
            log.info(f"  t+{h:<2}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%  Δ={delta:+.1f}%")

        if missing:
            log.error(f"  ⚠ Horizons manquants pour {model_name} : {missing}")
            log.error(f"  Lance d'abord resume_rf_intraday.py")
            continue

        avg_mae  = float(np.mean([r["MAE"]  for r in results.values()]))
        avg_rmse = float(np.mean([r["RMSE"] for r in results.values()]))
        avg_mape = float(np.mean([r["MAPE"] for r in results.values() if not np.isnan(r["MAPE"])]))
        results["avg"] = {"MAE": avg_mae, "RMSE": avg_rmse, "MAPE": avg_mape}
        log.info(f"  Moyenne  MAE={avg_mae:.4f}  RMSE={avg_rmse:.4f}  MAPE={avg_mape:.2f}%")
        all_results[model_name] = results

    if len(all_results) < 3:
        log.error("Modèles incomplets — relance resume_rf_intraday.py d'abord")
        return

    # Tableau final
    print_comparison(all_results, results_naive)

    # Sauvegarde JSON
    def _serial(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        raise TypeError(type(obj))

    out = {
        name: {str(k): v for k, v in res.items()}
        for name, res in all_results.items()
    }
    out["naive"] = {str(k): v for k, v in results_naive.items()}

    with open(MODELS_DIR / "compare_intraday_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_serial)

    log.info(f"\n✓ Métriques sauvegardées : {MODELS_DIR / 'compare_intraday_metrics.json'}")

if __name__ == "__main__":
    run()