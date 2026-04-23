"""
model_da_v4.py — Modèles XGBoost + LightGBM Day-Ahead (v4)
────────────────────────────────────────────────────────────
Différences vs v3 :
  1. La cible d'entraînement est `target_normalized`
     (charge divisée par le scaler PV mensuel)
  2. Les prédictions sont dénormalisées avant le calcul des métriques
     → MAE/RMSE/MAPE comparables avec Oiken et le naïf
  3. Le naïf et Oiken sont aussi exprimés en valeurs raw pour comparaison équitable
  4. Analyse saisonnière incluse (hiver vs été) pour voir si la normalisation
     a corrigé l'écart été constaté en v3

Auteur : Marius Fabbri
"""

import json, logging, time, sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from features_da_v4 import get_feature_columns
from clean_forecast import detect_frozen_forecast, filter_frozen_days
from denorm_utils import (
    denormalize_load,
    compute_metrics,
    split_by_season,
    validate_scaler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = SCRIPT_DIR / "data" / "processed" / "features_v4"
MODELS_DIR   = SCRIPT_DIR / "models_saved"

N_ITER       = 40
CV_FOLDS     = 5
RANDOM_STATE = 42

# Grilles hyperparamètres : identiques à v3 (on compare les architectures, pas les grilles)
XGB_PARAM_GRID = {
    "n_estimators":     [300, 500, 800],
    "max_depth":        [4, 6, 8],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [1.0, 2.0, 5.0],
}

LGB_PARAM_GRID = {
    "n_estimators":     [300, 500, 800],
    "max_depth":        [4, 6, 8, -1],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_samples":[5, 10, 20],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [1.0, 2.0, 5.0],
    "num_leaves":       [31, 63, 127],
}


def run_model_da():
    log.info("=" * 60)
    log.info("MODÈLES DAY-AHEAD v4 — avec normalisation PV")
    log.info("=" * 60)

    # ═══════════════════════════════════════════════════════════
    # 1. CHARGEMENT
    # ═══════════════════════════════════════════════════════════
    df_train = pl.read_parquet(FEATURES_DIR / "train_da_v4.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_da_v4.parquet")
    log.info(f"Train: {df_train.shape} | Test: {df_test.shape}")

    # ═══════════════════════════════════════════════════════════
    # 2. NETTOYAGE PRÉVISIONS OIKEN FIGÉES
    # ═══════════════════════════════════════════════════════════
    suspect_dates = detect_frozen_forecast(df_test, "forecast_load")
    df_test_clean = filter_frozen_days(df_test, suspect_dates)

    # ═══════════════════════════════════════════════════════════
    # 3. EXTRACTION ARRAYS
    # ═══════════════════════════════════════════════════════════
    feature_cols = get_feature_columns(df_train)
    log.info(f"{len(feature_cols)} features (dont pv_scaler + pv_scaler_target)")

    # Features (entraînement et test sur valeurs normalisées)
    X_train = df_train.select(feature_cols).to_numpy().astype(np.float32)
    X_test  = df_test_clean.select(feature_cols).to_numpy().astype(np.float32)

    # ── Cibles ──────────────────────────────────────────────────
    # y_train_norm : ce que le modèle optimise (signal sans dérive PV)
    y_train_norm = df_train["target_normalized"].to_numpy().astype(np.float32)

    # y_test_norm : pour diagnostic en espace normalisé
    y_test_norm  = df_test_clean["target_normalized"].to_numpy().astype(np.float32)

    # y_test_raw : valeurs réelles dans l'espace original → métriques finales
    y_test_raw   = df_test_clean["target_raw"].to_numpy().astype(np.float32)

    # ── Scaler cible (t+24h) ────────────────────────────────────
    # Nécessaire pour dénormaliser les prédictions :
    #   load_pred_raw[t] = load_pred_norm[t] × pv_scaler_target[t]
    scaler_target = df_test_clean["pv_scaler_target"].to_numpy().astype(np.float32)

    log.info(f"  y_train_norm : mean={y_train_norm.mean():.4f} std={y_train_norm.std():.4f}")
    log.info(f"  y_test_raw   : mean={y_test_raw.mean():.4f} std={y_test_raw.std():.4f}")

    # Validation du scaler
    validate_scaler(scaler_target, df_test_clean["timestamp"].to_list())

    all_metrics     = {}   # métriques dans l'espace raw (pour le rapport)
    all_metrics_norm = {}  # métriques dans l'espace normalisé (diagnostic)

    # ═══════════════════════════════════════════════════════════
    # 4. BASELINES — exprimées en valeurs RAW
    # ═══════════════════════════════════════════════════════════

    # Baseline naïve : load_lag_1d est déjà normalisé (calculé sur load_norm)
    # → on dénormalise avec le scaler cible pour revenir dans l'espace raw
    y_naive_norm = df_test_clean["load_lag_1d"].to_numpy()
    y_naive_raw  = denormalize_load(y_naive_norm, scaler_target)
    all_metrics["Naïf J-1"] = compute_metrics(y_test_raw, y_naive_raw, "Naïf J-1")

    # Oiken : déjà dans l'espace raw (forecast_load n'est pas normalisé)
    y_oiken = df_test_clean["forecast_load_target"].to_numpy()
    all_metrics["Oiken"] = compute_metrics(y_test_raw, y_oiken, "Oiken")

    # ═══════════════════════════════════════════════════════════
    # 5. ENTRAÎNEMENT XGBOOST
    # ═══════════════════════════════════════════════════════════
    log.info(f"\n── XGBoost ({N_ITER} iter, {CV_FOLDS} folds) ──")
    log.info("  Entraînement sur target_normalized (signal stationnaire)")
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    t0 = time.time()

    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        XGB_PARAM_GRID,
        n_iter=N_ITER, cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    xgb_search.fit(X_train, y_train_norm)

    # Prédiction en espace normalisé puis dénormalisation
    y_xgb_norm = xgb_search.best_estimator_.predict(X_test)
    y_xgb_raw  = denormalize_load(y_xgb_norm, scaler_target)

    all_metrics["XGBoost"]      = compute_metrics(y_test_raw, y_xgb_raw, "XGBoost")
    all_metrics_norm["XGBoost"] = compute_metrics(y_test_norm, y_xgb_norm, "XGBoost [norm]")
    log.info(f"  CV MAE (norm): {-xgb_search.best_score_:.4f} | Test MAE (raw): {all_metrics['XGBoost']['MAE']:.4f} ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════════════════
    # 6. ENTRAÎNEMENT LIGHTGBM
    # ═══════════════════════════════════════════════════════════
    log.info(f"\n── LightGBM ({N_ITER} iter, {CV_FOLDS} folds) ──")
    t0 = time.time()

    lgb_search = RandomizedSearchCV(
        lgb.LGBMRegressor(
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
        LGB_PARAM_GRID,
        n_iter=N_ITER, cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    lgb_search.fit(X_train, y_train_norm)

    y_lgb_norm = lgb_search.best_estimator_.predict(X_test)
    y_lgb_raw  = denormalize_load(y_lgb_norm, scaler_target)

    all_metrics["LightGBM"]      = compute_metrics(y_test_raw, y_lgb_raw, "LightGBM")
    all_metrics_norm["LightGBM"] = compute_metrics(y_test_norm, y_lgb_norm, "LightGBM [norm]")
    log.info(f"  CV MAE (norm): {-lgb_search.best_score_:.4f} | Test MAE (raw): {all_metrics['LightGBM']['MAE']:.4f} ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════════════════
    # 7. TABLEAU RÉCAPITULATIF
    # ═══════════════════════════════════════════════════════════
    oiken_mae = all_metrics["Oiken"]["MAE"]
    log.info("\n" + "=" * 75)
    log.info(f"  {'Modèle':<20} {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}  {'Biais':>8}  {'vs Oiken':>10}")
    log.info("  " + "-" * 68)
    for label, m in all_metrics.items():
        mape_s = f"{m['MAPE']:.2f}%" if not np.isnan(m["MAPE"]) else "N/A"
        vs = "ref" if label == "Oiken" else f"{100*(oiken_mae - m['MAE'])/oiken_mae:+.1f}%"
        log.info(f"  {label:<20} {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {mape_s:>8}  {m['bias']:>+8.4f}  {vs:>10}")
    log.info("=" * 75)

    # ═══════════════════════════════════════════════════════════
    # 8. ANALYSE SAISONNIÈRE
    # ═══════════════════════════════════════════════════════════
    timestamps = df_test_clean["timestamp"].to_list()

    log.info("\n── Analyse saisonnière (Hiver nov-mars / Été avr-oct) ──")
    seasonal = {
        "Naïf":     split_by_season(timestamps, y_test_raw, y_naive_raw),
        "Oiken":    split_by_season(timestamps, y_test_raw, y_oiken),
        "XGBoost":  split_by_season(timestamps, y_test_raw, y_xgb_raw),
        "LightGBM": split_by_season(timestamps, y_test_raw, y_lgb_raw),
    }

    log.info(f"\n  {'Modèle':<12} {'Hiver MAE':>10} {'Été MAE':>10} {'Δ Hiver':>10} {'Δ Été':>10}")
    log.info("  " + "-" * 52)
    oiken_w = seasonal["Oiken"]["winter"]["MAE"]
    oiken_s = seasonal["Oiken"]["summer"]["MAE"]
    for name, s in seasonal.items():
        w = s["winter"]["MAE"]
        su = s["summer"]["MAE"]
        dw = "ref" if name == "Oiken" else f"{100*(oiken_w - w)/oiken_w:+.1f}%"
        ds = "ref" if name == "Oiken" else f"{100*(oiken_s - su)/oiken_s:+.1f}%"
        log.info(f"  {name:<12} {w:>10.4f} {su:>10.4f} {dw:>10} {ds:>10}")

    # ═══════════════════════════════════════════════════════════
    # 9. FEATURE IMPORTANCE
    # ═══════════════════════════════════════════════════════════
    booster = xgb_search.best_estimator_.get_booster()
    scores  = booster.get_score(importance_type="gain")
    fname_map    = {f"f{i}": name for i, name in enumerate(feature_cols)}
    named_scores = {fname_map.get(k, k): v for k, v in scores.items()}

    log.info("\n── Top 15 features (gain XGBoost) ──")
    for rank, (name, score) in enumerate(sorted(named_scores.items(), key=lambda x: -x[1])[:15], 1):
        log.info(f"  {rank:2d}. {name:<45} {score:>10.1f}")

    log.info("\n  Note : si pv_scaler_target apparaît dans le top 5,")
    log.info("  la normalisation est bien utilisée par le modèle.")

    # ═══════════════════════════════════════════════════════════
    # 10. SAUVEGARDE
    # ═══════════════════════════════════════════════════════════
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_search.best_estimator_, MODELS_DIR / "xgb_da_v4.joblib")
    joblib.dump(lgb_search.best_estimator_, MODELS_DIR / "lgb_da_v4.joblib")

    with open(MODELS_DIR / "da_v4_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(MODELS_DIR / "da_v4_metrics_norm.json", "w") as f:
        json.dump(all_metrics_norm, f, indent=2)
    with open(MODELS_DIR / "da_v4_xgb_params.json", "w") as f:
        json.dump(xgb_search.best_params_, f, indent=2)
    with open(MODELS_DIR / "da_v4_lgb_params.json", "w") as f:
        json.dump(lgb_search.best_params_, f, indent=2)

    joblib.dump({
        # Valeurs raw (espace original — pour graphiques et métriques)
        "y_test":   y_test_raw,
        "y_naive":  y_naive_raw,
        "y_oiken":  y_oiken,
        "y_xgb":    y_xgb_raw,
        "y_lgb":    y_lgb_raw,
        # Valeurs normalisées (pour diagnostic)
        "y_test_norm":  y_test_norm,
        "y_xgb_norm":   y_xgb_norm,
        "y_lgb_norm":   y_lgb_norm,
        "scaler_target": scaler_target,
        # Méta
        "timestamps":   df_test_clean["timestamp"].to_list(),
        "results":      all_metrics,
        "results_norm": all_metrics_norm,
        "seasonal":     seasonal,
        "feature_cols": feature_cols,
        "xgb_importance": named_scores,
    }, MODELS_DIR / "da_v4_predictions.joblib")

    log.info(f"\n✓ Sauvegardé dans {MODELS_DIR}")
    return all_metrics


if __name__ == "__main__":
    run_model_da()
