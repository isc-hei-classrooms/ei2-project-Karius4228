"""
compare_models_da.py — Comparaison multi-modèles Day-Ahead
Auteur : Marius Fabbri

Protocole identique pour les 3 modèles :
  - RandomizedSearchCV + TimeSeriesSplit(5), scoring = neg_MAE
  - Données : train_da_v2.parquet / test_da_v2.parquet (pas _norm)
  - Features : get_feature_columns() — 33 features, anti-leakage garanti

Modèles comparés :
  1. Baseline naïve J-1
  2. Oiken forecast_load (benchmark indicatif — intégrité temporelle non garantie)
  3. XGBoost
  4. LightGBM
  5. Random Forest
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import FEATURES_DIR, MODELS_DIR
from features_da import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

N_ITER       = 50
CV_FOLDS     = 5
RANDOM_STATE = 42

# ─── Grilles d'hyperparamètres ────────────────────────────────────────────────

PARAM_GRID_XGB = {
    "n_estimators"     : [300, 500, 800],
    "max_depth"        : [4, 6, 8],
    "learning_rate"    : [0.01, 0.05, 0.1],
    "subsample"        : [0.7, 0.85, 1.0],
    "colsample_bytree" : [0.7, 0.85, 1.0],
    "min_child_weight" : [1, 5, 10],
    "reg_alpha"        : [0, 0.1, 1.0],
    "reg_lambda"       : [1.0, 2.0, 5.0],
}

PARAM_GRID_LGB = {
    "n_estimators"     : [300, 500, 800],
    "max_depth"        : [4, 6, 8, -1],
    "learning_rate"    : [0.01, 0.05, 0.1],
    "subsample"        : [0.7, 0.85, 1.0],
    "colsample_bytree" : [0.7, 0.85, 1.0],
    "min_child_samples": [10, 20, 50],
    "reg_alpha"        : [0, 0.1, 1.0],
    "reg_lambda"       : [1.0, 2.0, 5.0],
    "num_leaves"       : [31, 63, 127],
}

PARAM_GRID_RF = {
    "n_estimators"    : [200, 400, 600],
    "max_depth"       : [8, 12, 16, None],
    "max_features"    : [0.5, 0.7, 1.0, "sqrt"],
    "min_samples_leaf": [1, 5, 10],
    "max_samples"     : [0.7, 0.85, 1.0],
}


# ─── Métriques ────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mask = np.abs(yt) > 0.1
    mape = float(
        np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100
    ) if mask.sum() > 0 else float("nan")
    return {
        "MAE" : mae,
        "RMSE": rmse,
        "MAPE": mape,
        "n"   : int(valid.sum()),
    }


# ─── Entraînement générique ───────────────────────────────────────────────────

def train_and_evaluate(
    name: str,
    estimator,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple:
    log.info(f"\n── {name} — RandomizedSearchCV ({N_ITER} iter, {CV_FOLDS} folds) ──")
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    search = RandomizedSearchCV(
        estimator           = estimator,
        param_distributions = param_grid,
        n_iter              = N_ITER,
        cv                  = tscv,
        scoring             = "neg_mean_absolute_error",
        n_jobs              = -1,
        random_state        = RANDOM_STATE,
        verbose             = 1,
    )
    search.fit(X_train, y_train)

    log.info(f"  Meilleurs paramètres : {search.best_params_}")
    log.info(f"  MAE CV moyen (train) : {-search.best_score_:.4f}")

    best   = search.best_estimator_
    y_pred = best.predict(X_test)
    m      = compute_metrics(y_test, y_pred)
    log.info(
        f"  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%"
    )
    return m, best, search.best_params_


# ─── Tableau comparatif ───────────────────────────────────────────────────────

def print_comparison(results: dict) -> None:
    log.info("\n" + "═" * 72)
    log.info("  TABLEAU COMPARATIF FINAL — TEST SET DA")
    log.info("═" * 72)
    log.info(f"  {'Modèle':<32} {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}  {'vs Naïf':>8}")
    log.info("  " + "─" * 66)

    mae_naive = results["Naïf J-1"]["MAE"]
    for label, m in results.items():
        if label == "Naïf J-1":
            delta = "—"
        else:
            delta = f"{(1 - m['MAE']/mae_naive)*100:+.1f}%"
        mape_s = f"{m['MAPE']:>7.2f}%" if not np.isnan(m["MAPE"]) else "     N/A"
        note = " ⚠" if "indicatif" in label else ""
        log.info(
            f"  {label:<32} {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {mape_s}  {delta:>8}{note}"
        )

    log.info("═" * 72)
    log.info("  ⚠ Oiken : intégrité temporelle de forecast_load non garantie")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_comparison_da() -> dict:
    log.info("\n" + "═" * 60)
    log.info("COMPARAISON MULTI-MODÈLES — DAY-AHEAD (v2)")
    log.info("═" * 60)

    # ── Chargement ────────────────────────────────────────────────────────────
    df_train = pl.read_parquet(FEATURES_DIR / "train_da_v2.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_da_v2.parquet")
    log.info(f"Train : {df_train.shape} | Test : {df_test.shape}")

    feature_cols = get_feature_columns(df_train)
    log.info(f"{len(feature_cols)} features")

    X_train = df_train.select(feature_cols).to_numpy()
    y_train = df_train["target"].to_numpy()
    X_test  = df_test.select(feature_cols).to_numpy()
    y_test  = df_test["target"].to_numpy()

    all_metrics: dict = {}
    all_params:  dict = {}

    # ── Baseline naïve J-1 ────────────────────────────────────────────────────
    log.info("\n── Baseline naïve J-1 (load_lag_1d) ────────────────────────")
    y_naive = df_test["load_lag_1d"].to_numpy()
    m_naive = compute_metrics(y_test, y_naive)
    log.info(f"  MAE={m_naive['MAE']:.4f}  RMSE={m_naive['RMSE']:.4f}  MAPE={m_naive['MAPE']:.2f}%")
    all_metrics["Naïf J-1"] = m_naive

    # ── Forecast Oiken (benchmark indicatif) ──────────────────────────────────
    log.info("\n── Forecast Oiken (forecast_load) ───────────────────────────")
    log.info("  ⚠ Benchmark indicatif — intégrité temporelle non garantie")
    y_oiken = df_test["forecast_load"].to_numpy()
    valid   = ~(np.isnan(y_oiken) | np.isnan(y_test))
    if valid.sum() < len(y_test):
        log.warning(f"  {len(y_test) - valid.sum()} NaN filtrés dans forecast_load")
    m_oiken = compute_metrics(y_test[valid], y_oiken[valid])
    log.info(f"  MAE={m_oiken['MAE']:.4f}  RMSE={m_oiken['RMSE']:.4f}  MAPE={m_oiken['MAPE']:.2f}%")
    all_metrics["Oiken (indicatif) ⚠"] = m_oiken

    # ── 1. XGBoost ────────────────────────────────────────────────────────────
    m_xgb, model_xgb, params_xgb = train_and_evaluate(
        name      = "XGBoost",
        estimator = xgb.XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        param_grid = PARAM_GRID_XGB,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
    )
    all_metrics["XGBoost"] = m_xgb
    all_params["XGBoost"]  = params_xgb
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_xgb, MODELS_DIR / "compare_xgb_da.joblib")

    # ── 2. LightGBM ───────────────────────────────────────────────────────────
    m_lgb, model_lgb, params_lgb = train_and_evaluate(
        name      = "LightGBM",
        estimator = lgb.LGBMRegressor(
            objective="regression", random_state=RANDOM_STATE,
            n_jobs=1, verbose=-1,
        ),
        param_grid = PARAM_GRID_LGB,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
    )
    all_metrics["LightGBM"] = m_lgb
    all_params["LightGBM"]  = params_lgb
    joblib.dump(model_lgb, MODELS_DIR / "compare_lgb_da.joblib")

    # ── 3. Random Forest ──────────────────────────────────────────────────────
    m_rf, model_rf, params_rf = train_and_evaluate(
        name      = "Random Forest",
        estimator = RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        param_grid = PARAM_GRID_RF,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
    )
    all_metrics["Random Forest"] = m_rf
    all_params["Random Forest"]  = params_rf
    joblib.dump(model_rf, MODELS_DIR / "compare_rf_da.joblib")

    # ── Tableau comparatif ────────────────────────────────────────────────────
    print_comparison(all_metrics)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    with open(MODELS_DIR / "compare_da_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    with open(MODELS_DIR / "compare_da_params.json", "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2)

    log.info(f"\n✓ Métriques : {MODELS_DIR / 'compare_da_metrics.json'}")
    log.info(f"✓ Paramètres: {MODELS_DIR / 'compare_da_params.json'}")

    return all_metrics


if __name__ == "__main__":
    run_comparison_da()