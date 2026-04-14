"""
compare_models_intraday.py — Comparaison multi-modèles Intraday
Auteur : Marius Fabbri

Protocole identique pour les 3 modèles :
  - RandomizedSearchCV + TimeSeriesSplit(5), scoring = neg_MAE
  - Données : train_intraday_v2.parquet / test_intraday_v2.parquet
  - Features : get_feature_columns() — 35 features, NWP remplacées par horizon
  - Baseline naïve J-1 (load_lag_96) incluse pour référence
  - 12 modèles par algorithme (un par horizon)

Modèles comparés :
  1. XGBoost
  2. LightGBM
  3. Random Forest
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
from features_intraday import (
    build_nwp_for_horizon,
    get_feature_columns,
    get_nwp_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

N_ITER       = 50
CV_FOLDS     = 5
RANDOM_STATE = 42
HORIZON_MAX  = 12
NWP_COLS     = get_nwp_columns()

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
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n": int(valid.sum())}


# ─── Construction de X pour un horizon donné ─────────────────────────────────

def build_X_for_horizon(df: pl.DataFrame, base_features: list, h: int) -> np.ndarray:
    df_h = build_nwp_for_horizon(df, h)
    feature_cols_h = [
        f"{c}_h{h}" if c in NWP_COLS else c
        for c in base_features
    ]
    return df_h.select(feature_cols_h).to_numpy()


# ─── Entraînement d'un modèle sur les 12 horizons ────────────────────────────

def train_model_all_horizons(
    name: str,
    make_estimator,
    param_grid: dict,
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    base_features: list,
    results_naive: dict,
) -> tuple[dict, dict]:

    log.info(f"\n{'═'*60}")
    log.info(f"  {name} — 12 horizons × {N_ITER} iter × {CV_FOLDS} folds")
    log.info(f"{'═'*60}")

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    results: dict = {}
    params:  dict = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for h in range(1, HORIZON_MAX + 1):
        X_train = build_X_for_horizon(df_train, base_features, h)
        y_train = df_train[f"target_{h}"].to_numpy()
        X_test  = build_X_for_horizon(df_test,  base_features, h)
        y_test  = df_test[f"target_{h}"].to_numpy()

        # Filtre NaN sur y_train
        valid_train = ~np.isnan(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        search = RandomizedSearchCV(
            estimator           = make_estimator(),
            param_distributions = param_grid,
            n_iter              = N_ITER,
            cv                  = tscv,
            scoring             = "neg_mean_absolute_error",
            n_jobs              = -1,
            random_state        = RANDOM_STATE,
            verbose             = 0,
        )
        search.fit(X_train, y_train)

        best   = search.best_estimator_
        y_pred = best.predict(X_test)
        m      = compute_metrics(y_test, y_pred)
        results[h] = m
        params[h]  = search.best_params_

        mae_naive = results_naive[h]["MAE"]
        delta     = (1 - m["MAE"] / mae_naive) * 100
        log.info(
            f"  t+{h:<2}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}"
            f"  MAPE={m['MAPE']:.2f}%  Δ={delta:+.1f}%"
            f"  (CV={-search.best_score_:.4f})"
        )

        # Sauvegarde modèle
        model_name = name.lower().replace(" ", "_")
        joblib.dump(best, MODELS_DIR / f"compare_{model_name}_id_h{h}.joblib")

    # Moyenne
    avg_mae  = float(np.mean([r["MAE"]  for r in results.values()]))
    avg_rmse = float(np.mean([r["RMSE"] for r in results.values()]))
    avg_mape = float(np.mean([r["MAPE"] for r in results.values()
                               if not np.isnan(r["MAPE"])]))
    results["avg"] = {"MAE": avg_mae, "RMSE": avg_rmse, "MAPE": avg_mape}
    log.info(
        f"  {'Moyenne':<5}  MAE={avg_mae:.4f}  RMSE={avg_rmse:.4f}"
        f"  MAPE={avg_mape:.2f}%"
    )
    return results, params


# ─── Tableau comparatif final ─────────────────────────────────────────────────

def print_comparison(all_results: dict, results_naive: dict) -> None:
    models = [k for k in all_results.keys()]

    log.info("\n" + "═" * 78)
    log.info("  TABLEAU COMPARATIF FINAL — TEST SET INTRADAY")
    log.info("═" * 78)

    # En-tête
    header = f"  {'Horizon':<10}"
    for name in models:
        header += f"  {name+' MAE':>12}"
    log.info(header)
    log.info("  " + "─" * 72)

    # Par horizon
    mae_naive_avg = float(np.mean([results_naive[h]["MAE"] for h in range(1, 13)]))
    for h in range(1, HORIZON_MAX + 1):
        row = f"  t+{h:<8}"
        for name in models:
            mae = all_results[name][h]["MAE"]
            row += f"  {mae:>12.4f}"
        log.info(row)

    log.info("  " + "─" * 72)

    # Moyennes
    row_avg   = f"  {'Moyenne':<10}"
    row_naive = f"  {'vs Naïf':<10}"
    for name in models:
        avg = all_results[name]["avg"]["MAE"]
        delta = (1 - avg / mae_naive_avg) * 100
        row_avg   += f"  {avg:>12.4f}"
        row_naive += f"  {delta:>+11.1f}%"
    log.info(row_avg)
    log.info(f"  {'Naïf moy.':<10}  {mae_naive_avg:>12.4f}")
    log.info(row_naive)
    log.info("═" * 78)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_comparison_intraday() -> dict:
    log.info("\n" + "═" * 60)
    log.info("COMPARAISON MULTI-MODÈLES — INTRADAY (v2)")
    log.info("═" * 60)

    df_train = pl.read_parquet(FEATURES_DIR / "train_intraday_v2.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_intraday_v2.parquet")
    log.info(f"Train : {df_train.shape} | Test : {df_test.shape}")

    base_features = get_feature_columns(df_train)
    log.info(f"{len(base_features)} features de base")

    # Baseline naïve
    log.info("\n── Baseline naïve J-1 (load_lag_96) ────────────────────────")
    y_lag = df_test["load_lag_96"].to_numpy()
    results_naive = {}
    for h in range(1, HORIZON_MAX + 1):
        y_true = df_test[f"target_{h}"].to_numpy()
        results_naive[h] = compute_metrics(y_true, y_lag)
    mae_naive_avg = np.mean([r["MAE"] for r in results_naive.values()])
    log.info(f"  MAE moyenne : {mae_naive_avg:.4f}")

    all_results: dict = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    res_xgb, params_xgb = train_model_all_horizons(
        name          = "XGBoost",
        make_estimator= lambda: xgb.XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        param_grid    = PARAM_GRID_XGB,
        df_train=df_train, df_test=df_test,
        base_features =base_features,
        results_naive =results_naive,
    )
    all_results["XGBoost"] = res_xgb

    # ── LightGBM ──────────────────────────────────────────────────────────────
    res_lgb, params_lgb = train_model_all_horizons(
        name          = "LightGBM",
        make_estimator= lambda: lgb.LGBMRegressor(
            objective="regression", random_state=RANDOM_STATE,
            n_jobs=1, verbose=-1,
        ),
        param_grid    = PARAM_GRID_LGB,
        df_train=df_train, df_test=df_test,
        base_features =base_features,
        results_naive =results_naive,
    )
    all_results["LightGBM"] = res_lgb

    # ── Random Forest ─────────────────────────────────────────────────────────
    res_rf, params_rf = train_model_all_horizons(
        name          = "Random Forest",
        make_estimator= lambda: RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        param_grid    = PARAM_GRID_RF,
        df_train=df_train, df_test=df_test,
        base_features =base_features,
        results_naive =results_naive,
    )
    all_results["Random Forest"] = res_rf

    # Tableau final
    print_comparison(all_results, results_naive)

    # Sauvegarde
    def _serial(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        raise TypeError(type(obj))

    out_metrics = {
        name: {str(k): v for k, v in res.items()}
        for name, res in all_results.items()
    }
    out_params = {
        "XGBoost"      : {str(k): v for k, v in params_xgb.items()},
        "LightGBM"     : {str(k): v for k, v in params_lgb.items()},
        "Random Forest": {str(k): v for k, v in params_rf.items()},
    }

    with open(MODELS_DIR / "compare_intraday_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2, default=_serial)
    with open(MODELS_DIR / "compare_intraday_params.json", "w", encoding="utf-8") as f:
        json.dump(out_params, f, indent=2, default=_serial)

    log.info(f"\n✓ Métriques  : {MODELS_DIR / 'compare_intraday_metrics.json'}")
    log.info(f"✓ Paramètres : {MODELS_DIR / 'compare_intraday_params.json'}")

    return all_results


if __name__ == "__main__":
    run_comparison_intraday()