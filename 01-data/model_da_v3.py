"""
model_da_v3.py — Modèles XGBoost + LightGBM Day-Ahead (v3)
────────────────────────────────────────────────────────────
Utilise les features v3 (shifts temporels corrigés).

Pipeline :
  1. Charge train_da_v3 / test_da_v3
  2. Baselines : Naïf J-1 + Oiken (correctement aligné)
  3. XGBoost + LightGBM (RandomizedSearchCV + TimeSeriesSplit)
  4. Tableau comparatif + Feature importance
  5. Sauvegarde modèles / params / métriques

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

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from features_da_v3 import get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
FEATURES_DIR = Path(__file__).resolve().parent / "data" / "processed" / "features_v3"
MODELS_DIR   = Path(__file__).resolve().parent / "models_saved"

N_ITER       = 40
CV_FOLDS     = 5
RANDOM_STATE = 42

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


# ─── Métriques ────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mask = np.abs(yt) > 0.1
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.sum() > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n": int(valid.sum())}


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_model_da():
    log.info("=" * 60)
    log.info("MODÈLES DAY-AHEAD (v3 — shifts temporels corrigés)")
    log.info("=" * 60)

    # ── 1. Chargement ─────────────────────────────────────────────────────
    df_train = pl.read_parquet(FEATURES_DIR / "train_da_v3.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_da_v3.parquet")
    log.info(f"Train: {df_train.shape} | Test: {df_test.shape}")

    feature_cols = get_feature_columns(df_train)
    log.info(f"{len(feature_cols)} features")

    X_train = df_train.select(feature_cols).to_numpy().astype(np.float32)
    y_train = df_train["target"].to_numpy().astype(np.float32)
    X_test  = df_test.select(feature_cols).to_numpy().astype(np.float32)
    y_test  = df_test["target"].to_numpy().astype(np.float32)

    all_metrics = {}

    # ── 2. Baseline naïve J-1 ─────────────────────────────────────────────
    y_naive = df_test["load_lag_1d"].to_numpy()
    all_metrics["Naïf J-1"] = compute_metrics(y_test, y_naive)

    # ── 3. Benchmark Oiken (correctement aligné) ─────────────────────────
    y_oiken = df_test["forecast_load_target"].to_numpy()
    all_metrics["Oiken"] = compute_metrics(y_test, y_oiken)

    log.info(f"Naïf J-1:  MAE={all_metrics['Naïf J-1']['MAE']:.4f}")
    log.info(f"Oiken:     MAE={all_metrics['Oiken']['MAE']:.4f}")

    # ── 4. XGBoost ────────────────────────────────────────────────────────
    log.info(f"\n── XGBoost ({N_ITER} iter, {CV_FOLDS} folds) ──")
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    t0 = time.time()

    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                          random_state=RANDOM_STATE, n_jobs=1),
        XGB_PARAM_GRID, n_iter=N_ITER, cv=tscv,
        scoring="neg_mean_absolute_error", n_jobs=-1,
        random_state=RANDOM_STATE, verbose=1,
    )
    xgb_search.fit(X_train, y_train)
    y_xgb = xgb_search.best_estimator_.predict(X_test)
    all_metrics["XGBoost"] = compute_metrics(y_test, y_xgb)
    log.info(f"  CV MAE: {-xgb_search.best_score_:.4f} | Test MAE: {all_metrics['XGBoost']['MAE']:.4f} ({time.time()-t0:.0f}s)")

    # ── 5. LightGBM ──────────────────────────────────────────────────────
    log.info(f"\n── LightGBM ({N_ITER} iter, {CV_FOLDS} folds) ──")
    t0 = time.time()

    lgb_search = RandomizedSearchCV(
        lgb.LGBMRegressor(objective="regression", random_state=RANDOM_STATE,
                           n_jobs=1, verbose=-1),
        LGB_PARAM_GRID, n_iter=N_ITER, cv=tscv,
        scoring="neg_mean_absolute_error", n_jobs=-1,
        random_state=RANDOM_STATE, verbose=1,
    )
    lgb_search.fit(X_train, y_train)
    y_lgb = lgb_search.best_estimator_.predict(X_test)
    all_metrics["LightGBM"] = compute_metrics(y_test, y_lgb)
    log.info(f"  CV MAE: {-lgb_search.best_score_:.4f} | Test MAE: {all_metrics['LightGBM']['MAE']:.4f} ({time.time()-t0:.0f}s)")

    # ── 6. Tableau comparatif ─────────────────────────────────────────────
    oiken_mae = all_metrics["Oiken"]["MAE"]
    log.info("\n" + "=" * 70)
    log.info(f"  {'Modèle':<20} {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}  {'vs Oiken':>10}")
    log.info("  " + "-" * 60)
    for label, m in all_metrics.items():
        mape_s = f"{m['MAPE']:.2f}%" if not np.isnan(m["MAPE"]) else "N/A"
        vs = "ref" if label == "Oiken" else f"{100*(oiken_mae - m['MAE'])/oiken_mae:+.1f}%"
        log.info(f"  {label:<20} {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {mape_s:>8}  {vs:>10}")
    log.info("=" * 70)

    # ── 7. Feature importance ─────────────────────────────────────────────
    booster = xgb_search.best_estimator_.get_booster()
    scores = booster.get_score(importance_type="gain")
    fname_map = {f"f{i}": name for i, name in enumerate(feature_cols)}
    named_scores = {fname_map.get(k, k): v for k, v in scores.items()}
    log.info("\n── Top 15 features XGBoost (gain) ──")
    for rank, (name, score) in enumerate(sorted(named_scores.items(), key=lambda x: -x[1])[:15], 1):
        log.info(f"  {rank:2d}. {name:<40} {score:>10.1f}")

    # ── 8. Sauvegarde ─────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_search.best_estimator_, MODELS_DIR / "xgb_da_v3.joblib")
    joblib.dump(lgb_search.best_estimator_, MODELS_DIR / "lgb_da_v3.joblib")

    with open(MODELS_DIR / "da_v3_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(MODELS_DIR / "da_v3_xgb_params.json", "w") as f:
        json.dump(xgb_search.best_params_, f, indent=2)
    with open(MODELS_DIR / "da_v3_lgb_params.json", "w") as f:
        json.dump(lgb_search.best_params_, f, indent=2)

    # Prédictions pour visualisation
    joblib.dump({
        "y_test": y_test, "y_naive": y_naive, "y_oiken": y_oiken,
        "y_xgb": y_xgb, "y_lgb": y_lgb,
        "timestamps": df_test["timestamp"].to_list(),
        "results": all_metrics,
        "feature_cols": feature_cols,
        "xgb_importance": named_scores,
    }, MODELS_DIR / "da_v3_predictions.joblib")

    log.info(f"\n✓ Modèles sauvegardés dans {MODELS_DIR}")
    return all_metrics


if __name__ == "__main__":
    run_model_da()
