"""
resume_rf_intraday.py — Reprise Random Forest Intraday depuis h2
Auteur : Marius Fabbri

Lance uniquement les horizons RF manquants (h2..h12).
h1 est déjà sauvegardé.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import FEATURES_DIR, MODELS_DIR
from features_intraday import build_nwp_for_horizon, get_feature_columns, get_nwp_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

N_ITER       = 50
CV_FOLDS     = 5
RANDOM_STATE = 42
HORIZON_MAX  = 12
NWP_COLS     = get_nwp_columns()

PARAM_GRID_RF = {
    "n_estimators"    : [200, 400, 600],
    "max_depth"       : [8, 12, 16, None],
    "max_features"    : [0.5, 0.7, 1.0, "sqrt"],
    "min_samples_leaf": [1, 5, 10],
    "max_samples"     : [0.7, 0.85, 1.0],
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

def run():
    log.info("═"*60)
    log.info("REPRISE RANDOM FOREST INTRADAY — h2 à h12")
    log.info("═"*60)

    df_train = pl.read_parquet(FEATURES_DIR / "train_intraday_v2.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_intraday_v2.parquet")
    base_features = get_feature_columns(df_train)
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    for h in range(2, HORIZON_MAX + 1):
        out_path = MODELS_DIR / f"compare_random_forest_id_h{h}.joblib"
        if out_path.exists():
            log.info(f"  t+{h} déjà sauvegardé — ignoré")
            continue

        log.info(f"\n── Random Forest t+{h} ({h*15}min) ──────────────────────")

        X_train = build_X_for_horizon(df_train, base_features, h)
        y_train = df_train[f"target_{h}"].to_numpy()
        X_test  = build_X_for_horizon(df_test,  base_features, h)
        y_test  = df_test[f"target_{h}"].to_numpy()

        valid_train = ~np.isnan(y_train)
        X_train, y_train = X_train[valid_train], y_train[valid_train]

        search = RandomizedSearchCV(
            estimator           = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            param_distributions = PARAM_GRID_RF,
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

        log.info(f"  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%  (CV={-search.best_score_:.4f})")
        joblib.dump(best, out_path)
        log.info(f"  ✓ Sauvegardé : {out_path.name}")

    log.info("\n✓ Random Forest h2..h12 terminé")

if __name__ == "__main__":
    run()