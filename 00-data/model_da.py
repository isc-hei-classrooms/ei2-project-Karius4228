"""
model_da.py — Modèle XGBoost Day-Ahead (v2)
Auteur : Marius Fabbri

Pipeline :
  1. Charge train_da_v2 / test_da_v2 (pas _norm : XGBoost insensible à l'échelle)
  2. Construit X via get_feature_columns() — 33 features, anti-leakage garanti
  3. RandomizedSearchCV + TimeSeriesSplit(5) — pas de shuffle
  4. Évalue sur test set
  5. Tableau comparatif : Naïf J-1 | XGBoost DA
  6. Feature importance par gain
  7. Sauvegarde modèle / params / métriques
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import FEATURES_DIR, MODELS_DIR
from features_da import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── Hyperparamètres ──────────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators"     : [300, 500, 800],
    "max_depth"        : [4, 6, 8],
    "learning_rate"    : [0.01, 0.05, 0.1],
    "subsample"        : [0.7, 0.85, 1.0],
    "colsample_bytree" : [0.7, 0.85, 1.0],
    "min_child_weight" : [1, 5, 10],
    "reg_alpha"        : [0, 0.1, 1.0],
    "reg_lambda"       : [1.0, 2.0, 5.0],
}

N_ITER       = 50
CV_FOLDS     = 5
RANDOM_STATE = 42


# ─── Métriques ────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    MAE, RMSE, MAPE.
    MAPE calculé uniquement sur |y_true| > 0.1 — charge en z-score,
    valeurs nocturnes proches de 0 rendent la division instable.
    """
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
        "n_mape_excluded": int((~mask).sum()),
    }


# ─── Tableau comparatif ───────────────────────────────────────────────────────

def print_comparison(results: dict) -> None:
    log.info("\n" + "═" * 62)
    log.info("  TABLEAU COMPARATIF — TEST SET DA")
    log.info("═" * 62)
    log.info(f"  {'Modèle':<28} {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}")
    log.info("  " + "─" * 56)
    for label, m in results.items():
        mape_s = f"{m['MAPE']:>7.2f}%" if not np.isnan(m["MAPE"]) else "     N/A"
        log.info(
            f"  {label:<28} {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {mape_s}"
        )
    log.info("═" * 62)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_model_da() -> tuple:
    log.info("\n" + "═" * 60)
    log.info("MODÈLE XGBOOST — DAY-AHEAD (v2)")
    log.info("═" * 60)

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    df_train = pl.read_parquet(FEATURES_DIR / "train_da_v2.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_da_v2.parquet")
    log.info(f"Train : {df_train.shape} | Test : {df_test.shape}")

    # ── 2. Matrices X / y ─────────────────────────────────────────────────────
    feature_cols = get_feature_columns(df_train)
    log.info(f"{len(feature_cols)} features sélectionnées")

    X_train = df_train.select(feature_cols).to_numpy()
    y_train = df_train["target"].to_numpy()
    X_test  = df_test.select(feature_cols).to_numpy()
    y_test  = df_test["target"].to_numpy()

    all_metrics: dict = {}

    # ── 3. Baseline naïve J-1 ─────────────────────────────────────────────────
    log.info("\n── Baseline naïve J-1 (load_lag_1d) ────────────────────────")
    y_naive  = df_test["load_lag_1d"].to_numpy()
    m_naive  = compute_metrics(y_test, y_naive)
    log.info(
        f"  MAE={m_naive['MAE']:.4f}  "
        f"RMSE={m_naive['RMSE']:.4f}  "
        f"MAPE={m_naive['MAPE']:.2f}%"
    )
    all_metrics["Naïf J-1"] = m_naive

    # ── 4. Entraînement XGBoost ───────────────────────────────────────────────
    log.info(
        f"\n── XGBoost RandomizedSearchCV "
        f"({N_ITER} iter, {CV_FOLDS} folds) ──────────"
    )
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    base_model = xgb.XGBRegressor(
        objective    = "reg:squarederror",
        tree_method  = "hist",
        random_state = RANDOM_STATE,
        n_jobs       = 1,            # parallélisme délégué à RandomizedSearchCV
    )

    search = RandomizedSearchCV(
        estimator           = base_model,
        param_distributions = PARAM_GRID,
        n_iter              = N_ITER,
        cv                  = tscv,
        scoring             = "neg_mean_absolute_error",
        n_jobs              = -1,
        random_state        = RANDOM_STATE,
        verbose             = 1,
    )
    search.fit(X_train, y_train)

    log.info(f"\n  Meilleurs paramètres : {search.best_params_}")
    log.info(f"  MAE CV moyen (train) : {-search.best_score_:.4f}")

    # ── 5. Évaluation test set ────────────────────────────────────────────────
    best_model = search.best_estimator_
    y_pred     = best_model.predict(X_test)
    m_xgb      = compute_metrics(y_test, y_pred)
    log.info(
        f"\n  MAE={m_xgb['MAE']:.4f}  "
        f"RMSE={m_xgb['RMSE']:.4f}  "
        f"MAPE={m_xgb['MAPE']:.2f}%"
    )
    all_metrics["XGBoost DA"] = m_xgb

    # ── 6. Tableau comparatif ─────────────────────────────────────────────────
    print_comparison(all_metrics)

    # ── 7. Feature importance par gain ────────────────────────────────────────
    log.info("\n── Top 15 features (importance par gain) ────────────────────")
    booster      = best_model.get_booster()
    scores       = booster.get_score(importance_type="gain")
    fname_map    = {f"f{i}": name for i, name in enumerate(feature_cols)}
    named_scores = {fname_map.get(k, k): v for k, v in scores.items()}
    top15 = sorted(named_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    for rank, (name, score) in enumerate(top15, 1):
        log.info(f"  {rank:2d}. {name:<40} {score:>10.1f}")

    # ── 8. Sauvegarde ─────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS_DIR / "xgb_da.joblib")

    with open(MODELS_DIR / "xgb_da_best_params.json", "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, indent=2)

    with open(MODELS_DIR / "xgb_da_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    log.info(f"\n✓ Modèle     : {MODELS_DIR / 'xgb_da.joblib'}")
    log.info(f"✓ Paramètres : {MODELS_DIR / 'xgb_da_best_params.json'}")
    log.info(f"✓ Métriques  : {MODELS_DIR / 'xgb_da_metrics.json'}")

    return best_model, all_metrics


if __name__ == "__main__":
    run_model_da()