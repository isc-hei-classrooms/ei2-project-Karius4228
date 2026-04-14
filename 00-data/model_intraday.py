"""
model_intraday.py — Modèle XGBoost Intraday (v2)
Auteur : Marius Fabbri

Pipeline :
  À chaque instant t, prédit la charge pour les 12 prochains pas (3h).
  Un modèle XGBoost par horizon (12 modèles indépendants).

  Pour chaque horizon h :
    1. build_nwp_for_horizon(df, h) → NWP alignées sur t+h
    2. Remplace les 5 NWP brutes par les 5 NWP shiftées dans X
    3. RandomizedSearchCV + TimeSeriesSplit(5) sur train
    4. Évalue sur test → MAE/RMSE/MAPE
    5. Compare à baseline naïve J-1 (load_lag_96)

  Sauvegarde : un modèle .joblib par horizon + métriques globales .json
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
HORIZON_MAX  = 12
NWP_COLS     = get_nwp_columns()


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
        "n_mape_excluded": int((~mask).sum()),
    }


# ─── Construction de X pour un horizon donné ─────────────────────────────────

def build_X_for_horizon(df: pl.DataFrame, base_features: list[str], h: int) -> np.ndarray:
    """
    Pour l'horizon h :
      1. Ajoute les colonnes NWP shiftées (pred_*_h{h})
      2. Remplace les NWP brutes par les NWP alignées dans la liste de features
      3. Retourne numpy array X
    """
    df_h = build_nwp_for_horizon(df, h)

    # Substitution : NWP brutes → NWP shiftées
    feature_cols_h = [
        f"{c}_h{h}" if c in NWP_COLS else c
        for c in base_features
    ]

    return df_h.select(feature_cols_h).to_numpy()


# ─── Tableau comparatif ───────────────────────────────────────────────────────

def print_comparison(results_xgb: dict, results_naive: dict) -> None:
    log.info("\n" + "═" * 72)
    log.info("  TABLEAU COMPARATIF — TEST SET INTRADAY")
    log.info("═" * 72)
    log.info(
        f"  {'Horizon':<10} "
        f"{'Naïf MAE':>10} {'Naïf RMSE':>10} "
        f"{'XGB MAE':>10} {'XGB RMSE':>10} "
        f"{'Δ MAE':>8}"
    )
    log.info("  " + "─" * 66)

    maes_naive, maes_xgb = [], []
    for h in range(1, HORIZON_MAX + 1):
        if h not in results_xgb or h not in results_naive:
            continue
        mn = results_naive[h]
        mx = results_xgb[h]
        delta = (1 - mx["MAE"] / mn["MAE"]) * 100
        log.info(
            f"  t+{h:<8} "
            f"{mn['MAE']:>10.4f} {mn['RMSE']:>10.4f} "
            f"{mx['MAE']:>10.4f} {mx['RMSE']:>10.4f} "
            f"{delta:>+7.1f}%"
        )
        maes_naive.append(mn["MAE"])
        maes_xgb.append(mx["MAE"])

    log.info("  " + "─" * 66)
    avg_naive = float(np.mean(maes_naive))
    avg_xgb   = float(np.mean(maes_xgb))
    delta_avg = (1 - avg_xgb / avg_naive) * 100
    log.info(
        f"  {'Moyenne':<10} "
        f"{avg_naive:>10.4f} {'':>10} "
        f"{avg_xgb:>10.4f} {'':>10} "
        f"{delta_avg:>+7.1f}%"
    )
    log.info("═" * 72)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_model_intraday() -> dict:
    log.info("\n" + "═" * 60)
    log.info("MODÈLE XGBOOST — INTRADAY (v2)")
    log.info("═" * 60)

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    df_train = pl.read_parquet(FEATURES_DIR / "train_intraday_v2.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_intraday_v2.parquet")
    log.info(f"Train : {df_train.shape} | Test : {df_test.shape}")

    # ── 2. Features de base (sans NWP shiftées) ───────────────────────────────
    base_features = get_feature_columns(df_train)
    log.info(f"{len(base_features)} features de base (NWP remplacées par horizon)")

    # ── 3. Baseline naïve J-1 (load_lag_96) ──────────────────────────────────
    log.info("\n── Baseline naïve J-1 (load_lag_96) ────────────────────────")
    y_lag        = df_test["load_lag_96"].to_numpy()
    results_naive: dict = {}

    for h in range(1, HORIZON_MAX + 1):
        y_true = df_test[f"target_{h}"].to_numpy()
        results_naive[h] = compute_metrics(y_true, y_lag)

    avg_mae_naive = np.mean([r["MAE"] for r in results_naive.values()])
    log.info(f"  MAE moyenne (t+1..t+12) : {avg_mae_naive:.4f}")

    # ── 4. Boucle sur les 12 horizons ─────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_xgb:    dict = {}
    best_params_all: dict = {}
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    for h in range(1, HORIZON_MAX + 1):
        log.info(f"\n── Horizon t+{h} ({h*15}min) ── RandomizedSearchCV ({N_ITER} iter) ──")

        # X avec NWP alignées sur h
        X_train = build_X_for_horizon(df_train, base_features, h)
        y_train = df_train[f"target_{h}"].to_numpy()
        X_test  = build_X_for_horizon(df_test,  base_features, h)
        y_test  = df_test[f"target_{h}"].to_numpy()

        # Filtre NaN sur y_train (shift peut créer des NaN en fin de série)
        valid_train = ~np.isnan(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        base_model = xgb.XGBRegressor(
            objective    = "reg:squarederror",
            tree_method  = "hist",
            random_state = RANDOM_STATE,
            n_jobs       = 1,
        )

        search = RandomizedSearchCV(
            estimator           = base_model,
            param_distributions = PARAM_GRID,
            n_iter              = N_ITER,
            cv                  = tscv,
            scoring             = "neg_mean_absolute_error",
            n_jobs              = -1,
            random_state        = RANDOM_STATE,
            verbose             = 0,    # silencieux — 12 horizons
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred     = best_model.predict(X_test)
        m          = compute_metrics(y_test, y_pred)
        results_xgb[h]    = m
        best_params_all[h] = search.best_params_

        log.info(
            f"  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%"
            f"  (CV MAE={-search.best_score_:.4f})"
        )

        # Sauvegarde modèle par horizon
        joblib.dump(best_model, MODELS_DIR / f"xgb_intraday_h{h}.joblib")

    # ── 5. Tableau comparatif ─────────────────────────────────────────────────
    print_comparison(results_xgb, results_naive)

    # ── 6. Sauvegarde métriques + params ──────────────────────────────────────
    # Conversion clés int → str pour JSON
    metrics_out = {
        "xgb"  : {str(k): v for k, v in results_xgb.items()},
        "naive": {str(k): v for k, v in results_naive.items()},
    }
    params_out = {str(k): v for k, v in best_params_all.items()}

    with open(MODELS_DIR / "xgb_intraday_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    with open(MODELS_DIR / "xgb_intraday_best_params.json", "w", encoding="utf-8") as f:
        json.dump(params_out, f, indent=2)

    log.info(f"\n✓ Métriques  : {MODELS_DIR / 'xgb_intraday_metrics.json'}")
    log.info(f"✓ Paramètres : {MODELS_DIR / 'xgb_intraday_best_params.json'}")
    log.info(f"✓ Modèles    : {MODELS_DIR}/xgb_intraday_h1..h12.joblib")

    return results_xgb


if __name__ == "__main__":
    run_model_intraday()