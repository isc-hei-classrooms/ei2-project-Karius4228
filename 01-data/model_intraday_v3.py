"""
model_intraday_v3.py — Modèles Intraday multi-horizon (v3)
──────────────────────────────────────────────────────────
12 horizons (t+15min à t+3h), XGBoost + LightGBM par horizon.
NWP shiftées par horizon via build_nwp_for_horizon().

Auteur : Marius Fabbri
"""

import json, logging, time, sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from features_intraday_v3 import (
    get_feature_columns, get_nwp_columns, build_nwp_for_horizon, HORIZON_MAX,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path(__file__).resolve().parent / "data" / "processed" / "features_v3"
MODELS_DIR   = Path(__file__).resolve().parent / "models_saved"
RANDOM_STATE = 42

# Hyperparamètres fixes (raisonnables, pas de search pour accélérer)
XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
    reg_lambda=2.0, reg_alpha=0.1,
    objective="reg:squarederror", tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1,
)

LGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_samples=10,
    reg_lambda=2.0, reg_alpha=0.1, num_leaves=63,
    objective="regression", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)


def compute_metrics(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mask = np.abs(yt) > 0.1
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.sum() > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_model_intraday():
    log.info("=" * 60)
    log.info("MODÈLES INTRADAY (v3 — multi-horizon)")
    log.info("=" * 60)

    df_train = pl.read_parquet(FEATURES_DIR / "train_intraday_v3.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_intraday_v3.parquet")
    log.info(f"Train: {df_train.shape} | Test: {df_test.shape}")

    base_features = get_feature_columns(df_train)
    nwp_cols = get_nwp_columns()
    log.info(f"Base features: {len(base_features)} | NWP cols: {len(nwp_cols)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for h in range(1, HORIZON_MAX + 1):
        log.info(f"\n── Horizon h={h} (t+{h*15}min) ──")

        # Préparer NWP shiftées pour cet horizon
        tr_h = build_nwp_for_horizon(df_train, h)
        te_h = build_nwp_for_horizon(df_test, h)

        # Features = base + NWP_h{horizon} (remplace les NWP brutes)
        nwp_h_cols = [f"{c}_h{h}" for c in nwp_cols if f"{c}_h{h}" in tr_h.columns]
        # Exclure les NWP brutes, garder les _h{h}
        feature_cols = [c for c in base_features if c not in nwp_cols] + nwp_h_cols

        target_col = f"target_{h}"

        # Filtrer les nulls sur NWP shiftées et target
        mask_cols = [target_col] + nwp_h_cols
        for c in mask_cols:
            tr_h = tr_h.filter(pl.col(c).is_not_null())
            te_h = te_h.filter(pl.col(c).is_not_null())

        # Imputation médiane pour les résidus
        for c in feature_cols:
            if tr_h[c].null_count() > 0:
                med = float(tr_h[c].drop_nulls().median())
                tr_h = tr_h.with_columns(pl.col(c).fill_null(med))
                te_h = te_h.with_columns(pl.col(c).fill_null(med))

        X_train = tr_h.select(feature_cols).to_numpy().astype(np.float32)
        y_train = tr_h[target_col].to_numpy().astype(np.float32)
        X_test  = te_h.select(feature_cols).to_numpy().astype(np.float32)
        y_test  = te_h[target_col].to_numpy().astype(np.float32)

        log.info(f"  Train: {X_train.shape} | Test: {X_test.shape}")

        # Baseline naïve (charge J-1)
        y_naive = te_h["load_lag_96"].to_numpy()
        m_naive = compute_metrics(y_test, y_naive)

        # XGBoost
        t0 = time.time()
        model_xgb = xgb.XGBRegressor(**XGB_PARAMS)
        model_xgb.fit(X_train, y_train, verbose=0)
        y_xgb = model_xgb.predict(X_test)
        m_xgb = compute_metrics(y_test, y_xgb)
        joblib.dump(model_xgb, MODELS_DIR / f"xgb_id_v3_h{h}.joblib")

        # LightGBM
        model_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
        model_lgb.fit(X_train, y_train)
        y_lgb = model_lgb.predict(X_test)
        m_lgb = compute_metrics(y_test, y_lgb)
        joblib.dump(model_lgb, MODELS_DIR / f"lgb_id_v3_h{h}.joblib")

        dt = time.time() - t0
        log.info(f"  Naïf MAE={m_naive['MAE']:.4f} | XGB MAE={m_xgb['MAE']:.4f} | LGB MAE={m_lgb['MAE']:.4f} ({dt:.0f}s)")

        all_results[f"h{h}"] = {
            "naive": m_naive, "xgboost": m_xgb, "lightgbm": m_lgb,
            "n_train": X_train.shape[0], "n_test": X_test.shape[0],
            "n_features": len(feature_cols),
        }

    # ── Tableau récapitulatif ──
    log.info("\n" + "=" * 80)
    log.info(f"  {'Horizon':<10} {'Naïf MAE':>10} {'XGB MAE':>10} {'LGB MAE':>10} {'XGB vs Naïf':>12} {'LGB vs Naïf':>12}")
    log.info("  " + "-" * 70)
    for h in range(1, HORIZON_MAX + 1):
        r = all_results[f"h{h}"]
        n_mae = r["naive"]["MAE"]
        x_mae = r["xgboost"]["MAE"]
        l_mae = r["lightgbm"]["MAE"]
        x_vs = 100 * (n_mae - x_mae) / n_mae
        l_vs = 100 * (n_mae - l_mae) / n_mae
        log.info(f"  t+{h*15:3d}min   {n_mae:>10.4f} {x_mae:>10.4f} {l_mae:>10.4f} {x_vs:>+11.1f}% {l_vs:>+11.1f}%")

    # Moyenne
    avg_naive = np.mean([all_results[f"h{h}"]["naive"]["MAE"] for h in range(1, HORIZON_MAX+1)])
    avg_xgb = np.mean([all_results[f"h{h}"]["xgboost"]["MAE"] for h in range(1, HORIZON_MAX+1)])
    avg_lgb = np.mean([all_results[f"h{h}"]["lightgbm"]["MAE"] for h in range(1, HORIZON_MAX+1)])
    log.info("  " + "-" * 70)
    log.info(f"  {'Moyenne':<10} {avg_naive:>10.4f} {avg_xgb:>10.4f} {avg_lgb:>10.4f} {100*(avg_naive-avg_xgb)/avg_naive:>+11.1f}% {100*(avg_naive-avg_lgb)/avg_naive:>+11.1f}%")
    log.info("=" * 80)

    # Sauvegarde métriques
    with open(MODELS_DIR / "intraday_v3_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\n✓ Modèles et métriques sauvegardés dans {MODELS_DIR}")

    return all_results


if __name__ == "__main__":
    run_model_intraday()
