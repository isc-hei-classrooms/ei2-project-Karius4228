"""
model_intraday_v3.py — Modèles Intraday multi-horizon (v3)
──────────────────────────────────────────────────────────
Ce script entraîne un modèle XGBoost et un LightGBM par horizon
de prédiction (12 horizons : t+15min à t+3h).

Contrairement au DA qui fait UNE prédiction à 24h,
l'Intraday entraîne 12 modèles indépendants, un par horizon.

Auteur : Marius Fabbri
"""

import json, logging, time, sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb

# ── Configuration des chemins ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Imports locaux :
# - get_feature_columns : liste des features de base (hors NWP)
# - get_nwp_columns : liste des colonnes NWP brutes
# - build_nwp_for_horizon : décale les NWP selon l'horizon h
#   (pour h=4 soit t+60min, on prend la NWP pointée 60min dans le futur)
# - HORIZON_MAX : 12 (nombre d'horizons, de t+15min à t+180min)
from features_intraday_v3 import get_feature_columns, get_nwp_columns, build_nwp_for_horizon, HORIZON_MAX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = SCRIPT_DIR / "data" / "processed" / "features_v3"
MODELS_DIR   = SCRIPT_DIR / "models_saved"
RANDOM_STATE = 42

# ── Hyperparamètres fixes (pas de recherche pour l'Intraday) ──
# Choisis manuellement, piste d'amélioration : faire une RandomizedSearchCV
# sur un horizon médian (ex: h=6) puis réutiliser les params pour tous les horizons
XGB_PARAMS = dict(
    n_estimators=500,          # Nombre d'arbres
    max_depth=6,               # Profondeur max par arbre
    learning_rate=0.05,        # Taux d'apprentissage
    subsample=0.85,            # 85% des données par arbre
    colsample_bytree=0.85,     # 85% des features par arbre
    min_child_weight=5,        # Poids min par feuille
    reg_lambda=2.0,            # Régularisation L2
    reg_alpha=0.1,             # Régularisation L1
    objective="reg:squarederror",  # Loss = MSE
    tree_method="hist",        # Algorithme rapide
    random_state=RANDOM_STATE,
    n_jobs=-1,                 # Tous les cœurs CPU
)

LGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=10,      # Équivalent LightGBM de min_child_weight
    reg_lambda=2.0,
    reg_alpha=0.1,
    num_leaves=63,             # Nombre max de feuilles (spécifique LightGBM)
    objective="regression",    # Loss = MSE (nom LightGBM)
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,                # Pas de logs verbeux
)


def compute_metrics(y_true, y_pred):
    """
    Calcule MAE, RMSE et MAPE.
    Même fonction que dans model_da_v3.py.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mask = np.abs(yt) > 0.1
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.sum() > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_model_intraday():
    log.info("=" * 60)
    log.info("MODÈLES INTRADAY (v3)")
    log.info("=" * 60)

    # ═══════════════════════════════════════════════════════════
    # 1. CHARGEMENT DES DONNÉES
    # ═══════════════════════════════════════════════════════════
    # Les parquets contiennent les features de base + les 12 colonnes cibles
    # (target_1 à target_12) + les colonnes NWP brutes (avant décalage par horizon)
    df_train = pl.read_parquet(FEATURES_DIR / "train_intraday_v3.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_intraday_v3.parquet")
    log.info(f"Train: {df_train.shape} | Test: {df_test.shape}")

    # Features de base : lags charge, lags PV, rolling stats, encodage cyclique, etc.
    # (tout sauf les NWP qui seront ajoutées par horizon)
    base_features = get_feature_columns(df_train)

    # Colonnes NWP brutes : température, radiation, précipitations, etc.
    # Elles seront décalées différemment selon l'horizon
    nwp_cols = get_nwp_columns()
    log.info(f"Base features: {len(base_features)} | NWP cols: {len(nwp_cols)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ═══════════════════════════════════════════════════════════
    # 2. BOUCLE SUR LES 12 HORIZONS
    # ═══════════════════════════════════════════════════════════
    # Pour chaque horizon h (1 à 12), on entraîne un modèle indépendant
    # h=1 → t+15min, h=4 → t+60min, h=12 → t+180min
    for h in range(1, HORIZON_MAX + 1):
        log.info(f"\n── Horizon h={h} (t+{h*15}min) ──")

        # ── Préparation des NWP pour cet horizon ──
        # build_nwp_for_horizon décale les colonnes NWP de h pas dans le futur
        # Ex: pour h=4 (t+60min), on prend la NWP pointée vers t+60min
        # Les nouvelles colonnes sont nommées pred_temperature_ctrl_h4, etc.
        tr_h = build_nwp_for_horizon(df_train, h)
        te_h = build_nwp_for_horizon(df_test, h)

        # On construit la liste complète des features pour cet horizon :
        # features de base (sans les NWP brutes) + NWP décalées pour l'horizon h
        nwp_h_cols = [f"{c}_h{h}" for c in nwp_cols if f"{c}_h{h}" in tr_h.columns]
        feature_cols = [c for c in base_features if c not in nwp_cols] + nwp_h_cols

        # La cible pour cet horizon : target_h = charge nette à t+h*15min
        target_col = f"target_{h}"

        # ── Suppression des lignes avec des NaN dans la cible ou les NWP ──
        # On ne peut pas entraîner sur des lignes sans cible
        mask_cols = [target_col] + nwp_h_cols
        for c in mask_cols:
            tr_h = tr_h.filter(pl.col(c).is_not_null())
            te_h = te_h.filter(pl.col(c).is_not_null())

        # ── Imputation des NaN résiduels dans les features ──
        # Pour les rares features encore nulles, on remplace par la médiane du train
        # (la médiane du train est utilisée aussi pour le test, pour éviter le leakage)
        for c in feature_cols:
            if tr_h[c].null_count() > 0:
                med = float(tr_h[c].drop_nulls().median())
                tr_h = tr_h.with_columns(pl.col(c).fill_null(med))
                te_h = te_h.with_columns(pl.col(c).fill_null(med))

        # Conversion en matrices numpy
        X_train = tr_h.select(feature_cols).to_numpy().astype(np.float32)
        y_train = tr_h[target_col].to_numpy().astype(np.float32)
        X_test  = te_h.select(feature_cols).to_numpy().astype(np.float32)
        y_test  = te_h[target_col].to_numpy().astype(np.float32)

        log.info(f"  Train: {X_train.shape} | Test: {X_test.shape}")

        # ── Baseline naïve ──
        # Naïf = charge d'hier à la même heure (load_lag_96 = décalage de 96 pas = 24h)
        y_naive = te_h["load_lag_96"].to_numpy()
        m_naive = compute_metrics(y_test, y_naive)

        # ═══════════════════════════════════════════════════════
        # 3. ENTRAÎNEMENT DES MODÈLES POUR CET HORIZON
        # ═══════════════════════════════════════════════════════
        t0 = time.time()

        # XGBoost : entraînement direct avec les hyperparamètres fixes
        # (pas de RandomizedSearchCV ici, contrairement au DA)
        model_xgb = xgb.XGBRegressor(**XGB_PARAMS)
        model_xgb.fit(X_train, y_train, verbose=0)
        y_xgb = model_xgb.predict(X_test)
        m_xgb = compute_metrics(y_test, y_xgb)

        # Sauvegarde du modèle XGBoost pour cet horizon
        joblib.dump(model_xgb, MODELS_DIR / f"xgb_id_v3_h{h}.joblib")

        # LightGBM : même chose
        model_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
        model_lgb.fit(X_train, y_train)
        y_lgb = model_lgb.predict(X_test)
        m_lgb = compute_metrics(y_test, y_lgb)

        # Sauvegarde du modèle LightGBM pour cet horizon
        joblib.dump(model_lgb, MODELS_DIR / f"lgb_id_v3_h{h}.joblib")

        log.info(f"  Naïf={m_naive['MAE']:.4f} | XGB={m_xgb['MAE']:.4f} | LGB={m_lgb['MAE']:.4f} ({time.time()-t0:.0f}s)")

        # Stockage des résultats pour le tableau final
        all_results[f"h{h}"] = {
            "naive": m_naive, "xgboost": m_xgb, "lightgbm": m_lgb,
            "n_train": X_train.shape[0], "n_test": X_test.shape[0],
            "n_features": len(feature_cols),
        }

    # ═══════════════════════════════════════════════════════════
    # 4. TABLEAU RÉCAPITULATIF
    # ═══════════════════════════════════════════════════════════
    # Affiche les MAE de chaque horizon + le gain relatif vs le naïf
    log.info("\n" + "=" * 80)
    log.info(f"  {'Horizon':<10} {'Naïf':>10} {'XGB':>10} {'LGB':>10} {'XGB vs Naïf':>12} {'LGB vs Naïf':>12}")
    log.info("  " + "-" * 70)
    for h in range(1, HORIZON_MAX + 1):
        r = all_results[f"h{h}"]
        n, x, l = r["naive"]["MAE"], r["xgboost"]["MAE"], r["lightgbm"]["MAE"]
        # Gain = (MAE_naïf - MAE_modèle) / MAE_naïf * 100
        log.info(f"  t+{h*15:3d}min   {n:>10.4f} {x:>10.4f} {l:>10.4f} {100*(n-x)/n:>+11.1f}% {100*(n-l)/n:>+11.1f}%")

    # Moyennes sur tous les horizons
    avg_n = np.mean([all_results[f"h{h}"]["naive"]["MAE"] for h in range(1, HORIZON_MAX+1)])
    avg_x = np.mean([all_results[f"h{h}"]["xgboost"]["MAE"] for h in range(1, HORIZON_MAX+1)])
    avg_l = np.mean([all_results[f"h{h}"]["lightgbm"]["MAE"] for h in range(1, HORIZON_MAX+1)])
    log.info("  " + "-" * 70)
    log.info(f"  {'Moyenne':<10} {avg_n:>10.4f} {avg_x:>10.4f} {avg_l:>10.4f} {100*(avg_n-avg_x)/avg_n:>+11.1f}% {100*(avg_n-avg_l)/avg_n:>+11.1f}%")
    log.info("=" * 80)

    # ═══════════════════════════════════════════════════════════
    # 5. SAUVEGARDE DES MÉTRIQUES
    # ═══════════════════════════════════════════════════════════
    # Un seul fichier JSON avec les résultats de tous les horizons
    # (utilisé par visualize_intraday.py pour les graphiques)
    with open(MODELS_DIR / "intraday_v3_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\n✓ Sauvegardé dans {MODELS_DIR}")
    return all_results


# Point d'entrée : exécuter avec `python model_intraday_v3.py`
if __name__ == "__main__":
    run_model_intraday()