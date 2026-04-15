"""
model_da_v3.py — Modèles XGBoost + LightGBM Day-Ahead (v3)
────────────────────────────────────────────────────────────
Ce script entraîne deux modèles de prédiction de charge nette
à 24h (Day-Ahead) et les compare aux baselines (naïf + Oiken).

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

# ── Configuration des chemins ──
# On ajoute le dossier du script au PATH pour pouvoir importer nos modules
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Imports locaux :
# - get_feature_columns : retourne la liste des colonnes à utiliser comme features
# - detect_frozen_forecast / filter_frozen_days : détecte et exclut les jours
#   où la prévision Oiken est figée (données corrompues)
from features_da_v3 import get_feature_columns
from clean_forecast import detect_frozen_forecast, filter_frozen_days

# Configuration du logging pour suivre l'avancement dans le terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Chemins vers les données features et le dossier de sauvegarde des modèles
FEATURES_DIR = SCRIPT_DIR / "data" / "processed" / "features_v3"
MODELS_DIR   = SCRIPT_DIR / "models_saved"

# ── Paramètres de la recherche d'hyperparamètres ──
N_ITER       = 40          # Nombre de combinaisons testées (RandomizedSearchCV)
CV_FOLDS     = 5           # Nombre de folds pour la cross-validation temporelle
RANDOM_STATE = 42          # Graine aléatoire pour la reproductibilité

# Grille de recherche XGBoost : chaque paramètre a plusieurs valeurs possibles,
# le RandomizedSearchCV va en tester 40 combinaisons aléatoires
XGB_PARAM_GRID = {
    "n_estimators":     [300, 500, 800],       # Nombre d'arbres
    "max_depth":        [4, 6, 8],             # Profondeur max de chaque arbre
    "learning_rate":    [0.01, 0.05, 0.1],     # Taux d'apprentissage (petit = plus lent mais plus précis)
    "subsample":        [0.7, 0.85, 1.0],      # Fraction des données utilisées par arbre
    "colsample_bytree": [0.7, 0.85, 1.0],      # Fraction des features utilisées par arbre
    "min_child_weight": [1, 5, 10],            # Poids min par feuille (régularisation)
    "reg_alpha":        [0, 0.1, 1.0],         # Régularisation L1 (force certains poids à zéro)
    "reg_lambda":       [1.0, 2.0, 5.0],       # Régularisation L2 (réduit les poids extrêmes)
}

# Grille de recherche LightGBM : même logique, avec des paramètres spécifiques à LightGBM
LGB_PARAM_GRID = {
    "n_estimators":     [300, 500, 800],
    "max_depth":        [4, 6, 8, -1],         # -1 = pas de limite de profondeur
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_samples":[5, 10, 20],           # Équivalent LightGBM de min_child_weight
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [1.0, 2.0, 5.0],
    "num_leaves":       [31, 63, 127],         # Nombre max de feuilles par arbre (spécifique LightGBM)
}


def compute_metrics(y_true, y_pred):
    """
    Calcule MAE, RMSE et MAPE entre les valeurs réelles et prédites.
    - MAE  : erreur absolue moyenne (en z-score, sans unité)
    - RMSE : racine de l'erreur quadratique moyenne (pénalise plus les grosses erreurs)
    - MAPE : erreur en pourcentage (on filtre les valeurs proches de zéro pour éviter la division par ~0)
    """
    # On ignore les lignes avec des NaN
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]

    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    # MAPE : on ne calcule que sur les valeurs où |y| > 0.1
    # car diviser par une valeur proche de zéro donne des pourcentages absurdes
    mask = np.abs(yt) > 0.1
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.sum() > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n": int(valid.sum())}


def run_model_da():
    log.info("=" * 60)
    log.info("MODÈLES DAY-AHEAD (v3)")
    log.info("=" * 60)

    # ═══════════════════════════════════════════════════════════
    # 1. CHARGEMENT DES DONNÉES
    # ═══════════════════════════════════════════════════════════
    # Les fichiers parquet contiennent déjà toutes les features calculées
    # par features_da_v3.py (lags, rolling stats, NWP, encodage cyclique, etc.)
    df_train = pl.read_parquet(FEATURES_DIR / "train_da_v3.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_da_v3.parquet")
    log.info(f"Train: {df_train.shape} | Test: {df_test.shape}")

    # ═══════════════════════════════════════════════════════════
    # 2. NETTOYAGE DU TEST SET
    # ═══════════════════════════════════════════════════════════
    # On détecte les jours où la prévision Oiken est figée à une valeur constante
    # (ex: -1.996 pendant 96h en sept 2025 = données corrompues)
    # Ces jours sont exclus pour ne pas fausser la comparaison avec Oiken
    suspect_dates = detect_frozen_forecast(df_test, "forecast_load")
    df_test_clean = filter_frozen_days(df_test, suspect_dates)

    # ═══════════════════════════════════════════════════════════
    # 3. PRÉPARATION DES MATRICES X et y
    # ═══════════════════════════════════════════════════════════
    # get_feature_columns() retourne automatiquement les bonnes colonnes
    # en excluant timestamp, target, forecast_load, et tout ce qui ne doit
    # pas être utilisé comme feature
    feature_cols = get_feature_columns(df_train)
    log.info(f"{len(feature_cols)} features")

    # Conversion en tableaux numpy pour scikit-learn / XGBoost / LightGBM
    # float32 suffit et économise de la mémoire
    X_train = df_train.select(feature_cols).to_numpy().astype(np.float32)
    y_train = df_train["target"].to_numpy().astype(np.float32)  # La charge nette à prédire (J+1)
    X_test  = df_test_clean.select(feature_cols).to_numpy().astype(np.float32)
    y_test  = df_test_clean["target"].to_numpy().astype(np.float32)

    all_metrics = {}

    # ═══════════════════════════════════════════════════════════
    # 4. CALCUL DES BASELINES
    # ═══════════════════════════════════════════════════════════
    # Baseline naïve : on prédit que demain = hier à la même heure
    y_naive = df_test_clean["load_lag_1d"].to_numpy()
    all_metrics["Naïf J-1"] = compute_metrics(y_test, y_naive)

    # Baseline Oiken : la prévision du distributeur, alignée sur le bon timestamp
    # forecast_load_target = prévision Oiken pour le même instant que la cible
    y_oiken = df_test_clean["forecast_load_target"].to_numpy()
    all_metrics["Oiken"] = compute_metrics(y_test, y_oiken)

    log.info(f"Naïf:  MAE={all_metrics['Naïf J-1']['MAE']:.4f}")
    log.info(f"Oiken: MAE={all_metrics['Oiken']['MAE']:.4f}")

    # ═══════════════════════════════════════════════════════════
    # 5. ENTRAÎNEMENT XGBOOST avec recherche d'hyperparamètres
    # ═══════════════════════════════════════════════════════════
    log.info(f"\n── XGBoost ({N_ITER} iter, {CV_FOLDS} folds) ──")

    # TimeSeriesSplit : cross-validation respectant l'ordre chronologique
    # Contrairement à un KFold classique, on ne mélange jamais les données
    # Fold 1 : train=[0..20%] test=[20..40%]
    # Fold 2 : train=[0..40%] test=[40..60%]  etc.
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    t0 = time.time()

    # RandomizedSearchCV teste N_ITER combinaisons aléatoires de la grille
    # et retient celle qui minimise la MAE en cross-validation
    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(
            objective="reg:squarederror",  # Loss = MSE (erreur quadratique)
            tree_method="hist",            # Algorithme rapide par histogrammes
            random_state=RANDOM_STATE,
            n_jobs=1,                      # 1 thread par modèle (parallélisme géré par CV)
        ),
        XGB_PARAM_GRID,
        n_iter=N_ITER,                     # 40 combinaisons testées
        cv=tscv,                           # Cross-validation temporelle
        scoring="neg_mean_absolute_error", # On optimise la MAE (neg car sklearn maximise)
        n_jobs=-1,                         # Utiliser tous les cœurs CPU pour la CV
        random_state=RANDOM_STATE,
        verbose=1,
    )
    xgb_search.fit(X_train, y_train)

    # Prédiction sur le test set avec le meilleur modèle trouvé
    y_xgb = xgb_search.best_estimator_.predict(X_test)
    all_metrics["XGBoost"] = compute_metrics(y_test, y_xgb)
    log.info(f"  CV MAE: {-xgb_search.best_score_:.4f} | Test MAE: {all_metrics['XGBoost']['MAE']:.4f} ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════════════════
    # 6. ENTRAÎNEMENT LIGHTGBM avec recherche d'hyperparamètres
    # ═══════════════════════════════════════════════════════════
    log.info(f"\n── LightGBM ({N_ITER} iter, {CV_FOLDS} folds) ──")
    t0 = time.time()

    lgb_search = RandomizedSearchCV(
        lgb.LGBMRegressor(
            objective="regression",        # Loss = MSE (nom LightGBM pour squarederror)
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,                    # Supprime les logs verbeux de LightGBM
        ),
        LGB_PARAM_GRID,
        n_iter=N_ITER, cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    lgb_search.fit(X_train, y_train)

    y_lgb = lgb_search.best_estimator_.predict(X_test)
    all_metrics["LightGBM"] = compute_metrics(y_test, y_lgb)
    log.info(f"  CV MAE: {-lgb_search.best_score_:.4f} | Test MAE: {all_metrics['LightGBM']['MAE']:.4f} ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════════════════
    # 7. AFFICHAGE DU TABLEAU RÉCAPITULATIF
    # ═══════════════════════════════════════════════════════════
    # Compare tous les modèles avec Oiken comme référence
    oiken_mae = all_metrics["Oiken"]["MAE"]
    log.info("\n" + "=" * 70)
    log.info(f"  {'Modèle':<20} {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}  {'vs Oiken':>10}")
    log.info("  " + "-" * 60)
    for label, m in all_metrics.items():
        mape_s = f"{m['MAPE']:.2f}%" if not np.isnan(m["MAPE"]) else "N/A"
        # Calcul du gain relatif par rapport à Oiken : positif = meilleur que Oiken
        vs = "ref" if label == "Oiken" else f"{100*(oiken_mae - m['MAE'])/oiken_mae:+.1f}%"
        log.info(f"  {label:<20} {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {mape_s:>8}  {vs:>10}")
    log.info("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # 8. FEATURE IMPORTANCE
    # ═══════════════════════════════════════════════════════════
    # On extrait l'importance de chaque feature selon le "gain" :
    # combien chaque feature réduit l'erreur quand elle est utilisée pour couper un arbre
    booster = xgb_search.best_estimator_.get_booster()
    scores = booster.get_score(importance_type="gain")

    # XGBoost nomme les features "f0", "f1", etc. → on les renomme avec les vrais noms
    fname_map = {f"f{i}": name for i, name in enumerate(feature_cols)}
    named_scores = {fname_map.get(k, k): v for k, v in scores.items()}

    log.info("\n── Top 15 features (gain) ──")
    for rank, (name, score) in enumerate(sorted(named_scores.items(), key=lambda x: -x[1])[:15], 1):
        log.info(f"  {rank:2d}. {name:<40} {score:>10.1f}")

    # ═══════════════════════════════════════════════════════════
    # 9. SAUVEGARDE
    # ═══════════════════════════════════════════════════════════
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarde des modèles entraînés (pour réutilisation sans réentraîner)
    joblib.dump(xgb_search.best_estimator_, MODELS_DIR / "xgb_da_v3.joblib")
    joblib.dump(lgb_search.best_estimator_, MODELS_DIR / "lgb_da_v3.joblib")

    # Sauvegarde des métriques et des meilleurs hyperparamètres en JSON
    with open(MODELS_DIR / "da_v3_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(MODELS_DIR / "da_v3_xgb_params.json", "w") as f:
        json.dump(xgb_search.best_params_, f, indent=2)
    with open(MODELS_DIR / "da_v3_lgb_params.json", "w") as f:
        json.dump(lgb_search.best_params_, f, indent=2)

    # Sauvegarde complète des prédictions pour la visualisation ultérieure
    # (utilisé par visualize_da.py pour générer les graphiques)
    joblib.dump({
        "y_test": y_test, "y_naive": y_naive, "y_oiken": y_oiken,
        "y_xgb": y_xgb, "y_lgb": y_lgb,
        "timestamps": df_test_clean["timestamp"].to_list(),
        "results": all_metrics, "feature_cols": feature_cols,
        "xgb_importance": named_scores,
    }, MODELS_DIR / "da_v3_predictions.joblib")

    log.info(f"\n✓ Sauvegardé dans {MODELS_DIR}")
    return all_metrics


# Point d'entrée : exécuter avec `python model_da_v3.py`
if __name__ == "__main__":
    run_model_da()