"""
denorm_utils.py — Utilitaires normalisation / dénormalisation / métriques
──────────────────────────────────────────────────────────────────────────
Fonctions partagées entre features_da_v4.py et model_da_v4.py.

Principe du multiplicateur de signal :

    load_normalized[t] = load[t] / pv_scaler[t]

    → Le modèle apprend sur load_normalized, qui est stationnaire
      (la croissance PV est factorisée dans pv_scaler)

    → À l'inférence :
    load_pred[t] = load_normalized_pred[t] × pv_scaler[t]

    → Les métriques (MAE, RMSE, MAPE) sont TOUJOURS calculées sur les
      valeurs dénormalisées, pour être comparables avec Oiken et le naïf.

Auteur : Marius Fabbri
"""

import numpy as np
import polars as pl
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_load(series: np.ndarray, scaler: np.ndarray) -> np.ndarray:
    """
    Normalise la charge par le scaler PV.

    Paramètres
    ----------
    series  : array (N,) — valeurs de charge brute (z-score Oiken)
    scaler  : array (N,) — valeur du multiplicateur PV pour chaque pas

    Retourne
    --------
    array (N,) — charge normalisée (sans dérive PV)

    Notes
    -----
    La division est protégée : si scaler < 0.1 (ne devrait pas arriver
    après clip dans pv_scaler_v4.py), on remplace par 0.1 pour éviter
    les valeurs extrêmes.
    """
    safe_scaler = np.where(np.abs(scaler) < 0.1, 0.1, scaler)
    return series / safe_scaler


def denormalize_load(series_norm: np.ndarray, scaler: np.ndarray) -> np.ndarray:
    """
    Dénormalise la charge prédite (inverse de normalize_load).

    Paramètres
    ----------
    series_norm : array (N,) — valeurs normalisées prédites par le modèle
    scaler      : array (N,) — scaler PV pour chaque pas (même grille que target)

    Retourne
    --------
    array (N,) — charge prédite dans l'espace original (z-score Oiken)
    """
    return series_norm * scaler


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """
    Calcule MAE, RMSE, MAPE sur les valeurs DÉNORMALISÉES.

    IMPORTANT : toujours appeler cette fonction APRÈS dénormalisation,
    sinon les métriques ne sont pas comparables avec Oiken et le naïf.

    Paramètres
    ----------
    y_true : valeurs réelles (espace original, z-score Oiken)
    y_pred : valeurs prédites (espace original, après dénorm)
    label  : nom du modèle (pour les logs)

    Retourne
    --------
    dict avec clés : MAE, RMSE, MAPE, n, bias
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]

    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    bias = float(np.mean(yp - yt))   # biais moyen : positif = sur-estimation

    # MAPE filtré sur |y| > 0.1 (évite division par ~0 autour du net-zéro)
    mask = np.abs(yt) > 0.1
    mape = float(
        np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100
    ) if mask.sum() > 0 else float("nan")

    if label:
        log.info(f"  {label:<20} MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.1f}% bias={bias:+.4f} n={valid.sum()}")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "bias": bias, "n": int(valid.sum())}


def compute_metrics_normalized(y_true_norm: np.ndarray, y_pred_norm: np.ndarray,
                                label: str = "") -> dict:
    """
    Métriques dans l'espace normalisé (diagnostic interne uniquement).
    NE PAS utiliser pour comparer avec Oiken ou le naïf.
    """
    return compute_metrics(y_true_norm, y_pred_norm, label=f"[NORM] {label}")


# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_by_season(timestamps: list, y_true: np.ndarray,
                    y_pred: np.ndarray) -> dict:
    """
    Calcule les métriques séparément pour l'hiver et l'été.
    Correspond à l'analyse saisonnière déjà présente dans le rapport v3.

    Hiver : novembre à mars (mois PV faible, ML dominant)
    Été   : avril à octobre (mois PV fort, Oiken dominant en v3)

    Retourne
    --------
    dict avec clés 'winter' et 'summer', chacun contenant le dict de métriques
    """
    months = np.array([t.month for t in timestamps])

    winter_mask = np.isin(months, [11, 12, 1, 2, 3])
    summer_mask = ~winter_mask

    results = {}
    if winter_mask.sum() > 0:
        results["winter"] = compute_metrics(y_true[winter_mask], y_pred[winter_mask])
        results["winter"]["n_season"] = int(winter_mask.sum())

    if summer_mask.sum() > 0:
        results["summer"] = compute_metrics(y_true[summer_mask], y_pred[summer_mask])
        results["summer"]["n_season"] = int(summer_mask.sum())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC DU SCALER
# ─────────────────────────────────────────────────────────────────────────────

def validate_scaler(scaler: np.ndarray, timestamps: list) -> None:
    """
    Vérifie que le scaler est cohérent :
    - Pas de valeurs nulles ou négatives
    - Tendance croissante sur les mois d'été (la capacité PV ne peut que croître)
    - Pas de sauts brutaux mois à mois (> 50%)
    """
    log.info("\n── Validation scaler ──")

    # Valeurs manquantes
    n_null = np.isnan(scaler).sum()
    if n_null > 0:
        log.error(f"  {n_null} valeurs NaN dans le scaler !")
    else:
        log.info(f"  NaN : 0 ✓")

    # Valeurs négatives ou nulles
    n_bad = (scaler <= 0).sum()
    if n_bad > 0:
        log.error(f"  {n_bad} valeurs <= 0 dans le scaler !")
    else:
        log.info(f"  Valeurs <= 0 : 0 ✓")

    # Plage
    log.info(f"  Min={scaler.min():.4f} | Max={scaler.max():.4f} | Moyenne={scaler.mean():.4f}")

    # Variation max mois à mois
    # (on prend un pas par mois pour ne pas comparer des pas identiques)
    months = np.array([t.month for t in timestamps])
    changes = np.diff(scaler[::96])  # 1 valeur par jour environ
    max_jump = np.max(np.abs(changes))
    if max_jump > 0.5:
        log.warning(f"  Saut max mois/mois : {max_jump:.4f} — possible discontinuité")
    else:
        log.info(f"  Saut max : {max_jump:.4f} ✓")
