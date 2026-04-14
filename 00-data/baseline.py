"""
baseline.py — Baselines naïves de référence (DA et Intraday)
Auteur : Marius Fabbri

Baseline DA      : load_lag_1d → charge du même pas 15 min, la veille (J-1)
Baseline Intraday: load_lag_96 → charge du même pas J-1 (seul lag légal anti-leakage)

Ces baselines représentent la prédiction "sans modèle". Le tableau de comparaison
final (dans model_da.py) opposera :
  - Baseline naïve J-1  (ce fichier)
  - Forecast Oiken       (forecast_load, autre modèle opérationnel)
  - Nouveau modèle ML    (XGBoost / LightGBM)

Remarque MAPE : la charge est en z-score (mean≈0, std≈1). Les valeurs proches de 0
(nuits, creux) rendent le MAPE instable. Un seuil |y| > 0.1 filtre ces points, ce
qui sous-estime le nombre réel d'observations exclues. Préférer MAE/RMSE pour
les comparaisons inter-modèles.
"""

import polars as pl
import numpy as np
from pathlib import Path
import logging
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import FEATURES_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES
# ──────────────────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcule MAE, RMSE et MAPE en ignorant les paires où l'un des deux est NaN.
    MAPE calculé uniquement sur |y_true| > 0.1 (charge normalisée proche de 0
    → division instable).
    """
    # Filtrage NaN (lags manquants en début de série ou gaps)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() == 0:
        log.warning("  Aucune paire valide (tout NaN) — métriques non calculables.")
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"), "n": 0}

    yt = y_true[valid]
    yp = y_pred[valid]

    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    mape_mask = np.abs(yt) > 0.1
    n_excluded = valid.sum() - mape_mask.sum()
    if mape_mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((yt[mape_mask] - yp[mape_mask]) / yt[mape_mask])) * 100)

    return {
        "MAE" : mae,
        "RMSE": rmse,
        "MAPE": mape,
        "n"   : int(valid.sum()),
        "n_mape_excluded": int(n_excluded),
    }


def _print_metrics(m: dict, label: str) -> None:
    n_excl = m.get("n_mape_excluded", 0)
    mape_str = f"{m['MAPE']:.2f}% (excl. {n_excl} pts)" if not np.isnan(m["MAPE"]) else "N/A"
    log.info(f"  {label:30s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MAPE={mape_str}")


# ══════════════════════════════════════════════════════════════════════════════
# DAY-AHEAD
# ══════════════════════════════════════════════════════════════════════════════

def baseline_da() -> dict:
    """
    Évalue la baseline naïve J-1 sur le test set DA.

    Returns
    -------
    {"naive_j1": {"MAE": ..., "RMSE": ..., "MAPE": ..., "n": ..., "n_mape_excluded": ...}}
    """
    log.info("\n" + "═"*60)
    log.info("BASELINE — DAY-AHEAD")
    log.info("  Prédiction : load_lag_1d (charge même pas 15 min, J-1)")
    log.info("═"*60)

    df = pl.read_parquet(FEATURES_DIR / "test_da_v2.parquet")
    y_true  = df["target"].to_numpy()
    y_naive = df["load_lag_1d"].to_numpy()

    m = metrics(y_true, y_naive)
    _print_metrics(m, "Naïve J-1")

    return {"naive_j1": m}


# ══════════════════════════════════════════════════════════════════════════════
# INTRADAY
# ══════════════════════════════════════════════════════════════════════════════

def baseline_intraday() -> dict:
    """
    Évalue la baseline naïve J-1 sur les 12 horizons intraday.

    La seule baseline légale (anti-leakage) est load_lag_96 : la charge Oiken
    n'est disponible qu'avec ~2h de délai, donc pas de lag court possible.
    Le forecast_load Oiken n'est pas disponible en intraday.

    Returns
    -------
    {
      1: {"MAE": ..., "RMSE": ..., "MAPE": ..., "n": ...},
      ...
      12: {...},
      "avg": {"MAE": ..., "RMSE": ..., "MAPE": ...},
    }
    """
    log.info("\n" + "═"*60)
    log.info("BASELINE — INTRADAY")
    log.info("  Prédiction : load_lag_96 (charge même pas 15 min, J-1)")
    log.info("  Rappel     : pas de forecast_load Oiken disponible en intraday")
    log.info("═"*60)

    df = pl.read_parquet(FEATURES_DIR / "test_intraday_v2.parquet")
    y_lag = df["load_lag_96"].to_numpy()

    log.info(f"\n  {'Horizon':>8}  {'MAE':>8}  {'RMSE':>8}  {'MAPE':>10}  {'n':>7}")
    log.info("  " + "─"*50)

    results = {}
    for h in range(1, 13):
        col = f"target_{h}"
        if col not in df.columns:
            log.warning(f"  Colonne {col} absente — horizon ignoré.")
            continue
        y_true = df[col].to_numpy()
        m = metrics(y_true, y_lag)
        results[h] = m
        mape_str = f"{m['MAPE']:>7.2f}%" if not np.isnan(m["MAPE"]) else "    N/A"
        log.info(f"  t+{h:2d}     {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {mape_str}  {m['n']:>7}")

    if not results:
        log.error("Aucun horizon trouvé dans le test set.")
        return results

    # ── Moyenne sur les horizons disponibles ──────────────────────────────────
    log.info("  " + "─"*50)
    avg_mae  = float(np.mean([r["MAE"]  for r in results.values()]))
    avg_rmse = float(np.mean([r["RMSE"] for r in results.values()]))
    valid_mapes = [r["MAPE"] for r in results.values() if not np.isnan(r["MAPE"])]
    avg_mape = float(np.mean(valid_mapes)) if valid_mapes else float("nan")

    log.info(f"  {'Moyenne':>8}  {avg_mae:>8.4f}  {avg_rmse:>8.4f}  {avg_mape:>7.2f}%")
    results["avg"] = {"MAE": avg_mae, "RMSE": avg_rmse, "MAPE": avg_mape}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    da_results = baseline_da()
    id_results = baseline_intraday()

    # Sauvegarde JSON pour référence lors de l'évaluation des modèles
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Structure : da.naive_j1, intraday.1..12, intraday.avg
    output = {"da": da_results, "intraday": id_results}
    out_path = MODELS_DIR / "baseline_results.json"

    # Conversion des int numpy pour la sérialisation JSON
    def _to_python(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Type non sérialisable : {type(obj)}")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_to_python)
    log.info(f"\n✓ Résultats sauvegardés : {out_path}")
