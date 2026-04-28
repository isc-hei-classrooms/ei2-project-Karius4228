"""
calibrate_golden_v2.py — Calibration espace golden → espace train (VERSION UTC)
─────────────────────────────────────────────────────────────────────────────────
Différences vs calibrate_golden.py (v1) :
  - Joint sur timestamps UTC (Datetime[us, UTC]) — cohérent avec le pipeline v2
  - Utilise train_da_v4.parquet directement (les colonnes load_raw et
    forecast_load_target sont déjà dans l'espace train final)
  - Calibre load_G  → load_raw_T     (z-score brut, mean≈0, std≈0.93)
  - Calibre fcst_G  → forecast_load_T (même espace)

Sortie :
  03-data/data/processed/golden_calibration_v2.joblib
    {
      "load":     {"a", "b", "inv_a", "inv_b", "n", "residual_std"},
      "forecast": {"a", "b", "inv_a", "inv_b", "n", "residual_std"},
      "overlap_utc": (start, end),
      "note": "Fitté sur timestamps UTC communs golden_features_v2 vs train_da_v4"
    }

Usage :
  python calibrate_golden_v2.py

Auteur : Marius Fabbri
"""

import polars as pl
import numpy as np
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJ          = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228")
TRAIN_PARQUET = PROJ / "01-data" / "data" / "processed" / "features_v4" / "train_da_v4.parquet"
GOLDEN_FEATS  = PROJ / "03-data" / "data" / "processed" / "golden_features_v2.parquet"
OUTPUT        = PROJ / "03-data" / "data" / "processed" / "golden_calibration_v2.joblib"


def fit_linear(y_T: np.ndarray, y_G: np.ndarray, label: str) -> dict:
    """
    Fitte y_G = a * y_T + b.
    Retourne aussi l'inverse : y_T = inv_a * y_G + inv_b.
    """
    mask = ~(np.isnan(y_T) | np.isnan(y_G))
    yt, yg = y_T[mask], y_G[mask]
    a, b = np.polyfit(yt, yg, 1)
    residuals = yg - (a * yt + b)
    res_std = float(residuals.std())
    res_max = float(np.abs(residuals).max())
    inv_a = float(1.0 / a)
    inv_b = float(-b / a)

    log.info(f"  [{label}]")
    log.info(f"    Forward  : y_G = {a:.6f} * y_T + {b:+.6f}")
    log.info(f"    Inverse  : y_T = {inv_a:.6f} * y_G + {inv_b:+.6f}")
    log.info(f"    Résidu   : std={res_std:.6f}  max|res|={res_max:.6f}  n={len(yt)}")
    if res_std > 0.05:
        log.warning(f"    ⚠ Résidu std={res_std:.4f} > 0.05 — relation non purement linéaire")

    return {
        "a": float(a), "b": float(b),
        "inv_a": inv_a, "inv_b": inv_b,
        "n": int(len(yt)),
        "residual_std": res_std,
        "residual_max": res_max,
    }


def run():
    log.info("=" * 65)
    log.info("CALIBRATION GOLDEN v2 — ESPACE UTC")
    log.info("=" * 65)

    # ── Chargement train (déjà en UTC, espace z-score final) ──────────────────
    log.info("\n── Train : train_da_v4.parquet ──")
    df_t = pl.read_parquet(TRAIN_PARQUET)
    log.info(f"  Shape  : {df_t.shape}")
    log.info(f"  Période: {df_t['timestamp'].min()} → {df_t['timestamp'].max()}")
    # Colonnes cibles dans l'espace train :
    #   load_raw           = z-score brut de la charge (mean≈0, std≈0.93)
    #   forecast_load_target = forecast Oiken dans le même espace, décalé +24h
    # On veut calibrer golden.load → train.load_raw
    #                    golden.forecast_load_target → train.forecast_load_target
    df_t_sub = df_t.select(["timestamp", "load_raw", "forecast_load_target"]).rename({
        "load_raw":              "load_T",
        "forecast_load_target":  "fcst_T",
    })

    # ── Chargement golden features v2 (UTC) ───────────────────────────────────
    log.info("\n── Golden : golden_features_v2.parquet ──")
    df_g = pl.read_parquet(GOLDEN_FEATS)
    log.info(f"  Shape  : {df_g.shape}")
    log.info(f"  Période: {df_g['timestamp'].min()} → {df_g['timestamp'].max()}")
    # load_raw dans le golden = load brut du CSV (golden z-score, non calibré)
    # forecast_load_target = forecast brut décalé +24h (golden z-score)
    df_g_sub = df_g.select(["timestamp", "load_raw", "forecast_load_target"]).rename({
        "load_raw":              "load_G",
        "forecast_load_target":  "fcst_G",
    })

    # ── Join UTC ───────────────────────────────────────────────────────────────
    log.info("\n── Join sur timestamps UTC communs ──")
    df_join = df_t_sub.join(df_g_sub, on="timestamp", how="inner").sort("timestamp")
    log.info(f"  Lignes communes : {df_join.height}")
    log.info(f"  Overlap         : {df_join['timestamp'].min()} → {df_join['timestamp'].max()}")

    if df_join.height < 1000:
        raise ValueError(f"Trop peu de lignes communes ({df_join.height}) — vérifier les timestamps UTC")

    # ── Fit linéaire ──────────────────────────────────────────────────────────
    log.info("\n── Fit linéaire (y_G = a * y_T + b) ──")
    cal_load = fit_linear(
        df_join["load_T"].to_numpy(),
        df_join["load_G"].to_numpy(),
        "load_raw",
    )
    cal_fcst = fit_linear(
        df_join["fcst_T"].to_numpy(),
        df_join["fcst_G"].to_numpy(),
        "forecast_load_target",
    )

    # ── Vérification inverse ───────────────────────────────────────────────────
    log.info("\n── Vérification : (y_G → espace train) ≈ y_T ──")
    load_G_to_T = cal_load["inv_a"] * df_join["load_G"].to_numpy() + cal_load["inv_b"]
    delta = load_G_to_T - df_join["load_T"].to_numpy()
    log.info(f"  load  delta : mean={delta.mean():+.6f}  std={delta.std():.6f}  max|.|={np.abs(delta).max():.6f}")

    fcst_G_to_T = cal_fcst["inv_a"] * df_join["fcst_G"].to_numpy() + cal_fcst["inv_b"]
    delta_f = fcst_G_to_T - df_join["fcst_T"].to_numpy()
    log.info(f"  fcst  delta : mean={delta_f.mean():+.6f}  std={delta_f.std():.6f}  max|.|={np.abs(delta_f).max():.6f}")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    calibration = {
        "load":     cal_load,
        "forecast": cal_fcst,
        "n_overlap": df_join.height,
        "overlap_utc": (str(df_join["timestamp"].min()), str(df_join["timestamp"].max())),
        "note": "Fitté en UTC sur golden_features_v2 vs train_da_v4 (load_raw, forecast_load_target)",
    }
    joblib.dump(calibration, OUTPUT)
    log.info(f"\n✓ Sauvegardé : {OUTPUT}")
    log.info("\n  Transformations inverses (golden → espace train) :")
    log.info(f"    load_T     = {cal_load['inv_a']:.6f} * load_G     + {cal_load['inv_b']:+.6f}")
    log.info(f"    forecast_T = {cal_fcst['inv_a']:.6f} * forecast_G + {cal_fcst['inv_b']:+.6f}")
    log.info("=" * 65)
    log.info("\nProchaine étape : build_golden_parquet_v2_calibrated.py")


if __name__ == "__main__":
    run()