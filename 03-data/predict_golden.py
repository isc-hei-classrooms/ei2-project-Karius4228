"""
predict_golden.py — Inférence v4 sur le golden dataset (CORRIGÉ — 4 zones)
──────────────────────────────────────────────────────────────────────────
Charge xgb_da_v4 + lgb_da_v4, prédit sur golden_features.parquet, dénormalise,
puis calcule les métriques sur 4 zones temporelles distinctes :

  in_train  : ts ≤ 2024-11-06    (dans train original — sanity check stabilité)
  in_test   : 2024-11-07 → 2025-09-29  (dans test set original — déjà connu)
  unseen    : 2025-09-30 → fin   (VRAIMENT NOUVEAU — métrique principale)
  B_12mois  : 2025-04-01 → fin   (12 mois cycle complet, chevauche in_test+unseen)

Naïf J-1 : load_lag_1d × pv_scaler_target (charge raw J-1 ré-ajustée au PV J+1)

Sortie :
  03-data/results/golden_predictions.joblib
  03-data/results/golden_metrics.json

Auteur : Marius Fabbri
"""

import numpy as np
import polars as pl
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228")
DST_ROOT      = PROJECT_ROOT / "03-data"
DST_PROCESSED = DST_ROOT / "data" / "processed"
DST_MODELS    = DST_ROOT / "models"
DST_RESULTS   = DST_ROOT / "results"
DST_RESULTS.mkdir(parents=True, exist_ok=True)

# ── Bornes des 4 zones ────────────────────────────────────────────────────────
from datetime import datetime, timezone
TRAIN_END   = datetime(2024, 11, 6, 19, 30, tzinfo=timezone.utc)
TEST_END    = datetime(2025, 9, 29, 22, 0,  tzinfo=timezone.utc)
B_START     = datetime(2025, 3, 31, 22, 0,  tzinfo=timezone.utc)  # 2025-04-01 00:00 locale ≈ 22:00 UTC du 31 mars

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]
    if len(yt) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"),
                "bias": float("nan"), "n": 0}
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    bias = float(np.mean(yp - yt))
    mask = np.abs(yt) > 0.1
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.sum() > 0 else float("nan")
    if label:
        log.info(f"  {label:<22} MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.1f}%  bias={bias:+.4f}  n={valid.sum()}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "bias": bias, "n": int(valid.sum())}


def compute_seasonal(timestamps, y_true, y_pred):
    months = np.array([t.month for t in timestamps])
    winter = np.isin(months, [11, 12, 1, 2, 3])
    summer = ~winter
    results = {}
    if winter.sum() > 0:
        results["winter"] = compute_metrics(y_true[winter], y_pred[winter])
        results["winter"]["n_season"] = int(winter.sum())
    if summer.sum() > 0:
        results["summer"] = compute_metrics(y_true[summer], y_pred[summer])
        results["summer"]["n_season"] = int(summer.sum())
    return results


def compute_monthly(timestamps, y_true, y_pred):
    months = np.array([t.month for t in timestamps])
    years  = np.array([t.year  for t in timestamps])
    results = {}
    for y in sorted(set(years)):
        for m in sorted(set(months[years == y])):
            mask = (years == y) & (months == m)
            if mask.sum() < 10:
                continue
            key = f"{y}-{m:02d}"
            results[key] = compute_metrics(y_true[mask], y_pred[mask])
            results[key]["n"] = int(mask.sum())
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE PAR ZONE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_zone(zone_name: str, mask: np.ndarray, timestamps_full: list,
                 y_true_raw, y_oiken, y_xgb_raw, y_lgb_raw, y_naive):
    """Calcule métriques globales + saisonnières + mensuelles sur une zone."""
    n = mask.sum()
    log.info(f"\n{'='*65}")
    log.info(f"ZONE : {zone_name}  (n={n})")
    log.info(f"{'='*65}")

    if n < 10:
        log.warning(f"  Zone trop petite (n={n}) — skip")
        return None

    ts_zone = [timestamps_full[i] for i in range(len(timestamps_full)) if mask[i]]
    yt = y_true_raw[mask]
    yo = y_oiken[mask]
    yx = y_xgb_raw[mask]
    yl = y_lgb_raw[mask]
    yn = y_naive[mask]

    log.info(f"  Période : {min(ts_zone)} → {max(ts_zone)}")
    log.info(f"\n  Métriques globales :")
    m_naive = compute_metrics(yt, yn, "Naïf J-1")
    m_oiken = compute_metrics(yt, yo, "Oiken")
    m_xgb   = compute_metrics(yt, yx, "XGBoost v4")
    m_lgb   = compute_metrics(yt, yl, "LightGBM v4")

    gain_xgb = (m_oiken["MAE"] - m_xgb["MAE"]) / m_oiken["MAE"] * 100 if m_oiken["MAE"] > 0 else float("nan")
    gain_lgb = (m_oiken["MAE"] - m_lgb["MAE"]) / m_oiken["MAE"] * 100 if m_oiken["MAE"] > 0 else float("nan")
    log.info(f"\n  Gain XGB vs Oiken : {gain_xgb:+.1f}%")
    log.info(f"  Gain LGB vs Oiken : {gain_lgb:+.1f}%")

    log.info(f"\n  Saisonnier :")
    s_oiken = compute_seasonal(ts_zone, yt, yo)
    s_xgb   = compute_seasonal(ts_zone, yt, yx)
    s_lgb   = compute_seasonal(ts_zone, yt, yl)
    for season in ["winter", "summer"]:
        if season in s_oiken:
            mo = s_oiken[season]["MAE"]
            mx = s_xgb.get(season, {}).get("MAE", float("nan"))
            ml = s_lgb.get(season, {}).get("MAE", float("nan"))
            gx = (mo - mx) / mo * 100 if mo > 0 else float("nan")
            gl = (mo - ml) / mo * 100 if mo > 0 else float("nan")
            log.info(f"    {season:<7} Oiken={mo:.4f}  XGB={mx:.4f} ({gx:+.1f}%)  LGB={ml:.4f} ({gl:+.1f}%)  n={s_oiken[season]['n_season']}")

    monthly_oiken = compute_monthly(ts_zone, yt, yo)
    monthly_xgb   = compute_monthly(ts_zone, yt, yx)
    monthly_lgb   = compute_monthly(ts_zone, yt, yl)

    return {
        "n": int(n),
        "period_start": str(min(ts_zone)),
        "period_end"  : str(max(ts_zone)),
        "global": {
            "naive" : m_naive,
            "oiken" : m_oiken,
            "xgb_v4": m_xgb,
            "lgb_v4": m_lgb,
            "gain_xgb_vs_oiken_pct": gain_xgb,
            "gain_lgb_vs_oiken_pct": gain_lgb,
        },
        "seasonal": {"oiken": s_oiken, "xgb_v4": s_xgb, "lgb_v4": s_lgb},
        "monthly":  {"oiken": monthly_oiken, "xgb_v4": monthly_xgb, "lgb_v4": monthly_lgb},
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 65)
    log.info("INFÉRENCE GOLDEN DATASET — v4 (4 ZONES)")
    log.info("=" * 65)

    # ── Chargement features ───────────────────────────────────────────────────
    log.info("\n── Chargement golden_features.parquet ──")
    df = pl.read_parquet(DST_PROCESSED / "golden_features_v3.parquet")
    log.info(f"  Shape : {df.shape}")
    log.info(f"  Période : {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── Features ──────────────────────────────────────────────────────────────
    exclude = {
        "timestamp", "target_normalized", "target_raw",
        "forecast_load_target", "forecast_load", "net_load", "load_raw",
        "load_normalized", "load",
        "pv_local_kwh", "pv_remote_kwh", "pv_central_kwh", "pv_sion_kwh", "pv_sierre_kwh",
        "temperature_c", "radiation_wm2", "precipitation_mm", "humidity_pct",
        "sunshine_min", "wind_speed_ms",
        "pred_wind_ctrl", "pred_wind_std",
        "pred_temperature_std", "pred_radiation_std",
        "pred_temperature_ctrl", "pred_radiation_ctrl",
        "pred_precipitation_ctrl", "pred_humidity_ctrl", "pred_sunshine_ctrl",
    }
    feature_cols = sorted([c for c in df.columns
                           if c not in exclude and not c.endswith("_gap")])
    if "pv_scaler" in df.columns and "pv_scaler" not in feature_cols:
        feature_cols = sorted(feature_cols + ["pv_scaler"])

    log.info(f"  Features utilisées : {len(feature_cols)}")

    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y_true_raw    = df["target_raw"].to_numpy()
    y_true_norm   = df["target_normalized"].to_numpy()
    scaler_target = df["pv_scaler_target"].to_numpy()
    y_oiken       = df["forecast_load_target"].to_numpy()
    timestamps    = df["timestamp"].to_list()

    log.info(f"  X shape : {X.shape}  NaN : {np.isnan(X).sum()}")

    # ── Modèles ───────────────────────────────────────────────────────────────
    log.info("\n── Chargement modèles ──")
    xgb_model = joblib.load(DST_MODELS / "xgb_da_v4.joblib")
    lgb_model = joblib.load(DST_MODELS / "lgb_da_v4.joblib")
    log.info("  ✓ xgb_da_v4.joblib")
    log.info("  ✓ lgb_da_v4.joblib")

    # Vérification ordre features
    if hasattr(xgb_model, 'feature_names_in_'):
        train_features = list(xgb_model.feature_names_in_)
        if train_features != feature_cols:
            missing = set(train_features) - set(feature_cols)
            extra   = set(feature_cols) - set(train_features)
            if missing:
                log.warning(f"  Features manquantes vs train : {missing}")
            if extra:
                log.warning(f"  Features en trop vs train : {extra}")
            log.info("  Réordonnancement features selon le train...")
            feature_cols = train_features
            X = df.select(feature_cols).to_numpy().astype(np.float32)
        else:
            log.info("  ✓ Features identiques au train (ordre + noms)")

    # ── Prédiction ────────────────────────────────────────────────────────────
    log.info("\n── Prédiction (modèle.predict, AUCUN fit) ──")
    y_xgb_norm = xgb_model.predict(X)
    y_lgb_norm = lgb_model.predict(X)
    log.info(f"  XGB norm : mean={y_xgb_norm.mean():.4f}  std={y_xgb_norm.std():.4f}")
    log.info(f"  LGB norm : mean={y_lgb_norm.mean():.4f}  std={y_lgb_norm.std():.4f}")

    # ── Dénormalisation ───────────────────────────────────────────────────────
    log.info("\n── Dénormalisation ──")
    y_xgb_raw = y_xgb_norm * scaler_target
    y_lgb_raw = y_lgb_norm * scaler_target
    log.info(f"  XGB raw : mean={y_xgb_raw.mean():.4f}  std={y_xgb_raw.std():.4f}")
    log.info(f"  LGB raw : mean={y_lgb_raw.mean():.4f}  std={y_lgb_raw.std():.4f}")
    log.info(f"  Vrai    : mean={y_true_raw.mean():.4f}  std={y_true_raw.std():.4f}")

    # ── Naïf J-1 (interprétation 1) ───────────────────────────────────────────
    # load_lag_1d (normalisé) × pv_scaler_target (J+1)
    # → "charge d'hier ré-ajustée au niveau PV de demain"
    y_naive = df["load_lag_1d"].to_numpy() * scaler_target

    # ── Construction des masques de zones ─────────────────────────────────────
    log.info("\n── Découpage en 4 zones ──")
    ts_arr = np.array(timestamps)

    mask_in_train = ts_arr <= TRAIN_END
    mask_in_test  = (ts_arr > TRAIN_END) & (ts_arr <= TEST_END)
    mask_unseen   = ts_arr > TEST_END
    mask_B        = ts_arr >= B_START

    log.info(f"  in_train  : {mask_in_train.sum():>6}  (≤ {TRAIN_END.date()})")
    log.info(f"  in_test   : {mask_in_test.sum():>6}  ({TRAIN_END.date()} → {TEST_END.date()})")
    log.info(f"  unseen    : {mask_unseen.sum():>6}  (> {TEST_END.date()})")
    log.info(f"  B_12mois  : {mask_B.sum():>6}  (≥ {B_START.date()})")

    # ── Analyse par zone ──────────────────────────────────────────────────────
    zones = {}
    for name, mask in [("in_train", mask_in_train),
                       ("in_test",  mask_in_test),
                       ("unseen",   mask_unseen),
                       ("B_12mois", mask_B)]:
        result = analyze_zone(name, mask, timestamps,
                              y_true_raw, y_oiken, y_xgb_raw, y_lgb_raw, y_naive)
        if result is not None:
            zones[name] = result

    # ── Résumé final ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("RÉSUMÉ — MAE par zone")
    log.info("=" * 65)
    log.info(f"  {'Zone':<10} {'n':>7} {'Naïf':>8} {'Oiken':>8} {'XGB':>8} {'LGB':>8} {'GainXGB':>10} {'GainLGB':>10}")
    for zname, zdata in zones.items():
        g = zdata["global"]
        log.info(f"  {zname:<10} {zdata['n']:>7} "
                 f"{g['naive']['MAE']:>8.4f} {g['oiken']['MAE']:>8.4f} "
                 f"{g['xgb_v4']['MAE']:>8.4f} {g['lgb_v4']['MAE']:>8.4f} "
                 f"{g['gain_xgb_vs_oiken_pct']:>+9.1f}% {g['gain_lgb_vs_oiken_pct']:>+9.1f}%")

    # ── Sauvegarde résultats ──────────────────────────────────────────────────
    log.info("\n── Sauvegarde ──")

    predictions = {
        "timestamps"    : timestamps,
        "y_true_raw"    : y_true_raw,
        "y_true_norm"   : y_true_norm,
        "y_xgb_raw"     : y_xgb_raw,
        "y_lgb_raw"     : y_lgb_raw,
        "y_xgb_norm"    : y_xgb_norm,
        "y_lgb_norm"    : y_lgb_norm,
        "y_oiken"       : y_oiken,
        "y_naive"       : y_naive,
        "scaler_target" : scaler_target,
        "feature_cols"  : feature_cols,
        "zones": {
            "TRAIN_END" : str(TRAIN_END),
            "TEST_END"  : str(TEST_END),
            "B_START"   : str(B_START),
            "mask_in_train": mask_in_train,
            "mask_in_test" : mask_in_test,
            "mask_unseen"  : mask_unseen,
            "mask_B"       : mask_B,
        },
    }
    joblib.dump(predictions, DST_RESULTS / "golden_predictions.joblib")
    log.info(f"  ✓ golden_predictions.joblib")

    metrics = {
        "zones": zones,
        "meta": {
            "n_total"     : len(y_true_raw),
            "period_start": str(df["timestamp"].min()),
            "period_end"  : str(df["timestamp"].max()),
            "n_features"  : len(feature_cols),
            "TRAIN_END"   : str(TRAIN_END),
            "TEST_END"    : str(TEST_END),
            "B_START"     : str(B_START),
        }
    }
    with open(DST_RESULTS / "golden_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  ✓ golden_metrics.json")

    log.info("\n" + "=" * 65)
    log.info("✓ Inférence terminée")
    log.info("\nProchaine étape : lancer golden_marimo.py (ou son équivalent)")
    log.info("=" * 65)

    return predictions, metrics


if __name__ == "__main__":
    predictions, metrics = run()