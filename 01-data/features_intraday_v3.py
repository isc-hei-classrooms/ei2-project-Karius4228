"""
features_intraday_v3.py — Features Intraday CORRIGÉES
─────────────────────────────────────────────────────
Mêmes corrections que DA v3 :
  1. Forward-fill des NWP AVANT les shifts
  2. Shifts TEMPORELS via join sur timestamp décalé
  3. Filtrage minimal + imputation médiane

Contraintes temporelles (prédiction à l'instant t) :
  - Charge Oiken   : livrée à 2h J → lags ≥ J-1 (24h)
  - PV local       : dispo ~15 min → lags courts OK (t-15min, t-1h)
  - PV remote      : livré à 2h J → lags ≥ J-1
  - Mesures météo  : délai 1h → shift 4 pas = t-1h
  - NWP            : shift(-h) appliqué par horizon à l'entraînement

Auteur : Marius Fabbri (corrigé avec audit Claude)
"""

import polars as pl
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
COL_TIMESTAMP     = "timestamp"
COL_LOAD          = "load"
COL_FORECAST_LOAD = "forecast_load"
COL_PV_LOCAL      = "pv_local_kwh"
COL_PV_REMOTE     = "pv_remote_kwh"
COL_PV_CENTRAL    = "pv_central_kwh"
COL_PV_SIERRE     = "pv_sierre_kwh"
COL_TEMP          = "temperature_c"
COL_GLOB          = "radiation_wm2"
COL_PRECIP        = "precipitation_mm"
COL_HUMIDITY      = "humidity_pct"

NWP_COLS = [
    "pred_temperature_ctrl", "pred_radiation_ctrl", "pred_precipitation_ctrl",
    "pred_humidity_ctrl", "pred_sunshine_ctrl",
]

METEO_REAL_COLS = [COL_TEMP, COL_GLOB, COL_HUMIDITY, COL_PRECIP]

HORIZON_MAX = 12  # 12 pas = 3h
TRAIN_RATIO = 0.70
DELTA_1D = timedelta(hours=24)
DELTA_7D = timedelta(days=7)
DELTA_15MIN = timedelta(minutes=15)
DELTA_1H = timedelta(hours=1)

HOLIDAYS_FIXED = {(1,1),(1,2),(3,19),(5,1),(8,1),(8,15),(11,1),(12,8),(12,25),(12,26)}
HOLIDAYS_MOBILE = {
    "2022-04-15","2022-04-18","2022-05-26","2022-06-06",
    "2023-04-07","2023-04-10","2023-05-18","2023-05-29",
    "2024-03-29","2024-04-01","2024-05-09","2024-05-20",
    "2025-04-18","2025-04-21","2025-05-29","2025-06-09",
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES — SHIFT TEMPOREL
# ══════════════════════════════════════════════════════════════════════════════

def temporal_shift(df: pl.DataFrame, col: str, delta: timedelta, alias: str) -> pl.DataFrame:
    """Récupère la valeur de `col` au timestamp (t - delta)."""
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) + delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")


def temporal_shift_forward(df: pl.DataFrame, col: str, delta: timedelta, alias: str) -> pl.DataFrame:
    """Récupère la valeur de `col` au timestamp (t + delta)."""
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) - delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")


# ══════════════════════════════════════════════════════════════════════════════
# GROUPES DE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_load_lags(df: pl.DataFrame) -> pl.DataFrame:
    """Charge livrée à 2h J → lags J-1 et J-7 uniquement."""
    log.info("  [Groupe 1] Lags charge (J-1, J-7) — shift temporel...")
    df = temporal_shift(df, COL_LOAD, DELTA_1D, "load_lag_96")
    df = temporal_shift(df, COL_LOAD, DELTA_7D, "load_lag_672")
    return df


def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Rolling sur données J-1 (grille régulière → shift(96) OK)."""
    log.info("  [Groupe 2] Rolling mean/std (24h, 7j) — sur données J-1...")
    shifted = pl.col(COL_LOAD).shift(96)
    return df.with_columns([
        shifted.rolling_mean(window_size=96).alias("rolling_mean_24h"),
        shifted.rolling_std(window_size=96).alias("rolling_std_24h"),
        shifted.rolling_mean(window_size=672).alias("rolling_mean_7d"),
    ])


def add_pv_lags(df: pl.DataFrame) -> pl.DataFrame:
    """
    PV local : lags courts (t-15min, t-1h) + longs (J-1, J-7).
    PV remote : lags longs uniquement.
    """
    log.info("  [Groupe 3] PV local (t-15min, t-1h, J-1, J-7) + PV remote (J-1, J-7)...")
    if COL_PV_LOCAL in df.columns:
        df = temporal_shift(df, COL_PV_LOCAL, DELTA_15MIN, "pv_local_lag_1")
        df = temporal_shift(df, COL_PV_LOCAL, DELTA_1H, "pv_local_lag_4")
        df = temporal_shift(df, COL_PV_LOCAL, DELTA_1D, "pv_local_lag_96")
        df = temporal_shift(df, COL_PV_LOCAL, DELTA_7D, "pv_local_lag_672")
        # Rolling PV local sur dernière heure
        df = df.with_columns(
            pl.col(COL_PV_LOCAL).shift(1).rolling_mean(window_size=4)
            .alias("pv_local_rolling_mean_1h")
        )
    if COL_PV_REMOTE in df.columns:
        df = temporal_shift(df, COL_PV_REMOTE, DELTA_1D, "pv_remote_lag_96")
        df = temporal_shift(df, COL_PV_REMOTE, DELTA_7D, "pv_remote_lag_672")
    return df


def add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
    log.info("  [Groupe 4] Encodage cyclique...")
    ts = pl.col(COL_TIMESTAMP)
    hour_frac = ts.dt.hour() + ts.dt.minute() / 60.0
    return df.with_columns([
        (2 * np.pi * hour_frac / 24).sin().alias("hour_sin"),
        (2 * np.pi * hour_frac / 24).cos().alias("hour_cos"),
        (2 * np.pi * ts.dt.weekday() / 7).sin().alias("weekday_sin"),
        (2 * np.pi * ts.dt.weekday() / 7).cos().alias("weekday_cos"),
        (2 * np.pi * (ts.dt.month() - 1) / 12).sin().alias("month_sin"),
        (2 * np.pi * (ts.dt.month() - 1) / 12).cos().alias("month_cos"),
    ])


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    log.info("  [Groupe 5] Variables calendaires...")
    ts = pl.col(COL_TIMESTAMP)
    is_weekend = (ts.dt.weekday() >= 5).alias("is_weekend")
    fixed_list = [f"{m}-{d}" for m, d in HOLIDAYS_FIXED]
    date_md = ts.dt.month().cast(pl.Utf8) + "-" + ts.dt.day().cast(pl.Utf8)
    date_full = (ts.dt.year().cast(pl.Utf8) + "-"
                 + ts.dt.month().cast(pl.Utf8).str.zfill(2) + "-"
                 + ts.dt.day().cast(pl.Utf8).str.zfill(2))
    is_holiday = (date_md.is_in(fixed_list) | date_full.is_in(list(HOLIDAYS_MOBILE))).alias("is_holiday")
    month = ts.dt.month()
    return df.with_columns([
        is_weekend, is_holiday,
        ((month == 12) | (month <= 2)).alias("is_winter"),
        ((month >= 3) & (month <= 5)).alias("is_spring"),
        ((month >= 6) & (month <= 8)).alias("is_summer"),
        ((month >= 9) & (month <= 11)).alias("is_autumn"),
    ])


def add_meteo_recent(df: pl.DataFrame) -> pl.DataFrame:
    """Mesures météo avec délai réel de 1h (shift temporel 1h)."""
    log.info("  [Groupe 6] Météo récente (t-1h, shift temporel)...")
    for c in METEO_REAL_COLS:
        if c in df.columns:
            df = temporal_shift(df, c, DELTA_1H, f"{c}_recent")
    return df


def add_nwp_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    NWP conservées BRUTES après forward-fill.
    Le shift(-h) sera appliqué par build_nwp_for_horizon() à l'entraînement.
    """
    log.info("  [Groupe 7] NWP brutes (forward-fill, shift par horizon à l'entraînement)...")
    available = [c for c in NWP_COLS if c in df.columns]
    # Forward-fill (NWP constantes entre publications)
    df = df.with_columns([pl.col(c).forward_fill().alias(c) for c in available])
    log.info(f"  {len(available)} NWP forward-filled")
    return df


def add_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Interactions sur mesures météo récentes (t-1h)."""
    log.info("  [Groupe 8] Interactions sur mesures récentes...")
    t = f"{COL_TEMP}_recent"
    r = f"{COL_GLOB}_recent"
    exprs = []
    if t in df.columns:
        exprs.append((pl.col(t) ** 2).alias("temp_squared"))
    if t in df.columns and r in df.columns:
        exprs.append((pl.col(t) * pl.col(r)).alias("temp_x_rad"))
    return df.with_columns(exprs) if exprs else df


def build_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Cibles multi-horizon : target_1 (t+15min) à target_12 (t+3h)."""
    log.info(f"  Cibles Y : target_1 à target_{HORIZON_MAX} (shifts temporels)...")
    for h in range(1, HORIZON_MAX + 1):
        delta = timedelta(minutes=15 * h)
        df = temporal_shift_forward(df, COL_LOAD, delta, f"target_{h}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRE POUR L'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def build_nwp_for_horizon(df: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """
    Pour un horizon donné (1 à 12), récupère les NWP au timestamp cible.
    Utilise temporal_shift_forward au lieu de shift(-h).

    Usage dans model_intraday_v3.py :
        df_h = build_nwp_for_horizon(df_train, h=4)
    """
    available = [c for c in NWP_COLS if c in df.columns]
    delta = timedelta(minutes=15 * horizon)
    for c in available:
        df = temporal_shift_forward(df, c, delta, f"{c}_h{horizon}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLAGE
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df_oiken, df_meteo_real, df_meteo_pred):
    log.info("\n" + "=" * 60 + "\nFEATURES INTRADAY (v3 — shifts temporels)\n" + "=" * 60)

    log.info("\nJointure des 3 sources...")
    mr = [COL_TIMESTAMP] + [c for c in df_meteo_real.columns
                             if not c.endswith("_gap") and c != COL_TIMESTAMP]
    mp = [COL_TIMESTAMP] + [c for c in df_meteo_pred.columns
                             if not c.endswith("_gap") and c != COL_TIMESTAMP]
    df = (df_oiken
          .join(df_meteo_real.select(mr), on=COL_TIMESTAMP, how="left")
          .join(df_meteo_pred.select(mp), on=COL_TIMESTAMP, how="left"))
    log.info(f"  Shape après jointure : {df.shape}")

    # Vérifier grille régulière
    diffs = df[COL_TIMESTAMP].diff().dt.total_seconds().drop_nulls()
    n_regular = (diffs == 900).sum()
    log.info(f"  Grille régulière: {n_regular}/{diffs.len()} ({100*n_regular/diffs.len():.1f}%)")

    for fn in [add_load_lags, add_rolling_features, add_pv_lags,
               add_cyclical_features, add_calendar_features,
               add_meteo_recent, add_nwp_features, add_interactions, build_targets]:
        df = fn(df)

    # Filtrage minimal
    n0 = df.height
    df = df.filter(pl.col(f"target_{HORIZON_MAX}").is_not_null())
    log.info(f"\n  Lignes sans target_{HORIZON_MAX} : {n0 - df.height}")

    n1 = df.height
    df = df.filter(pl.col("load_lag_96").is_not_null())
    log.info(f"  Lignes sans load_lag_96 : {n1 - df.height}")

    n2 = df.height
    df = df.filter(pl.col("load_lag_672").is_not_null())
    log.info(f"  Lignes sans load_lag_672 : {n2 - df.height}")

    # Imputation médiane pour les résidus
    feature_cols = get_feature_columns(df)
    for c in feature_cols:
        nc = df[c].null_count()
        if nc > 0:
            med = df[c].drop_nulls().median()
            log.info(f"  Imputation: {c} ({nc} nulls, {100*nc/df.height:.1f}%)")
            df = df.with_columns(pl.col(c).fill_null(med).alias(c))

    log.info(f"  Shape finale : {df.shape}")
    return df


def train_test_split(df):
    i = int(len(df) * TRAIN_RATIO)
    tr, te = df[:i], df[i:]
    log.info(f"\nSplit {TRAIN_RATIO:.0%} : Train {tr.height} | Test {te.height}")
    return tr, te


def get_feature_columns(df):
    exclude = {
        COL_TIMESTAMP, COL_FORECAST_LOAD, "net_load",
        COL_PV_LOCAL, COL_PV_REMOTE, COL_PV_CENTRAL, COL_PV_SIERRE,
        COL_TEMP, COL_GLOB, COL_PRECIP, COL_HUMIDITY,
        "sunshine_min", "wind_speed_ms", COL_LOAD,
        "pred_wind_ctrl", "pred_wind_std",
    }
    return sorted([c for c in df.columns
                   if c not in exclude
                   and not c.endswith("_gap")
                   and not c.startswith("target_")])


def get_target_columns():
    return [f"target_{h}" for h in range(1, HORIZON_MAX + 1)]


def get_nwp_columns():
    return list(NWP_COLS)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_engineering_intraday(data_dir: Path = None):
    if data_dir is None:
        from config import PROCESSED_DIR
        data_dir = PROCESSED_DIR

    log.info("Chargement données nettoyées...")
    df_o = pl.read_parquet(data_dir / "oiken_clean_v2.parquet")
    df_r = pl.read_parquet(data_dir / "meteo_real_clean.parquet")
    df_p = pl.read_parquet(data_dir / "meteo_pred_clean.parquet")

    df = build_feature_matrix(df_o, df_r, df_p)
    tr, te = train_test_split(df)

    out_dir = data_dir / "features_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    tr.write_parquet(out_dir / "train_intraday_v3.parquet")
    te.write_parquet(out_dir / "test_intraday_v3.parquet")
    log.info(f"✓ Sauvegardé dans {out_dir}")

    fc = get_feature_columns(df)
    log.info(f"\n── Intraday v3 : {len(fc)} features, {HORIZON_MAX} cibles ──")
    for c in fc:
        n = df[c].null_count()
        log.info(f"  [{'✓' if n == 0 else f'! {100*n/df.height:.1f}%'}] {c}")

    return tr, te


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    tr, te = run_feature_engineering_intraday(args.data_dir)
    fc = get_feature_columns(tr)
    tc = get_target_columns()
    print(f"\nTrain: {tr.shape} | Test: {te.shape} | Features: {len(fc)} | Targets: {len(tc)}")
    for i, c in enumerate(fc, 1):
        print(f"  {i:2d}. {c}")
