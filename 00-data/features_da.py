"""
src/processing/features_da.py
─────────────────────────────
Construction de la matrice X et du vecteur Y pour le modèle Day-Ahead.

v2 : Séparation PV local (central+sierre) vs PV remote
     - pv_local et pv_remote ont des lags séparés (tous ≥ J-1 pour DA)
     - NWP pointées sur heure cible J+1 (shift -96)
     - Rolling décalés de 96 (anti-leakage)

Contraintes temporelles (gate closure = 11h J) :
  - Charge Oiken  : livrée à 2h J → lags ≥ J-1
  - PV local      : dispo ~15min, mais pour DA on utilise lags ≥ J-1
  - PV remote     : livré à 2h J → lags ≥ J-1
  - Mesures météo : délai 1h, on utilise lag J-1
  - NWP           : shift(-96) = heure cible J+1

Sortie  : data/features/train_da_v2.parquet, test_da_v2.parquet
Auteur : Marius Fabbri
"""

import polars as pl
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import (
    PROCESSED_DIR, FEATURES_DIR,
    COL_TIMESTAMP, COL_LOAD, COL_FORECAST_LOAD,
    COL_PV_LOCAL, COL_PV_REMOTE, COL_PV_CENTRAL, COL_PV_SIERRE,
    COL_TEMP, COL_GLOB, COL_PRECIP, COL_HUMIDITY,
    COL_PRED_TEMP_CTRL, COL_PRED_GLOB_CTRL, COL_PRED_PREC_CTRL,
    COL_PRED_SUN_CTRL, COL_PRED_HUM_CTRL,
    HORIZON_DA, TRAIN_RATIO,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LAG_1D = 96
LAG_2D = 192
LAG_7D = 672

METEO_REAL_COLS = [COL_TEMP, COL_GLOB, COL_HUMIDITY, COL_PRECIP]
NWP_COLS = [COL_PRED_TEMP_CTRL, COL_PRED_GLOB_CTRL, COL_PRED_PREC_CTRL,
            COL_PRED_HUM_CTRL, COL_PRED_SUN_CTRL]

HOLIDAYS_FIXED = {(1,1),(1,2),(3,19),(5,1),(8,1),(8,15),(11,1),(12,8),(12,25),(12,26)}
HOLIDAYS_MOBILE = {
    "2022-04-15","2022-04-18","2022-05-26","2022-06-06",
    "2023-04-07","2023-04-10","2023-05-18","2023-05-29",
    "2024-03-29","2024-04-01","2024-05-09","2024-05-20",
    "2025-04-18","2025-04-21","2025-05-29","2025-06-09",
}

def add_load_lags(df):
    log.info("  [Groupe 1] Lags de charge (J-1, J-2, J-7)...")
    return df.with_columns([
        pl.col(COL_LOAD).shift(LAG_1D).alias("load_lag_1d"),
        pl.col(COL_LOAD).shift(LAG_2D).alias("load_lag_2d"),
        pl.col(COL_LOAD).shift(LAG_7D).alias("load_lag_7d"),
    ])

def add_rolling_features(df):
    log.info("  [Groupe 2] Rolling mean/std (24h, 7j) — sur données J-1...")
    return df.with_columns([
        pl.col(COL_LOAD).shift(LAG_1D).rolling_mean(window_size=LAG_1D).alias("rolling_mean_24h"),
        pl.col(COL_LOAD).shift(LAG_1D).rolling_std(window_size=LAG_1D).alias("rolling_std_24h"),
        pl.col(COL_LOAD).shift(LAG_1D).rolling_mean(window_size=LAG_7D).alias("rolling_mean_7d"),
    ])

def add_pv_lags(df):
    """
    Lags PV séparés : pv_local (central+sierre) et pv_remote.
    Pour le DA, les deux sont disponibles en J-1 (livrés à 2h J).
    On les sépare car profils de production différents.
    """
    log.info("  [Groupe 3] Lags PV local (J-1, J-7) + PV remote (J-1, J-7)...")
    exprs = []
    if COL_PV_LOCAL in df.columns:
        exprs.extend([
            pl.col(COL_PV_LOCAL).shift(LAG_1D).alias("pv_local_lag_1d"),
            pl.col(COL_PV_LOCAL).shift(LAG_7D).alias("pv_local_lag_7d"),
        ])
    if COL_PV_REMOTE in df.columns:
        exprs.extend([
            pl.col(COL_PV_REMOTE).shift(LAG_1D).alias("pv_remote_lag_1d"),
            pl.col(COL_PV_REMOTE).shift(LAG_7D).alias("pv_remote_lag_7d"),
        ])
    return df.with_columns(exprs) if exprs else df

def add_cyclical_features(df):
    log.info("  [Groupe 4] Encodage cyclique...")
    ts = pl.col(COL_TIMESTAMP)
    return df.with_columns([
        (2*np.pi*ts.dt.hour()/24).sin().alias("hour_sin"),
        (2*np.pi*ts.dt.hour()/24).cos().alias("hour_cos"),
        (2*np.pi*ts.dt.weekday()/7).sin().alias("weekday_sin"),
        (2*np.pi*ts.dt.weekday()/7).cos().alias("weekday_cos"),
        (2*np.pi*(ts.dt.month()-1)/12).sin().alias("month_sin"),
        (2*np.pi*(ts.dt.month()-1)/12).cos().alias("month_cos"),
    ])

def add_calendar_features(df):
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

def add_meteo_lags(df):
    log.info("  [Groupe 6] Météo réelle lag J-1 (shift 96)...")
    exprs = [pl.col(c).shift(LAG_1D).alias(f"{c}_lag_1d")
             for c in METEO_REAL_COLS if c in df.columns]
    return df.with_columns(exprs) if exprs else df

def add_nwp_features(df):
    log.info("  [Groupe 7] NWP (shift -96 → heure cible J+1)...")
    available = [c for c in NWP_COLS if c in df.columns]
    if len(available) < len(NWP_COLS):
        log.warning(f"  NWP manquantes : {set(NWP_COLS) - set(available)}")
    exprs = [pl.col(c).shift(-HORIZON_DA).alias(f"{c}_target") for c in available]
    return df.with_columns(exprs) if exprs else df

def add_interactions(df):
    log.info("  [Groupe 8] Interactions (temp², temp×rad) sur NWP cible...")
    t = f"{COL_PRED_TEMP_CTRL}_target"
    r = f"{COL_PRED_GLOB_CTRL}_target"
    exprs = []
    if t in df.columns:
        exprs.append((pl.col(t)**2).alias("nwp_temp_squared"))
    if t in df.columns and r in df.columns:
        exprs.append((pl.col(t)*pl.col(r)).alias("nwp_temp_x_rad"))
    return df.with_columns(exprs) if exprs else df

def build_target(df):
    log.info("  Cible Y : load[t + 96]...")
    return df.with_columns(pl.col(COL_LOAD).shift(-HORIZON_DA).alias("target"))

def build_feature_matrix(df_oiken, df_meteo_real, df_meteo_pred):
    log.info("\n" + "═"*60 + "\nFEATURES DAY-AHEAD (v2)\n" + "═"*60)
    log.info("\nJointure des 3 sources...")
    mr = [COL_TIMESTAMP] + [c for c in df_meteo_real.columns if not c.endswith("_gap") and c != COL_TIMESTAMP]
    mp = [COL_TIMESTAMP] + [c for c in df_meteo_pred.columns if not c.endswith("_gap") and c != COL_TIMESTAMP]
    df = (df_oiken
          .join(df_meteo_real.select(mr), on=COL_TIMESTAMP, how="left")
          .join(df_meteo_pred.select(mp), on=COL_TIMESTAMP, how="left"))
    log.info(f"  Shape après jointure : {df.shape}")

    for fn in [add_load_lags, add_rolling_features, add_pv_lags,
               add_cyclical_features, add_calendar_features,
               add_meteo_lags, add_nwp_features, add_interactions, build_target]:
        df = fn(df)

    n = df.height
    df = df.filter(pl.col("target").is_not_null())
    log.info(f"\n  Lignes sans target : {n - df.height}")
    check = [c for c in df.columns if "lag" in c or "rolling" in c or c.endswith("_target")]
    n = df.height
    df = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in check]))
    log.info(f"  Lignes incomplètes : {n - df.height}")
    log.info(f"  Shape finale : {df.shape}")
    return df

def train_test_split(df):
    i = int(len(df) * TRAIN_RATIO)
    tr, te = df[:i], df[i:]
    log.info(f"\nSplit {TRAIN_RATIO:.0%} : Train {tr.height} | Test {te.height}")
    return tr, te

def get_feature_columns(df):
    exclude = {COL_TIMESTAMP, "target", COL_FORECAST_LOAD, "net_load",
               COL_PV_LOCAL, COL_PV_REMOTE, COL_PV_CENTRAL, COL_PV_SIERRE,
               COL_TEMP, COL_GLOB, COL_PRECIP, COL_HUMIDITY,
               "sunshine_min", "wind_speed_ms", COL_LOAD,
               # Vent exclu (r=-0.252, pas d'éolien) — ces colonnes arrivent
               # via la jointure meteo_pred mais ne doivent pas être features
               "pred_wind_ctrl", "pred_wind_std"}
    exclude.update(NWP_COLS)
    return sorted([c for c in df.columns if c not in exclude and not c.endswith("_gap")])

def run_feature_engineering_da():
    log.info("Chargement données nettoyées (v2)...")
    df_o = pl.read_parquet(PROCESSED_DIR / "oiken_clean_v2.parquet")
    df_r = pl.read_parquet(PROCESSED_DIR / "meteo_real_clean.parquet")
    df_p = pl.read_parquet(PROCESSED_DIR / "meteo_pred_clean.parquet")
    df = build_feature_matrix(df_o, df_r, df_p)
    tr, te = train_test_split(df)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    tr.write_parquet(FEATURES_DIR / "train_da_v2.parquet")
    te.write_parquet(FEATURES_DIR / "test_da_v2.parquet")
    log.info("✓ Sauvegardé train_da_v2 / test_da_v2")
    fc = get_feature_columns(df)
    log.info(f"\n── DA v2 : {len(fc)} features ──")
    for c in fc:
        n = df[c].null_count()
        log.info(f"  [{'✓' if n==0 else f'! {100*n/df.height:.1f}%'}] {c}")
    return tr, te

if __name__ == "__main__":
    tr, te = run_feature_engineering_da()
    fc = get_feature_columns(tr)
    print(f"\nTrain: {tr.shape} | Test: {te.shape} | Features: {len(fc)}")
    for i, c in enumerate(fc, 1):
        print(f"  {i:2d}. {c}")

