"""
clean.py — Nettoyage des données brutes (Oiken, météo réelle, NWP)
───────────────────────────────────────────────────────────────────
Applique 5 étapes de nettoyage sur chaque source :
  outliers IQR → bornes physiques → interpolation courte → flags trous longs

v2 : Adapté à la séparation pv_local / pv_remote
     - pv_total et net_load supprimés
     - pv_sion déjà exclu dans load_oiken.py

Entrée  : data/processed/*_raw_v2.parquet (ou meteo_real/pred.parquet)
Sortie  : data/processed/*_clean_v2.parquet

Auteur : Marius Fabbri
"""

import polars as pl
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import (
    PROCESSED_DIR, COL_TIMESTAMP,
    COL_LOAD, COL_FORECAST_LOAD,
    COL_PV_CENTRAL, COL_PV_SIERRE, COL_PV_REMOTE,
    COL_PV_LOCAL,
    COL_TEMP, COL_GLOB, COL_PRECIP, COL_HUMIDITY, COL_SUNSHINE, COL_WIND_SPEED,
    COL_PRED_TEMP_CTRL, COL_PRED_TEMP_STD,
    COL_PRED_GLOB_CTRL, COL_PRED_GLOB_STD,
    COL_PRED_PREC_CTRL, COL_PRED_WIND_CTRL, COL_PRED_WIND_STD,
    COL_PRED_SUN_CTRL, COL_PRED_HUM_CTRL,
    MAX_INTERP_STEPS, IQR_FACTOR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Colonnes NWP corrompues → exclues
NWP_COLS_TO_DROP = [COL_PRED_TEMP_STD, COL_PRED_GLOB_STD]

NWP_COLS_VALID = [
    COL_PRED_TEMP_CTRL, COL_PRED_GLOB_CTRL, COL_PRED_PREC_CTRL,
    COL_PRED_WIND_CTRL, COL_PRED_WIND_STD, COL_PRED_SUN_CTRL, COL_PRED_HUM_CTRL,
]

# Bornes physiques absolues
PHYSICAL_BOUNDS = {
    COL_LOAD:           (None, None),
    COL_PV_CENTRAL:     (0, None),
    COL_PV_SIERRE:      (0, None),
    COL_PV_REMOTE:      (0, None),
    COL_PV_LOCAL:       (0, None),
    COL_TEMP:           (-40, 50),
    COL_GLOB:           (0, 1400),
    COL_PRECIP:         (0, None),
    COL_HUMIDITY:       (0, 100),
    COL_SUNSHINE:       (0, 15),
    COL_WIND_SPEED:     (0, 100),
    COL_PRED_TEMP_CTRL: (-40, 50),
    COL_PRED_GLOB_CTRL: (0, 1400),
    COL_PRED_PREC_CTRL: (0, None),
    COL_PRED_WIND_CTRL: (0, 100),
    COL_PRED_SUN_CTRL:  (0, 15),
    COL_PRED_HUM_CTRL:  (0, 100),
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════
# ces fonctions sont conçues pour être appliquées à chaque source séparément, avant le merge final.
def report_nulls(df: pl.DataFrame, label: str):
    log.info(f"\n[{label}] Rapport nulls :")
    for col in df.columns:
        if col == COL_TIMESTAMP:
            continue
        n = df[col].null_count()
        if n == 0:
            log.info(f"  ✓ {col:40s}: 0")
            continue
        rle = df[col].is_null().cast(pl.Int32).rle()
        max_gap = int(rle.struct.field("len").filter(rle.struct.field("value") == 1).max() or 0)
        log.warning(f"  ! {col:40s}: {n:6d} ({100*n/df.height:.1f}%) | max gap {max_gap} pas")


def remove_outliers(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    exprs = []
    for col in [c for c in columns if c in df.columns]:
        s = df[col].drop_nulls()
        if s.len() == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_out:
            log.warning(f"    ! {col:35s}: {n_out} outliers → null")
        exprs.append(
            pl.when((pl.col(col) < lo) | (pl.col(col) > hi))
              .then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
        )
    return df.with_columns(exprs) if exprs else df


def apply_physical_bounds(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        cond = pl.lit(False)
        if lo is not None:
            cond = cond | (pl.col(col) < lo)
        if hi is not None:
            cond = cond | (pl.col(col) > hi)
        exprs.append(pl.when(cond).then(pl.lit(None)).otherwise(pl.col(col)).alias(col))
    return df.with_columns(exprs) if exprs else df


def interpolate_short_gaps(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    for col in [c for c in columns if c in df.columns]:
        if df[col].null_count() == 0:
            continue
        is_null = df[col].is_null()
        rle = is_null.cast(pl.Int32).rle()
        lengths = rle.struct.field("len").to_list()
        values = rle.struct.field("value").to_list()
        gap_lens = pl.Series("gap_len", [l for l, v in zip(lengths, values) for _ in range(l)])
        short_mask = is_null & (gap_lens <= MAX_INTERP_STEPS)
        interp = df[col].interpolate(method="linear")
        df = df.with_columns(
            pl.when(short_mask).then(interp).otherwise(pl.col(col)).alias(col)
        )
    return df


def flag_long_gaps(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    flags = [pl.col(col).is_null().alias(f"{col}_gap")
             for col in columns if col in df.columns]
    return df.with_columns(flags) if flags else df


# ══════════════════════════════════════════════════════════════════════════════
# NETTOYAGE PAR SOURCE
# ══════════════════════════════════════════════════════════════════════════════

def clean_oiken(df: pl.DataFrame, output_name: str = "oiken_clean_v2.parquet") -> pl.DataFrame:
    """
    Nettoyage Oiken v2.

    Changements vs v1 :
      - pv_sion déjà exclu dans load_oiken.py
      - pv_total supprimé (pas d'agrégation de sources aux dispos différentes)
      - net_load supprimé (combinaison linéaire de la cible → leakage)
      - On nettoie pv_local (central+sierre) comme agrégat
      - On nettoie pv_remote séparément
      - Après nettoyage des composantes, on recalcule pv_local proprement
    """
    log.info("\n" + "═"*60 + "\nNETTOYAGE OIKEN (v2)\n" + "═"*60)

    # Colonnes à nettoyer (individuelles, pas l'agrégat pv_local)
    cols_individual = [COL_LOAD, COL_FORECAST_LOAD,
                       COL_PV_CENTRAL, COL_PV_SIERRE, COL_PV_REMOTE]

    report_nulls(df, "Oiken brut")
    df = remove_outliers(df, cols_individual)
    df = apply_physical_bounds(df)
    df = interpolate_short_gaps(df, cols_individual)

    # Recalculer pv_local après nettoyage des composantes
    df = df.with_columns(
        pl.sum_horizontal(COL_PV_CENTRAL, COL_PV_SIERRE).alias(COL_PV_LOCAL),
    )
    log.info("  → pv_local_kwh recalculé après nettoyage des composantes")

    # Flags sur les colonnes finales
    cols_flag = [COL_LOAD, COL_FORECAST_LOAD,
                 COL_PV_CENTRAL, COL_PV_SIERRE, COL_PV_REMOTE, COL_PV_LOCAL]
    df = flag_long_gaps(df, cols_flag)
    report_nulls(df, "Oiken nettoyé (v2)")

    path = PROCESSED_DIR / output_name
    df.write_parquet(path)
    log.info(f"✓ Sauvegardé : {path} | {df.shape}")
    return df


def clean_meteo_real(df: pl.DataFrame, output_name: str = "meteo_real_clean.parquet") -> pl.DataFrame:
    """Nettoyage météo réelle (inchangé vs v1)."""
    log.info("\n" + "═"*60 + "\nNETTOYAGE MÉTÉO RÉELLE\n" + "═"*60)

    cols = [c for c in [COL_TEMP, COL_GLOB, COL_PRECIP, COL_HUMIDITY, COL_SUNSHINE, COL_WIND_SPEED]
            if c in df.columns]

    report_nulls(df, "Météo brute")
    df = remove_outliers(df, cols)
    df = apply_physical_bounds(df)

    hour = df[COL_TIMESTAMP].dt.hour()
    is_night = (hour >= 21) | (hour <= 5)
    night_exprs = [pl.when(is_night).then(0.0).otherwise(pl.col(c)).alias(c)
                   for c in [COL_GLOB, COL_SUNSHINE] if c in df.columns]
    if night_exprs:
        df = df.with_columns(night_exprs)

    df = interpolate_short_gaps(df, cols)
    df = flag_long_gaps(df, cols)
    report_nulls(df, "Météo nettoyée")

    path = PROCESSED_DIR / output_name
    df.write_parquet(path)
    log.info(f"✓ Sauvegardé : {path} | {df.shape}")
    return df


def clean_meteo_pred(df: pl.DataFrame, output_name: str = "meteo_pred_clean.parquet") -> pl.DataFrame:
    """Nettoyage NWP (inchangé vs v1)."""
    log.info("\n" + "═"*60 + "\nNETTOYAGE PRÉVISIONS NWP\n" + "═"*60)

    df = df.drop([c for c in NWP_COLS_TO_DROP if c in df.columns])
    valid = [c for c in NWP_COLS_VALID if c in df.columns]

    report_nulls(df, "NWP brut")
    df = apply_physical_bounds(df)
    df = df.with_columns([pl.col(c).forward_fill(limit=3).alias(c) for c in valid])
    df = flag_long_gaps(df, valid)
    report_nulls(df, "NWP nettoyé")

    path = PROCESSED_DIR / output_name
    df.write_parquet(path)
    log.info(f"✓ Sauvegardé : {path} | {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_all_cleaning() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Nettoie les 3 sources depuis les Parquets bruts."""
    sources = {
        "oiken":      ("oiken_raw_v2.parquet", clean_oiken),
        "meteo_real": ("meteo_real.parquet",    clean_meteo_real),
        "meteo_pred": ("meteo_pred.parquet",    clean_meteo_pred),
    }
    results = {}
    for key, (filename, clean_fn) in sources.items():
        path = PROCESSED_DIR / filename
        if path.exists():
            results[key] = clean_fn(pl.read_parquet(path))
        else:
            log.error(f"Manquant : {path}")
    return results.get("oiken"), results.get("meteo_real"), results.get("meteo_pred")


if __name__ == "__main__":
    df_oiken, df_real, df_pred = run_all_cleaning()
    for name, df in [("Oiken", df_oiken), ("Météo réelle", df_real), ("NWP", df_pred)]:
        if df is not None:
            print(f"\n{name} : {df.shape}")
            print(df.head(3))
