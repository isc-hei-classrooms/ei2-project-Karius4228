"""
features_da_v4.py — Features Day-Ahead avec normalisation PV (v4)
──────────────────────────────────────────────────────────────────
Différences vs v3 :
  1. Chargement du scaler pv_scaler_v4.parquet (calculé par pv_scaler_v4.py)
  2. Normalisation de `load` AVANT la construction des features
       → tous les lags et rolling stats sont calculés sur load_normalized
       → le modèle n'apprend jamais de valeur absolue liée au niveau PV 2022
  3. La cible est `target_normalized` (load[t+24h] / scaler[t+24h])
  4. La colonne `pv_scaler_target` est sauvegardée dans le parquet
     pour permettre la dénormalisation dans model_da_v4.py
  5. `load_raw` conservé pour debug et comparaison

Ordre d'exécution :
  python pv_scaler_v4.py      # d'abord, calcule le scaler
  python features_da_v4.py    # ensuite, construit les features

Auteur : Marius Fabbri
"""

import polars as pl
import numpy as np
from pathlib import Path
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR    = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"
OUTPUT_DIR    = PROCESSED_DIR / "features_v4"

# ── Noms de colonnes ──────────────────────────────────────────────────────────
COL_TIMESTAMP     = "timestamp"
COL_LOAD          = "load"
COL_LOAD_NORM     = "load_normalized"   # charge après division par scaler
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

TRAIN_RATIO = 0.70
DELTA_1D    = timedelta(hours=24)
DELTA_2D    = timedelta(hours=48)
DELTA_7D    = timedelta(days=7)

HOLIDAYS_FIXED  = {(1,1),(1,2),(3,19),(5,1),(8,1),(8,15),(11,1),(12,8),(12,25),(12,26)}
HOLIDAYS_MOBILE = {
    "2022-04-15","2022-04-18","2022-05-26","2022-06-06",
    "2023-04-07","2023-04-10","2023-05-18","2023-05-29",
    "2024-03-29","2024-04-01","2024-05-09","2024-05-20",
    "2025-04-18","2025-04-21","2025-05-29","2025-06-09",
}


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES (identiques v3)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_shift(df, col, delta, alias):
    """Récupère la valeur de col à t-delta via join sur timestamp."""
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) + delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")


def temporal_shift_forward(df, col, delta, alias):
    """Récupère la valeur de col à t+delta via join sur timestamp."""
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) - delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION — ÉTAPE CLÉ v4
# ─────────────────────────────────────────────────────────────────────────────

def attach_pv_scaler(df: pl.DataFrame, df_scaler: pl.DataFrame) -> pl.DataFrame:
    """
    Joint le scaler mensuel (year, month, pv_scaler) sur la grille 15min.

    Le scaler est calculé UNE FOIS sur le dataset complet par pv_scaler_v4.py
    et appliqué ici uniformément, sans regarder le futur (le scaler du mois M
    est basé sur les données historiques du mois M → pas de leakage).
    """
    log.info("  [0a] Jointure scaler PV")
    df = df.with_columns([
        pl.col(COL_TIMESTAMP).dt.year().alias("_year"),
        pl.col(COL_TIMESTAMP).dt.month().alias("_month"),
    ])
    df = df.join(
        df_scaler.select(["year", "month", "pv_scaler"])
                 .rename({"year": "_year", "month": "_month"}),
        on=["_year", "_month"],
        how="left",
    ).drop(["_year", "_month"])

    n_null = df["pv_scaler"].null_count()
    if n_null > 0:
        log.warning(f"  {n_null} pas sans scaler → forward/backward fill")
        df = df.with_columns(
            pl.col("pv_scaler").forward_fill().backward_fill()
        )

    log.info(f"  Scaler : min={df['pv_scaler'].min():.4f} max={df['pv_scaler'].max():.4f}")
    return df


def normalize_load_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Crée load_normalized = load / pv_scaler.

    load_raw est conservé pour debug.
    Après cette étape, TOUTES les features lag/rolling sont construites
    sur load_normalized (pas sur load brut).
    """
    log.info("  [0b] Normalisation load → load_normalized")
    df = df.with_columns([
        pl.col(COL_LOAD).alias("load_raw"),  # backup
        (pl.col(COL_LOAD) / pl.col("pv_scaler")).alias(COL_LOAD_NORM),
    ])
    # On remplace la colonne 'load' par load_normalized pour la suite
    # → toutes les fonctions qui utilisent COL_LOAD travailleront sur le signal normalisé
    df = df.with_columns(pl.col(COL_LOAD_NORM).alias(COL_LOAD))
    log.info(
        f"  load_raw   : mean={df['load_raw'].mean():.4f} std={df['load_raw'].std():.4f}\n"
        f"  load_norm  : mean={df[COL_LOAD].mean():.4f} std={df[COL_LOAD].std():.4f}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES (identiques v3 — travaillent maintenant sur load normalisé)
# ─────────────────────────────────────────────────────────────────────────────

def add_load_lags(df):
    log.info("  [1] Lags de charge normalisée (J-1, J-2, J-7)")
    df = temporal_shift(df, COL_LOAD, DELTA_1D, "load_lag_1d")
    df = temporal_shift(df, COL_LOAD, DELTA_2D, "load_lag_2d")
    df = temporal_shift(df, COL_LOAD, DELTA_7D, "load_lag_7d")
    return df


def add_rolling_features(df):
    log.info("  [2] Rolling mean/std (24h, 7j) sur charge normalisée")
    shifted = pl.col(COL_LOAD).shift(96)
    return df.with_columns([
        shifted.rolling_mean(window_size=96).alias("rolling_mean_24h"),
        shifted.rolling_std(window_size=96).alias("rolling_std_24h"),
        shifted.rolling_mean(window_size=672).alias("rolling_mean_7d"),
    ])


def add_pv_lags(df):
    log.info("  [3] Lags PV local + remote (J-1, J-7)")
    if COL_PV_LOCAL in df.columns:
        df = temporal_shift(df, COL_PV_LOCAL, DELTA_1D, "pv_local_lag_1d")
        df = temporal_shift(df, COL_PV_LOCAL, DELTA_7D, "pv_local_lag_7d")
    if COL_PV_REMOTE in df.columns:
        df = temporal_shift(df, COL_PV_REMOTE, DELTA_1D, "pv_remote_lag_1d")
        df = temporal_shift(df, COL_PV_REMOTE, DELTA_7D, "pv_remote_lag_7d")
    return df


def add_cyclical_features(df):
    log.info("  [4] Encodage cyclique")
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


def add_calendar_features(df):
    log.info("  [5] Variables calendaires")
    ts = pl.col(COL_TIMESTAMP)
    is_weekend = (ts.dt.weekday() >= 5).alias("is_weekend")
    fixed_list = [f"{m}-{d}" for m, d in HOLIDAYS_FIXED]
    date_md   = ts.dt.month().cast(pl.Utf8) + "-" + ts.dt.day().cast(pl.Utf8)
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
    log.info("  [6] Météo réelle lag J-1")
    for c in METEO_REAL_COLS:
        if c in df.columns:
            df = temporal_shift(df, c, DELTA_1D, f"{c}_lag_1d")
    return df


def add_nwp_features(df):
    log.info("  [7] NWP forward-fill + shift +24h")
    available = [c for c in NWP_COLS if c in df.columns]
    df = df.with_columns([pl.col(c).forward_fill().alias(c) for c in available])
    for c in available:
        df = temporal_shift_forward(df, c, DELTA_1D, f"{c}_target")
    return df


def add_interactions(df):
    log.info("  [8] Interactions NWP")
    t = "pred_temperature_ctrl_target"
    r = "pred_radiation_ctrl_target"
    exprs = []
    if t in df.columns:
        exprs.append((pl.col(t) ** 2).alias("nwp_temp_squared"))
    if t in df.columns and r in df.columns:
        exprs.append((pl.col(t) * pl.col(r)).alias("nwp_temp_x_rad"))
    return df.with_columns(exprs) if exprs else df


def add_pv_scaler_feature(df):
    """
    [NOUVEAU v4] Ajoute pv_scaler comme feature explicite.

    En plus de la normalisation, on donne au modèle la valeur du scaler
    directement comme feature → il peut apprendre que scaler=1.8
    signifie "beaucoup de PV installé donc la charge nette peut être
    très négative même avec peu de radiation".

    C'est la feature clé qui rend le modèle conscient de la croissance PV.
    """
    log.info("  [9] Feature pv_scaler explicite (proxy capacité PV)")
    # pv_scaler est déjà dans df depuis attach_pv_scaler()
    # On ajoute aussi le scaler cible (valeur au moment de la prédiction t+24h)
    df = temporal_shift_forward(df, "pv_scaler", DELTA_1D, "pv_scaler_target")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CIBLE
# ─────────────────────────────────────────────────────────────────────────────

def build_target(df):
    """
    Construit deux cibles :
      - target_normalized : load_normalized[t+24h] — CE QUE LE MODÈLE PRÉDIT
      - target_raw        : load_raw[t+24h]        — pour dénorm et comparaison
      - forecast_load_target : prévision Oiken alignée (benchmark)
      - pv_scaler_target  : scaler à t+24h (pour dénormaliser la prédiction)
    """
    log.info("  [T] Cible normalisée + raw + scaler cible")
    # Cible normalisée : le modèle optimise là-dessus
    df = temporal_shift_forward(df, COL_LOAD, DELTA_1D, "target_normalized")
    # Cible brute : pour les métriques finales
    df = temporal_shift_forward(df, "load_raw", DELTA_1D, "target_raw")
    # Benchmark Oiken
    df = temporal_shift_forward(df, COL_FORECAST_LOAD, DELTA_1D, "forecast_load_target")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LISTE DES FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_columns(df):
    """
    Retourne la liste des features en excluant les colonnes cibles,
    les colonnes brutes, et les colonnes internes de construction.

    Nouveau en v4 : pv_scaler (valeur au moment de la prédiction) et
    pv_scaler_target sont inclus comme features.
    """
    exclude = {
        COL_TIMESTAMP,
        "target_normalized", "target_raw",
        "forecast_load_target",
        COL_FORECAST_LOAD,
        "net_load", "load_raw",
        COL_LOAD_NORM,          # la version non-laggée, pas une feature
        COL_LOAD,               # idem (déjà remplacé par normalized)
        COL_PV_LOCAL, COL_PV_REMOTE, COL_PV_CENTRAL, COL_PV_SIERRE,
        COL_TEMP, COL_GLOB, COL_PRECIP, COL_HUMIDITY,
        "sunshine_min", "wind_speed_ms",
        "pred_wind_ctrl", "pred_wind_std",
    }
    exclude.update(NWP_COLS)
    return sorted([c for c in df.columns
                   if c not in exclude and not c.endswith("_gap")])


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(df_oiken, df_meteo_real, df_meteo_pred, df_scaler):
    log.info("\n" + "=" * 60 + "\nFEATURES DAY-AHEAD (v4 — normalisation PV)\n" + "=" * 60)

    # Jointure données
    mr = [COL_TIMESTAMP] + [c for c in df_meteo_real.columns
                             if not c.endswith("_gap") and c != COL_TIMESTAMP]
    mp = [COL_TIMESTAMP] + [c for c in df_meteo_pred.columns
                             if not c.endswith("_gap") and c != COL_TIMESTAMP]
    df = (df_oiken
          .join(df_meteo_real.select(mr), on=COL_TIMESTAMP, how="left")
          .join(df_meteo_pred.select(mp), on=COL_TIMESTAMP, how="left"))
    log.info(f"  Shape après jointure : {df.shape}")

    # Vérification grille
    diffs = df[COL_TIMESTAMP].diff().dt.total_seconds().drop_nulls()
    n_regular = (diffs == 900).sum()
    log.info(f"  Grille régulière : {n_regular}/{diffs.len()} ({100*n_regular/diffs.len():.1f}%)")

    # ── ÉTAPE CLÉ v4 : normalisation AVANT construction des features ──────────
    df = attach_pv_scaler(df, df_scaler)
    df = normalize_load_column(df)
    # À partir d'ici, COL_LOAD = load_normalized
    # Tous les lags et rolling stats portent sur le signal stationnaire

    # Construction des features (identiques v3 mais sur load normalisé)
    for fn in [add_load_lags, add_rolling_features, add_pv_lags,
               add_cyclical_features, add_calendar_features,
               add_meteo_lags, add_nwp_features, add_interactions,
               add_pv_scaler_feature,  # NOUVEAU v4
               build_target]:
        df = fn(df)

    # ── Filtrage des lignes sans cible ou sans lag critique ───────────────────
    n0 = df.height
    df = df.filter(pl.col("target_normalized").is_not_null())
    log.info(f"\n  Sans target_normalized : {n0 - df.height}")
    n1 = df.height
    df = df.filter(pl.col("target_raw").is_not_null())
    log.info(f"  Sans target_raw : {n1 - df.height}")
    n2 = df.height
    df = df.filter(pl.col("load_lag_1d").is_not_null())
    log.info(f"  Sans load_lag_1d : {n2 - df.height}")
    n3 = df.height
    df = df.filter(pl.col("load_lag_7d").is_not_null())
    log.info(f"  Sans load_lag_7d : {n3 - df.height}")

    # ── Imputation médiane résiduelle ─────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    for c in feature_cols:
        nc = df[c].null_count()
        if nc > 0:
            med = df[c].drop_nulls().median()
            log.info(f"  Imputation : {c} ({nc} nulls, {100*nc/df.height:.1f}%)")
            df = df.with_columns(pl.col(c).fill_null(med).alias(c))

    log.info(f"  Shape finale : {df.shape}")
    log.info(f"  Features : {len(feature_cols)}")
    return df


def train_test_split(df):
    i = int(len(df) * TRAIN_RATIO)
    tr, te = df[:i], df[i:]
    log.info(f"\nSplit {TRAIN_RATIO:.0%} :")
    log.info(f"  Train : {tr.height} | {tr[COL_TIMESTAMP].min()} → {tr[COL_TIMESTAMP].max()}")
    log.info(f"  Test  : {te.height} | {te[COL_TIMESTAMP].min()} → {te[COL_TIMESTAMP].max()}")
    return tr, te


def run_feature_engineering_da():
    log.info("Chargement données...")
    df_o = pl.read_parquet(PROCESSED_DIR / "oiken_clean_v2.parquet")
    df_r = pl.read_parquet(PROCESSED_DIR / "meteo_real_clean.parquet")
    df_p = pl.read_parquet(PROCESSED_DIR / "meteo_pred_clean.parquet")
    df_s = pl.read_parquet(PROCESSED_DIR / "pv_scaler_v4.parquet")
    log.info(f"  Scaler : {df_s.height} mois")

    df = build_feature_matrix(df_o, df_r, df_p, df_s)
    tr, te = train_test_split(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tr.write_parquet(OUTPUT_DIR / "train_da_v4.parquet")
    te.write_parquet(OUTPUT_DIR / "test_da_v4.parquet")
    log.info(f"\n✓ Sauvegardé dans {OUTPUT_DIR}")

    fc = get_feature_columns(df)
    log.info(f"\n── DA v4 : {len(fc)} features ──")
    for c in fc:
        log.info(f"  {'✓' if df[c].null_count()==0 else '!'} {c}")
    return tr, te


if __name__ == "__main__":
    tr, te = run_feature_engineering_da()
    fc = get_feature_columns(tr)
    print(f"\nTrain: {tr.shape} | Test: {te.shape} | Features: {len(fc)}")
    for i, c in enumerate(fc, 1):
        print(f"  {i:2d}. {c}")
