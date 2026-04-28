"""
build_golden_parquet_v2.py — Features v4 sur golden, FIDÈLE au pipeline train
──────────────────────────────────────────────────────────────────────────────
Reconstruction COMPLÈTE qui reproduit exactement features_da_v4.py original.

Différences vs build_golden_parquet.py (v1) :
  1. Tout le pipeline tourne en UTC (timestamps en Datetime[us, UTC]).
     Le golden CSV est en heure locale Europe/Zurich naïve → on convertit
     vers UTC dès le chargement.
  2. temporal_shift utilise une JOINTURE temporelle (pas un shift d'index).
     Robuste aux DST et aux trous.
  3. Scaler PV calculé avec SOLAR_HOUR_MIN=10, MAX=14 en UTC (= ~12-16h locale,
     juste après le pic solaire d'été — c'est ce que fait pv_scaler_v4.py).
  4. PAS de calibration load_G → load_T (pas nécessaire si tout est aligné).
  5. pv_local_kwh = pv_central_kwh + pv_sierre_kwh (sans Sion).

Sortie :
  03-data/data/processed/golden_features_v2.parquet

Auteur : Marius Fabbri
"""

import polars as pl
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228")
DST_ROOT      = PROJECT_ROOT / "03-data"
DST_PROCESSED = DST_ROOT / "data" / "processed"
DST_MODELS    = DST_ROOT / "models"
DST_RAW       = DST_ROOT / "data" / "raw"
TRAIN_PARQUET = PROJECT_ROOT / "01-data" / "data" / "processed" / "features_v4" / "train_da_v4.parquet"

# ── Constantes (identiques à features_da_v4.py + pv_scaler_v4.py) ─────────────
COL_TIMESTAMP     = "timestamp"
COL_LOAD          = "load"
COL_LOAD_NORM     = "load_normalized"
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

DELTA_1D = timedelta(hours=24)
DELTA_2D = timedelta(hours=48)
DELTA_7D = timedelta(days=7)

SOLAR_HOUR_MIN = 10
SOLAR_HOUR_MAX = 14
QUANTILE_LOW   = 0.05
SMOOTH_WINDOW  = 3

HOLIDAYS_FIXED  = {(1,1),(1,2),(3,19),(5,1),(8,1),(8,15),(11,1),(12,8),(12,25),(12,26)}
HOLIDAYS_MOBILE = {
    "2022-04-15","2022-04-18","2022-05-26","2022-06-06",
    "2023-04-07","2023-04-10","2023-05-18","2023-05-29",
    "2024-03-29","2024-04-01","2024-05-09","2024-05-20",
    "2025-04-18","2025-04-21","2025-05-29","2025-06-09",
    "2026-04-03","2026-04-06","2026-05-14","2026-05-25",
}


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL SHIFT — IDENTIQUE À L'ORIGINAL (par jointure)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_shift(df, col, delta, alias):
    """Récupère col à t-delta via join sur timestamp. Robuste aux DST/trous."""
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) + delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")


def temporal_shift_forward(df, col, delta, alias):
    """Récupère col à t+delta via join sur timestamp."""
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) - delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — Chargement golden CSV + conversion UTC
# ─────────────────────────────────────────────────────────────────────────────

def load_golden_csv() -> pl.DataFrame:
    log.info("── Étape 1 : Chargement golden CSV (heure locale → UTC) ──")
    p = DST_RAW / "oiken-golden-dataset.csv"
    df = pl.read_csv(p, try_parse_dates=False).rename({
        "standardised load [-]"                : "load",
        "standardised forecast load [-]"       : "forecast_load",
        "central valais solar production [kWh]": "pv_central_kwh",
        "sion area solar production [kWh]"     : "pv_sion_kwh",
        "sierre area production [kWh]"         : "pv_sierre_kwh",
        "remote solar production [kWh]"        : "pv_remote_kwh",
    })

    # Parse timestamp (heure locale Europe/Zurich naïve)
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime("us"), "%d/%m/%Y %H:%M")
    )

    # Conversion local naïf → UTC.
    # replace_time_zone("Europe/Zurich") interprète le naïf comme local,
    # convert_time_zone("UTC") le convertit ensuite vers UTC.
    # ambiguous="earliest" : pour la 1h doublée du retour à l'heure d'hiver,
    # on garde la première occurrence (convention la plus standard).
    # non_existent="null" : le trou de 1h au passage à l'heure d'été (jamais
    # rencontré en pratique car le golden n'a pas ces pas) → null.
    log.info("  Conversion timestamps : Europe/Zurich naïf → UTC")
    df = df.with_columns(
        pl.col("timestamp")
          .dt.replace_time_zone("Europe/Zurich", ambiguous="earliest", non_existent="null")
          .dt.convert_time_zone("UTC")
    )

    # Drop éventuels timestamps null après conversion (heures inexistantes DST)
    n_before = df.height
    df = df.filter(pl.col("timestamp").is_not_null())
    if df.height < n_before:
        log.info(f"  Drop {n_before - df.height} timestamps inexistants (DST spring-forward)")

    # Drop doublons éventuels (heures redoublées DST automne, on garde la 1ère)
    n_before = df.height
    df = df.unique(subset=["timestamp"], keep="first").sort("timestamp")
    if df.height < n_before:
        log.info(f"  Drop {n_before - df.height} doublons (DST fall-back)")

    # pv_local_kwh = pv_central + pv_sierre (sans Sion → match train)
    df = df.with_columns(
        (pl.col("pv_central_kwh") + pl.col("pv_sierre_kwh")).alias("pv_local_kwh")
    )

    log.info(f"  Shape : {df.shape}")
    log.info(f"  Période UTC : {df['timestamp'].min()} → {df['timestamp'].max()}")
    log.info(f"  pv_local_kwh : mean={df['pv_local_kwh'].mean():.2f} max={df['pv_local_kwh'].max():.2f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — Jointure météo en UTC
# ─────────────────────────────────────────────────────────────────────────────

def join_meteo(df: pl.DataFrame) -> pl.DataFrame:
    log.info("── Étape 2 : Jointure météo (UTC vs UTC) ──")
    df_r = pl.read_parquet(DST_PROCESSED / "meteo_real_clean.parquet")
    df_p = pl.read_parquet(DST_PROCESSED / "meteo_pred_clean.parquet")

    # Vérifier que la météo est en UTC (sinon problème de pipeline météo)
    if df_r["timestamp"].dtype.time_zone != "UTC":
        log.warning(f"  meteo_real timestamp tz = {df_r['timestamp'].dtype.time_zone}, attendu UTC")
    if df_p["timestamp"].dtype.time_zone != "UTC":
        log.warning(f"  meteo_pred timestamp tz = {df_p['timestamp'].dtype.time_zone}, attendu UTC")

    log.info(f"  Golden    : {df['timestamp'].min()} → {df['timestamp'].max()}")
    log.info(f"  Meteo real: {df_r['timestamp'].min()} → {df_r['timestamp'].max()}")
    log.info(f"  Meteo pred: {df_p['timestamp'].min()} → {df_p['timestamp'].max()}")

    real_cols = ["timestamp"] + [c for c in df_r.columns if c != "timestamp" and not c.endswith("_gap")]
    pred_cols = ["timestamp"] + [c for c in df_p.columns if c != "timestamp" and not c.endswith("_gap")]

    df = df.join(df_r.select(real_cols), on="timestamp", how="left")
    df = df.join(df_p.select(pred_cols), on="timestamp", how="left")

    for col in METEO_REAL_COLS + NWP_COLS:
        if col in df.columns:
            n = df[col].null_count()
            pct = 100 * n / df.height
            if n > 0:
                log.info(f"  {col:<40} {n:>6} nulls ({pct:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — Scaler PV (extension)
# ─────────────────────────────────────────────────────────────────────────────

def build_extended_scaler(df_golden: pl.DataFrame) -> pl.DataFrame:
    """
    Étend pv_scaler_v4.parquet (oct 2022 → sept 2025) avec les nouveaux mois
    du golden (oct 2025 → avr 2026) en UTILISANT LA FENÊTRE 10-14h UTC
    (identique à pv_scaler_v4.py original).
    """
    log.info("── Étape 3 : Extension scaler PV (10-14h UTC, identique original) ──")
    df_existing = pl.read_parquet(DST_PROCESSED / "pv_scaler_v4.parquet")
    covered = set(zip(df_existing["year"].to_list(), df_existing["month"].to_list()))
    log.info(f"  Scaler existant : {df_existing.height} mois")

    df_solar = (
        df_golden
        .with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.hour().alias("hour"),
        ])
        .filter((pl.col("hour") >= SOLAR_HOUR_MIN) & (pl.col("hour") < SOLAR_HOUR_MAX))
    )

    new_months = [
        (y, m) for y, m in
        df_solar.select(["year", "month"]).unique().sort(["year", "month"]).rows()
        if (y, m) not in covered
    ]

    if not new_months:
        log.info("  Aucun nouveau mois → scaler existant utilisé tel quel")
        return df_existing

    log.info(f"  Nouveaux mois : {new_months}")

    new_months_keys = [f"{y}-{m:02d}" for y, m in new_months]
    df_new = (
        df_solar
        .with_columns(
            (pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2))
            .alias("_ym_key")
        )
        .filter(pl.col("_ym_key").is_in(new_months_keys))
        .drop("_ym_key")
        .group_by(["year", "month"])
        .agg(
            pl.col("load").quantile(QUANTILE_LOW).alias("load_q05"),
            pl.col("load").count().alias("n_obs"),
        )
        .sort(["year", "month"])
        .with_columns(
            pl.col("load_q05").neg().clip(lower_bound=0.5).alias("raw_signal")
        )
    )

    df_existing_prep = df_existing.select([
        pl.col("year").cast(pl.Int32),
        pl.col("month").cast(pl.Int32),
        pl.col("raw_signal").cast(pl.Float64),
        pl.col("load_q05").cast(pl.Float64),
        pl.lit(0).cast(pl.UInt32).alias("n_obs"),
    ])
    df_new_prep = df_new.select([
        pl.col("year").cast(pl.Int32),
        pl.col("month").cast(pl.Int32),
        pl.col("raw_signal").cast(pl.Float64),
        pl.col("load_q05").cast(pl.Float64),
        pl.col("n_obs").cast(pl.UInt32),
    ])
    df_all = pl.concat([df_existing_prep, df_new_prep], how="vertical").sort(["year", "month"])

    df_all = df_all.with_columns(
        pl.col("raw_signal").rolling_mean(window_size=SMOOTH_WINDOW, min_samples=1).alias("raw_signal_smooth")
    )

    first_val = df_all["raw_signal_smooth"][0]
    if first_val == 0:
        first_val = 1.0

    df_all = df_all.with_columns(
        (pl.col("raw_signal_smooth") / first_val).clip(lower_bound=0.1).alias("pv_scaler")
    )

    log.info("  Scaler étendu (fin) :")
    for row in df_all.iter_rows(named=True):
        if row["year"] >= 2025:
            marker = " ← NOUVEAU" if (row["year"], row["month"]) in new_months else ""
            log.info(f"    {row['year']}-{row['month']:02d}  scaler={row['pv_scaler']:.4f}{marker}")

    out = DST_PROCESSED / "pv_scaler_v4_extended.parquet"
    df_all.select(["year", "month", "pv_scaler", "raw_signal", "load_q05"]).write_parquet(out)
    return df_all.select(["year", "month", "pv_scaler", "raw_signal", "load_q05"])


def attach_pv_scaler(df: pl.DataFrame, df_scaler: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("timestamp").dt.year().alias("_year"),
        pl.col("timestamp").dt.month().alias("_month"),
    ])
    df = df.join(
        df_scaler.select(["year", "month", "pv_scaler"])
                 .rename({"year": "_year", "month": "_month"}),
        on=["_year", "_month"], how="left",
    ).drop(["_year", "_month"])

    n_null = df["pv_scaler"].null_count()
    if n_null > 0:
        log.warning(f"  {n_null} pas sans scaler → forward/backward fill")
        df = df.with_columns(pl.col("pv_scaler").forward_fill().backward_fill())

    log.info(f"  Scaler attaché : min={df['pv_scaler'].min():.4f}  max={df['pv_scaler'].max():.4f}")
    return df


def normalize_load_column(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col(COL_LOAD).alias("load_raw"),
        (pl.col(COL_LOAD) / pl.col("pv_scaler")).alias(COL_LOAD_NORM),
    ])
    df = df.with_columns(pl.col(COL_LOAD_NORM).alias(COL_LOAD))
    log.info(f"  load_raw  : mean={df['load_raw'].mean():.4f}  std={df['load_raw'].std():.4f}")
    log.info(f"  load_norm : mean={df[COL_LOAD].mean():.4f}  std={df[COL_LOAD].std():.4f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES (IDENTIQUES À features_da_v4.py)
# ─────────────────────────────────────────────────────────────────────────────

def add_load_lags(df):
    log.info("  [1] Lags charge normalisée (J-1, J-2, J-7)")
    df = temporal_shift(df, COL_LOAD, DELTA_1D, "load_lag_1d")
    df = temporal_shift(df, COL_LOAD, DELTA_2D, "load_lag_2d")
    df = temporal_shift(df, COL_LOAD, DELTA_7D, "load_lag_7d")
    return df

def add_rolling_features(df):
    log.info("  [2] Rolling mean/std")
    shifted = pl.col(COL_LOAD).shift(96)
    return df.with_columns([
        shifted.rolling_mean(window_size=96).alias("rolling_mean_24h"),
        shifted.rolling_std(window_size=96).alias("rolling_std_24h"),
        shifted.rolling_mean(window_size=672).alias("rolling_mean_7d"),
    ])

def add_pv_lags(df):
    log.info("  [3] Lags PV")
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
    log.info("  [5] Calendaire")
    ts = pl.col(COL_TIMESTAMP)
    fixed_list = [f"{m}-{d}" for m, d in HOLIDAYS_FIXED]
    date_md   = ts.dt.month().cast(pl.Utf8) + "-" + ts.dt.day().cast(pl.Utf8)
    date_full = (ts.dt.year().cast(pl.Utf8) + "-"
                 + ts.dt.month().cast(pl.Utf8).str.zfill(2) + "-"
                 + ts.dt.day().cast(pl.Utf8).str.zfill(2))
    month = ts.dt.month()
    return df.with_columns([
        (ts.dt.weekday() >= 5).alias("is_weekend"),
        (date_md.is_in(fixed_list) | date_full.is_in(list(HOLIDAYS_MOBILE))).alias("is_holiday"),
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
    log.info("  [9] Feature pv_scaler explicite")
    df = temporal_shift_forward(df, "pv_scaler", DELTA_1D, "pv_scaler_target")
    return df

def build_target(df):
    log.info("  [T] Cibles")
    df = temporal_shift_forward(df, COL_LOAD,          DELTA_1D, "target_normalized")
    df = temporal_shift_forward(df, "load_raw",        DELTA_1D, "target_raw")
    df = temporal_shift_forward(df, COL_FORECAST_LOAD, DELTA_1D, "forecast_load_target")
    return df


def get_feature_columns(df):
    exclude = {
        COL_TIMESTAMP,
        "target_normalized", "target_raw", "forecast_load_target",
        COL_FORECAST_LOAD, "net_load", "load_raw",
        COL_LOAD_NORM, COL_LOAD,
        COL_PV_LOCAL, COL_PV_REMOTE, COL_PV_CENTRAL, COL_PV_SIERRE, "pv_sion_kwh",
        COL_TEMP, COL_GLOB, COL_PRECIP, COL_HUMIDITY,
        "sunshine_min", "wind_speed_ms",
        "pred_wind_ctrl", "pred_wind_std",
        "pred_temperature_std", "pred_radiation_std",
    }
    exclude.update(NWP_COLS)
    return sorted([c for c in df.columns if c not in exclude and not c.endswith("_gap")])


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(df_golden: pl.DataFrame):
    """Compare golden vs train sur la zone overlap (ts communs UTC)."""
    log.info("── Sanity check : overlap UTC golden vs train ──")
    df_train = pl.read_parquet(TRAIN_PARQUET)

    # join sur timestamp UTC commun (les deux sont en Datetime[us, UTC])
    common = df_train.select(["timestamp"]).join(
        df_golden.select(["timestamp"]), on="timestamp", how="inner"
    )
    log.info(f"  Timestamps communs UTC : {common.height}")

    if common.height < 100:
        log.warning("  Très peu d'overlap — vérifier la conversion timezone")
        return

    key_features = ["load_lag_1d", "pv_local_lag_1d", "pv_remote_lag_1d",
                    "rolling_mean_24h", "pv_scaler"]
    log.info(f"  {'Feature':<25} {'Train mean':>12} {'Gold mean':>12} {'delta_mean':>12}")
    for c in key_features:
        if c not in df_train.columns or c not in df_golden.columns:
            continue
        # Joindre sur les timestamps communs
        j = df_train.select(["timestamp", c]).rename({c: "T"}).join(
            df_golden.select(["timestamp", c]).rename({c: "G"}),
            on="timestamp", how="inner"
        )
        tm = j["T"].mean()
        gm = j["G"].mean()
        delta = gm - tm
        flag = " ⚠" if abs(delta) > 0.05 * abs(tm + 1e-9) else ""
        log.info(f"  {c:<25} {tm:>12.4f} {gm:>12.4f} {delta:>+12.4f}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 65)
    log.info("BUILD GOLDEN FEATURES v4 (V2 — pipeline UTC, fidèle à l'original)")
    log.info("=" * 65)

    df = load_golden_csv()
    df = join_meteo(df)

    df_scaler = build_extended_scaler(df)
    log.info("── Normalisation load ──")
    df = attach_pv_scaler(df, df_scaler)
    df = normalize_load_column(df)

    log.info("── Construction features ──")
    for fn in [add_load_lags, add_rolling_features, add_pv_lags,
               add_cyclical_features, add_calendar_features,
               add_meteo_lags, add_nwp_features, add_interactions,
               add_pv_scaler_feature, build_target]:
        df = fn(df)

    n0 = df.height
    df = df.filter(pl.col("target_normalized").is_not_null())
    df = df.filter(pl.col("target_raw").is_not_null())
    df = df.filter(pl.col("load_lag_1d").is_not_null())
    df = df.filter(pl.col("load_lag_7d").is_not_null())
    log.info(f"  Lignes après filtrage : {df.height} (supprimées : {n0 - df.height})")

    # Imputation médianes train
    feature_cols = get_feature_columns(df)
    medians_path = DST_MODELS / "medians_da.joblib"
    if not medians_path.exists():
        raise FileNotFoundError(f"medians_da.joblib introuvable")
    medians = joblib.load(medians_path)
    if not isinstance(medians, dict) or len(medians) == 0:
        raise ValueError(f"medians_da.joblib vide")
    log.info(f"  ✓ Médianes train chargées : {len(medians)} colonnes")

    for c in feature_cols:
        if df[c].null_count() > 0:
            n_nulls = df[c].null_count()
            if c in medians:
                med = medians[c]
                df = df.with_columns(pl.col(c).fill_null(float(med)))
                log.info(f"  Imputé : {c} ({n_nulls} nulls, médiane train={med:.4f})")
            else:
                med = df[c].drop_nulls().median()
                df = df.with_columns(pl.col(c).fill_null(float(med)))
                log.warning(f"  Imputé (FALLBACK) : {c} ({n_nulls} nulls, médiane golden={med:.4f})")

    remaining = {c: df[c].null_count() for c in feature_cols if df[c].null_count() > 0}
    if remaining:
        raise ValueError(f"Nulls résiduels après imputation : {remaining}")
    log.info("  ✓ Aucun null résiduel")

    log.info(f"\n  Shape finale    : {df.shape}")
    log.info(f"  Période golden  : {df['timestamp'].min()} → {df['timestamp'].max()}")
    log.info(f"  Nombre features : {len(feature_cols)}")

    sanity_check(df)

    out = DST_PROCESSED / "golden_features_v2.parquet"
    df.write_parquet(out)
    log.info(f"\n✓ Sauvegardé : {out}")
    log.info("\nProchaine étape : adapter predict_golden.py pour lire golden_features_v2.parquet")
    log.info("=" * 65)

    return df, feature_cols


if __name__ == "__main__":
    df, feature_cols = run()