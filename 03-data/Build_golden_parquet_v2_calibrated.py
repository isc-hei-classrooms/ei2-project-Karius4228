"""
build_golden_parquet_v2_calibrated.py — Patch calibration forecast_load
─────────────────────────────────────────────────────────────────────────
SEULE DIFFÉRENCE vs build_golden_parquet_v2.py :
  - Charge golden_calibration_v2.joblib (fitté en UTC)
  - Applique la transformation inverse AVANT build_target() :
        load_raw_cal     = inv_a_load * load_raw + inv_b_load
        forecast_load_cal = inv_a_fcst * forecast_load + inv_b_fcst
  - Ces colonnes calibrées sont utilisées dans build_target() pour
    produire target_raw et forecast_load_target dans l'espace train.

Résultat attendu :
  - y_true_raw (= target_raw) dans l'espace z-score train (mean≈0, std≈0.93)
  - y_oiken (= forecast_load_target) dans le même espace
  - Métriques Oiken comparables à celles du train_da_v4

Sortie :
  03-data/data/processed/golden_features_v3.parquet  ← NOM DISTINCT

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

# ── Constantes identiques à build_golden_parquet_v2.py ───────────────────────
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
# Helpers identiques à v2
# ─────────────────────────────────────────────────────────────────────────────

def temporal_shift(df, col, delta, alias):
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) + delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")

def temporal_shift_forward(df, col, delta, alias):
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) - delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")


# ─────────────────────────────────────────────────────────────────────────────
# NOUVEAU : Application de la calibration
# ─────────────────────────────────────────────────────────────────────────────

def apply_calibration(df: pl.DataFrame, cal: dict) -> pl.DataFrame:
    """
    Transforme load_raw et forecast_load de l'espace golden vers l'espace train.

    Transformation inverse : y_T = inv_a * y_G + inv_b

    Les colonnes calibrées écrasent les originales UNIQUEMENT pour load_raw
    et forecast_load (les features ML utilisent load_normalized = load/pv_scaler
    et ne sont PAS affectées, car load_normalized est calculé avant cette étape).

    Appeler APRÈS normalize_load_column() (qui produit load_raw),
    AVANT build_target() (qui consomme load_raw et forecast_load).
    """
    log.info("── Calibration golden → espace train ──")

    c_load = cal["load"]
    c_fcst = cal["forecast"]

    log.info(f"  load_raw  : inv_a={c_load['inv_a']:.6f}  inv_b={c_load['inv_b']:+.6f}  (résidu_std_fit={c_load['residual_std']:.6f})")
    log.info(f"  fcst_load : inv_a={c_fcst['inv_a']:.6f}  inv_b={c_fcst['inv_b']:+.6f}  (résidu_std_fit={c_fcst['residual_std']:.6f})")

    # Stats avant
    log.info(f"  Avant : load_raw     mean={df['load_raw'].mean():.4f}  std={df['load_raw'].std():.4f}")
    log.info(f"  Avant : forecast_load mean={df[COL_FORECAST_LOAD].mean():.4f}  std={df[COL_FORECAST_LOAD].std():.4f}")

    df = df.with_columns([
        (pl.col("load_raw") * c_load["inv_a"] + c_load["inv_b"]).alias("load_raw"),
        (pl.col(COL_FORECAST_LOAD) * c_fcst["inv_a"] + c_fcst["inv_b"]).alias(COL_FORECAST_LOAD),
    ])

    # Stats après
    log.info(f"  Après : load_raw     mean={df['load_raw'].mean():.4f}  std={df['load_raw'].std():.4f}")
    log.info(f"  Après : forecast_load mean={df[COL_FORECAST_LOAD].mean():.4f}  std={df[COL_FORECAST_LOAD].std():.4f}")
    log.info(f"  (Référence train) : load_raw mean≈0.0427  std≈0.9279 | fcst mean≈0.0527  std≈0.9302)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Toutes les fonctions identiques à build_golden_parquet_v2.py
# (copiées intégralement pour que ce script soit autonome)
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
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime("us"), "%d/%m/%Y %H:%M")
    )
    log.info("  Conversion timestamps : Europe/Zurich naïf → UTC")
    df = df.with_columns(
        pl.col("timestamp")
          .dt.replace_time_zone("Europe/Zurich", ambiguous="earliest", non_existent="null")
          .dt.convert_time_zone("UTC")
    )
    n_before = df.height
    df = df.filter(pl.col("timestamp").is_not_null())
    if df.height < n_before:
        log.info(f"  Drop {n_before - df.height} timestamps inexistants (DST spring-forward)")
    n_before = df.height
    df = df.unique(subset=["timestamp"], keep="first").sort("timestamp")
    if df.height < n_before:
        log.info(f"  Drop {n_before - df.height} doublons (DST fall-back)")
    df = df.with_columns(
        (pl.col("pv_central_kwh") + pl.col("pv_sierre_kwh")).alias("pv_local_kwh")
    )
    log.info(f"  Shape : {df.shape}")
    log.info(f"  Période UTC : {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def join_meteo(df: pl.DataFrame) -> pl.DataFrame:
    log.info("── Étape 2 : Jointure météo (UTC vs UTC) ──")
    df_r = pl.read_parquet(DST_PROCESSED / "meteo_real_clean.parquet")
    df_p = pl.read_parquet(DST_PROCESSED / "meteo_pred_clean.parquet")
    real_cols = ["timestamp"] + [c for c in df_r.columns if c != "timestamp" and not c.endswith("_gap")]
    pred_cols = ["timestamp"] + [c for c in df_p.columns if c != "timestamp" and not c.endswith("_gap")]
    df = df.join(df_r.select(real_cols), on="timestamp", how="left")
    df = df.join(df_p.select(pred_cols), on="timestamp", how="left")
    return df


def build_extended_scaler(df_golden: pl.DataFrame) -> pl.DataFrame:
    log.info("── Étape 3 : Extension scaler PV ──")
    df_existing = pl.read_parquet(DST_PROCESSED / "pv_scaler_v4.parquet")
    covered = set(zip(df_existing["year"].to_list(), df_existing["month"].to_list()))
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
        .with_columns(pl.col("load_q05").neg().clip(lower_bound=0.5).alias("raw_signal"))
    )
    df_existing_prep = df_existing.select([
        pl.col("year").cast(pl.Int32), pl.col("month").cast(pl.Int32),
        pl.col("raw_signal").cast(pl.Float64), pl.col("load_q05").cast(pl.Float64),
        pl.lit(0).cast(pl.UInt32).alias("n_obs"),
    ])
    df_new_prep = df_new.select([
        pl.col("year").cast(pl.Int32), pl.col("month").cast(pl.Int32),
        pl.col("raw_signal").cast(pl.Float64), pl.col("load_q05").cast(pl.Float64),
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
    out = DST_PROCESSED / "pv_scaler_v4_extended.parquet"
    df_all.select(["year", "month", "pv_scaler", "raw_signal", "load_q05"]).write_parquet(out)
    return df_all.select(["year", "month", "pv_scaler", "raw_signal", "load_q05"])


def attach_pv_scaler(df, df_scaler):
    df = df.with_columns([
        pl.col("timestamp").dt.year().alias("_year"),
        pl.col("timestamp").dt.month().alias("_month"),
    ])
    df = df.join(
        df_scaler.select(["year", "month", "pv_scaler"])
                 .rename({"year": "_year", "month": "_month"}),
        on=["_year", "_month"], how="left",
    ).drop(["_year", "_month"])
    if df["pv_scaler"].null_count() > 0:
        df = df.with_columns(pl.col("pv_scaler").forward_fill().backward_fill())
    return df


def normalize_load_column(df):
    df = df.with_columns([
        pl.col(COL_LOAD).alias("load_raw"),
        (pl.col(COL_LOAD) / pl.col("pv_scaler")).alias(COL_LOAD_NORM),
    ])
    df = df.with_columns(pl.col(COL_LOAD_NORM).alias(COL_LOAD))
    return df


def add_load_lags(df):
    log.info("  [1] Lags charge normalisée")
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
    """
    NOTA : load_raw et forecast_load ont été calibrés AVANT cet appel.
    target_raw et forecast_load_target sont donc dans l'espace train.
    """
    log.info("  [T] Cibles (calibrées)")
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


def sanity_check(df_golden: pl.DataFrame):
    log.info("── Sanity check : overlap UTC golden vs train ──")
    df_train = pl.read_parquet(TRAIN_PARQUET)
    common = df_train.select(["timestamp"]).join(
        df_golden.select(["timestamp"]), on="timestamp", how="inner"
    )
    log.info(f"  Timestamps communs UTC : {common.height}")
    if common.height < 100:
        log.warning("  Très peu d'overlap")
        return
    # Vérification critique : target_raw doit être proche entre train et golden calibré
    for col in ["target_raw", "forecast_load_target"]:
        if col not in df_train.columns or col not in df_golden.columns:
            continue
        j = df_train.select(["timestamp", col]).rename({col: "T"}).join(
            df_golden.select(["timestamp", col]).rename({col: "G"}),
            on="timestamp", how="inner"
        )
        delta = (j["G"] - j["T"])
        log.info(f"  {col:30s} Δ mean={delta.mean():+.4f}  std={delta.std():.4f}  max|.|={delta.abs().max():.4f}")
        if delta.std() > 0.05:
            log.warning(f"  ⚠ std Δ > 0.05 pour {col} — vérifier la calibration")
        else:
            log.info(f"  ✓ {col} bien aligné avec l'espace train")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 65)
    log.info("BUILD GOLDEN FEATURES v3 (calibré, espace train)")
    log.info("=" * 65)

    # 1. Charger la calibration UTC
    cal_path = DST_PROCESSED / "golden_calibration_v2.joblib"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"golden_calibration_v2.joblib introuvable.\n"
            f"Lancer d'abord : python calibrate_golden_v2.py"
        )
    cal = joblib.load(cal_path)
    log.info(f"\n✓ Calibration chargée : overlap UTC {cal['overlap_utc']}")
    log.info(f"  load  : inv_a={cal['load']['inv_a']:.6f}  inv_b={cal['load']['inv_b']:+.6f}")
    log.info(f"  fcst  : inv_a={cal['forecast']['inv_a']:.6f}  inv_b={cal['forecast']['inv_b']:+.6f}")

    # 2. Pipeline identique à v2 jusqu'à normalize_load_column
    df = load_golden_csv()
    df = join_meteo(df)
    df_scaler = build_extended_scaler(df)
    log.info("── Normalisation load ──")
    df = attach_pv_scaler(df, df_scaler)
    df = normalize_load_column(df)
    # → À ce stade : load_raw = load brut golden (z-score golden)
    #                forecast_load = forecast brut golden (z-score golden)
    #                load (COL_LOAD) = load_raw / pv_scaler (normalisé PV) — utilisé pour les features ML

    # 3. CALIBRATION : ramener load_raw et forecast_load dans l'espace train
    #    AVANT build_target() et AVANT les features (qui utilisent COL_LOAD normalisé, pas load_raw)
    df = apply_calibration(df, cal)
    # → load_raw et forecast_load sont maintenant dans l'espace z-score train
    # → COL_LOAD (= load_normalized) N'EST PAS TOUCHÉ → les features ML restent identiques à v2

    # 4. Features + cibles
    log.info("── Construction features ──")
    for fn in [add_load_lags, add_rolling_features, add_pv_lags,
               add_cyclical_features, add_calendar_features,
               add_meteo_lags, add_nwp_features, add_interactions,
               add_pv_scaler_feature, build_target]:
        df = fn(df)

    # 5. Filtrage nulls
    n0 = df.height
    df = df.filter(pl.col("target_normalized").is_not_null())
    df = df.filter(pl.col("target_raw").is_not_null())
    df = df.filter(pl.col("load_lag_1d").is_not_null())
    df = df.filter(pl.col("load_lag_7d").is_not_null())
    log.info(f"  Lignes après filtrage : {df.height} (supprimées : {n0 - df.height})")

    # 6. Imputation médianes train
    feature_cols = get_feature_columns(df)
    medians_path = DST_MODELS / "medians_da.joblib"
    if not medians_path.exists():
        raise FileNotFoundError("medians_da.joblib introuvable")
    medians = joblib.load(medians_path)
    if not isinstance(medians, dict) or len(medians) == 0:
        raise ValueError("medians_da.joblib vide")
    log.info(f"  ✓ Médianes train chargées : {len(medians)} colonnes")
    for c in feature_cols:
        if df[c].null_count() > 0:
            n_nulls = df[c].null_count()
            if c in medians:
                df = df.with_columns(pl.col(c).fill_null(float(medians[c])))
                log.info(f"  Imputé : {c} ({n_nulls} nulls)")
            else:
                med = df[c].drop_nulls().median()
                df = df.with_columns(pl.col(c).fill_null(float(med)))
                log.warning(f"  Imputé (FALLBACK) : {c} ({n_nulls} nulls)")

    remaining = {c: df[c].null_count() for c in feature_cols if df[c].null_count() > 0}
    if remaining:
        raise ValueError(f"Nulls résiduels : {remaining}")

    log.info(f"\n  Shape finale   : {df.shape}")
    log.info(f"  Période golden : {df['timestamp'].min()} → {df['timestamp'].max()}")
    log.info(f"  Nb features    : {len(feature_cols)}")

    # 7. Sanity check vs train
    sanity_check(df)

    # 8. Sauvegarde → v3 pour distinguer du v2 non calibré
    out = DST_PROCESSED / "golden_features_v3.parquet"
    df.write_parquet(out)
    log.info(f"\n✓ Sauvegardé : {out}")
    log.info("\nProchaine étape : adapter predict_golden.py pour lire golden_features_v3.parquet")
    log.info("=" * 65)

    return df, feature_cols


if __name__ == "__main__":
    df, feature_cols = run()