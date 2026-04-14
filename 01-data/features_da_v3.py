"""
features_da_v3.py — Features Day-Ahead CORRIGÉES
─────────────────────────────────────────────────
Corrections vs v2 :
  1. Forward-fill des NWP AVANT les shifts (élimine 99% des nulls)
  2. Shifts TEMPORELS via join sur timestamp décalé (pas shift par ligne)
     → Grille 15min complète préservée, lags = vrais J-1 / J-2 / J-7
  3. Benchmark forecast_load correctement aligné sur la cible
  4. forecast_load_target ajouté pour comparaison équitable
  5. Filtrage minimal des nulls (imputation médiane pour résidus)

Contraintes temporelles (gate closure = 11h J) :
  - Charge Oiken  : livrée à 2h J → lags ≥ J-1 (24h)
  - PV local/remote : livrés à 2h J → lags ≥ J-1
  - Mesures météo : délai 1h → lag J-1 pour DA
  - NWP           : pointées sur heure cible J+1 (shift -24h)

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

TRAIN_RATIO = 0.70
DELTA_1D = timedelta(hours=24)
DELTA_2D = timedelta(hours=48)
DELTA_7D = timedelta(days=7)

HOLIDAYS_FIXED = {(1,1),(1,2),(3,19),(5,1),(8,1),(8,15),(11,1),(12,8),(12,25),(12,26)}
HOLIDAYS_MOBILE = {
    "2022-04-15","2022-04-18","2022-05-26","2022-06-06",
    "2023-04-07","2023-04-10","2023-05-18","2023-05-29",
    "2024-03-29","2024-04-01","2024-05-09","2024-05-20",
    "2025-04-18","2025-04-21","2025-05-29","2025-06-09",
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES — SHIFT TEMPOREL (le fix principal)
# ══════════════════════════════════════════════════════════════════════════════

def temporal_shift(df: pl.DataFrame, col: str, delta: timedelta, alias: str) -> pl.DataFrame:
    """
    Shift temporel correct : joint la valeur de `col` au timestamp (t - delta).

    Contrairement à pl.col().shift(N) qui décale de N LIGNES,
    cette fonction décale de `delta` en TEMPS RÉEL.
    Si le timestamp (t - delta) n'existe pas, le résultat est null.
    """
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) + delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")


def temporal_shift_forward(df: pl.DataFrame, col: str, delta: timedelta, alias: str) -> pl.DataFrame:
    """
    Shift FORWARD : récupère la valeur de `col` au timestamp (t + delta).
    Utilisé pour : target = load[t + 24h], NWP_target = NWP[t + 24h].
    """
    lookup = df.select([
        (pl.col(COL_TIMESTAMP) - delta).alias(COL_TIMESTAMP),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on=COL_TIMESTAMP, how="left")


# ══════════════════════════════════════════════════════════════════════════════
# GROUPES DE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_load_lags(df: pl.DataFrame) -> pl.DataFrame:
    """Lags de charge J-1, J-2, J-7 (shifts temporels corrects)."""
    log.info("  [Groupe 1] Lags de charge (J-1, J-2, J-7) — shift temporel...")
    df = temporal_shift(df, COL_LOAD, DELTA_1D, "load_lag_1d")
    df = temporal_shift(df, COL_LOAD, DELTA_2D, "load_lag_2d")
    df = temporal_shift(df, COL_LOAD, DELTA_7D, "load_lag_7d")
    return df


def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rolling mean/std sur 24h et 7j, calculés sur la charge DISPONIBLE (J-1+).

    On utilise d'abord le lag J-1 (déjà calculé), puis on applique le rolling
    sur la série complète (pas de trous → rolling correct).
    Méthode : shift temporel de 24h, puis rolling sur les données avant ce point.
    """
    log.info("  [Groupe 2] Rolling mean/std (24h, 7j) — sur load_lag_1d...")
    # On prend load décalé de 24h et on calcule le rolling directement
    # Sur grille régulière, shift(96) est fiable maintenant
    shifted = pl.col(COL_LOAD).shift(96)
    return df.with_columns([
        shifted.rolling_mean(window_size=96).alias("rolling_mean_24h"),
        shifted.rolling_std(window_size=96).alias("rolling_std_24h"),
        shifted.rolling_mean(window_size=672).alias("rolling_mean_7d"),
    ])


def add_pv_lags(df: pl.DataFrame) -> pl.DataFrame:
    """Lags PV local et remote, J-1 et J-7 (shifts temporels)."""
    log.info("  [Groupe 3] Lags PV local (J-1, J-7) + PV remote (J-1, J-7)...")
    if COL_PV_LOCAL in df.columns:
        df = temporal_shift(df, COL_PV_LOCAL, DELTA_1D, "pv_local_lag_1d")
        df = temporal_shift(df, COL_PV_LOCAL, DELTA_7D, "pv_local_lag_7d")
    if COL_PV_REMOTE in df.columns:
        df = temporal_shift(df, COL_PV_REMOTE, DELTA_1D, "pv_remote_lag_1d")
        df = temporal_shift(df, COL_PV_REMOTE, DELTA_7D, "pv_remote_lag_7d")
    return df


def add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
    log.info("  [Groupe 4] Encodage cyclique...")
    ts = pl.col(COL_TIMESTAMP)
    # Encode l'heure avec les minutes (6h30 ≠ 6h00)
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


def add_meteo_lags(df: pl.DataFrame) -> pl.DataFrame:
    """Météo réelle lag J-1 (shift temporel 24h)."""
    log.info("  [Groupe 6] Météo réelle lag J-1 (shift temporel 24h)...")
    for c in METEO_REAL_COLS:
        if c in df.columns:
            df = temporal_shift(df, c, DELTA_1D, f"{c}_lag_1d")
    return df


def add_nwp_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    NWP pointées sur l'heure cible (t + 24h).
    Forward-fill AVANT le shift pour combler les trous NWP.
    """
    log.info("  [Groupe 7] NWP → forward-fill + shift temporel +24h...")
    available = [c for c in NWP_COLS if c in df.columns]

    # Forward-fill des NWP (prévisions constantes entre publications)
    df = df.with_columns([pl.col(c).forward_fill().alias(c) for c in available])

    # Shift forward de 24h : récupère la NWP pour l'heure cible
    for c in available:
        df = temporal_shift_forward(df, c, DELTA_1D, f"{c}_target")

    return df


def add_interactions(df: pl.DataFrame) -> pl.DataFrame:
    log.info("  [Groupe 8] Interactions (temp², temp×rad) sur NWP cible...")
    t = "pred_temperature_ctrl_target"
    r = "pred_radiation_ctrl_target"
    exprs = []
    if t in df.columns:
        exprs.append((pl.col(t) ** 2).alias("nwp_temp_squared"))
    if t in df.columns and r in df.columns:
        exprs.append((pl.col(t) * pl.col(r)).alias("nwp_temp_x_rad"))
    return df.with_columns(exprs) if exprs else df


def build_target(df: pl.DataFrame) -> pl.DataFrame:
    """Target = load à t+24h. Benchmark = forecast_load à t+24h."""
    log.info("  Cible Y : load[t + 24h] | Benchmark : forecast_load[t + 24h]...")
    df = temporal_shift_forward(df, COL_LOAD, DELTA_1D, "target")
    # Benchmark Oiken correctement aligné : prévision pour le même timestamp que la cible
    df = temporal_shift_forward(df, COL_FORECAST_LOAD, DELTA_1D, "forecast_load_target")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLAGE
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df_oiken: pl.DataFrame, df_meteo_real: pl.DataFrame,
                         df_meteo_pred: pl.DataFrame) -> pl.DataFrame:
    log.info("\n" + "═" * 60 + "\nFEATURES DAY-AHEAD (v3 — shifts temporels)\n" + "═" * 60)

    # Jointure
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
    assert n_regular / diffs.len() > 0.99, "ERREUR: grille non régulière!"

    # Construction features
    for fn in [add_load_lags, add_rolling_features, add_pv_lags,
               add_cyclical_features, add_calendar_features,
               add_meteo_lags, add_nwp_features, add_interactions, build_target]:
        df = fn(df)

    # ── Filtrage minimal ──
    n0 = df.height
    # 1. Target obligatoire
    df = df.filter(pl.col("target").is_not_null())
    log.info(f"\n  Lignes sans target : {n0 - df.height}")

    # 2. Lag J-1 de charge obligatoire (feature principale)
    n1 = df.height
    df = df.filter(pl.col("load_lag_1d").is_not_null())
    log.info(f"  Lignes sans load_lag_1d : {n1 - df.height}")

    # 3. Lag J-7 de charge obligatoire
    n2 = df.height
    df = df.filter(pl.col("load_lag_7d").is_not_null())
    log.info(f"  Lignes sans load_lag_7d : {n2 - df.height}")

    # 4. Imputation médiane pour les features restantes avec quelques nulls
    feature_cols = get_feature_columns(df)
    for c in feature_cols:
        nc = df[c].null_count()
        if nc > 0:
            med = df[c].drop_nulls().median()
            pct = 100 * nc / df.height
            log.info(f"  Imputation médiane: {c} ({nc} nulls, {pct:.1f}%) → {med:.4f}")
            df = df.with_columns(pl.col(c).fill_null(med).alias(c))

    log.info(f"  Shape finale : {df.shape}")

    # Vérification anti-leakage des shifts
    _verify_temporal_shifts(df)

    return df


def _verify_temporal_shifts(df: pl.DataFrame) -> None:
    """Vérifie que les shifts temporels sont corrects."""
    log.info("\n── Vérification anti-leakage (shifts temporels) ──")

    # load_lag_1d doit correspondre à load 24h avant
    sample = df.filter(
        pl.col("load_lag_1d").is_not_null()
    ).select([COL_TIMESTAMP, COL_LOAD, "load_lag_1d"]).head(5)

    for row in sample.iter_rows(named=True):
        ts = row[COL_TIMESTAMP]
        lag_val = row["load_lag_1d"]
        # Chercher la valeur de load 24h avant
        ts_24h_ago = ts - DELTA_1D
        match = df.filter(pl.col(COL_TIMESTAMP) == ts_24h_ago).select(COL_LOAD)
        if match.height > 0:
            expected = match[COL_LOAD][0]
            ok = abs(lag_val - expected) < 1e-10
            log.info(f"  ts={ts}, load_lag_1d={lag_val:.6f}, load[t-24h]={expected:.6f} → {'✓' if ok else '✗ ERREUR!'}")
        else:
            log.info(f"  ts={ts}, load_lag_1d={lag_val:.6f}, load[t-24h] non trouvé (début de série)")

    # Target doit être load 24h après
    sample2 = df.filter(
        pl.col("target").is_not_null()
    ).select([COL_TIMESTAMP, COL_LOAD, "target"]).head(3)

    for row in sample2.iter_rows(named=True):
        ts = row[COL_TIMESTAMP]
        target_val = row["target"]
        ts_24h_later = ts + DELTA_1D
        match = df.filter(pl.col(COL_TIMESTAMP) == ts_24h_later).select(COL_LOAD)
        if match.height > 0:
            expected = match[COL_LOAD][0]
            ok = abs(target_val - expected) < 1e-10
            log.info(f"  ts={ts}, target={target_val:.6f}, load[t+24h]={expected:.6f} → {'✓' if ok else '✗ ERREUR!'}")

    log.info("  ✓ Vérification terminée")


# ══════════════════════════════════════════════════════════════════════════════
# SÉLECTION FEATURES / SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """
    Retourne les colonnes features (X) en excluant :
    - timestamp, target, benchmark forecast
    - colonnes brutes (load, PV, météo réelle, NWP brutes)
    - flags _gap
    - vent (exclu : pas d'éolien dans la zone)
    """
    exclude = {
        COL_TIMESTAMP, "target", "forecast_load_target",
        COL_FORECAST_LOAD, "net_load",
        COL_PV_LOCAL, COL_PV_REMOTE, COL_PV_CENTRAL, COL_PV_SIERRE,
        COL_TEMP, COL_GLOB, COL_PRECIP, COL_HUMIDITY,
        "sunshine_min", "wind_speed_ms", COL_LOAD,
        "pred_wind_ctrl", "pred_wind_std",
    }
    exclude.update(NWP_COLS)
    return sorted([c for c in df.columns
                   if c not in exclude and not c.endswith("_gap")])


def train_test_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    i = int(len(df) * TRAIN_RATIO)
    tr, te = df[:i], df[i:]
    log.info(f"\nSplit {TRAIN_RATIO:.0%} chronologique:")
    log.info(f"  Train: {tr.height} lignes | {tr[COL_TIMESTAMP].min()} → {tr[COL_TIMESTAMP].max()}")
    log.info(f"  Test:  {te.height} lignes | {te[COL_TIMESTAMP].min()} → {te[COL_TIMESTAMP].max()}")
    return tr, te


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_engineering_da(data_dir: Path = None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data" / "processed"

    log.info("Chargement données nettoyées...")
    df_o = pl.read_parquet(data_dir / "oiken_clean_v2.parquet")
    df_r = pl.read_parquet(data_dir / "meteo_real_clean.parquet")
    df_p = pl.read_parquet(data_dir / "meteo_pred_clean.parquet")

    df = build_feature_matrix(df_o, df_r, df_p)
    tr, te = train_test_split(df)

    out_dir = data_dir / "features_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    tr.write_parquet(out_dir / "train_da_v3.parquet")
    te.write_parquet(out_dir / "test_da_v3.parquet")
    log.info(f"✓ Sauvegardé dans {out_dir}")

    fc = get_feature_columns(df)
    log.info(f"\n── DA v3 : {len(fc)} features ──")
    for c in fc:
        n = df[c].null_count()
        log.info(f"  [{'✓' if n == 0 else f'! {100*n/df.height:.1f}%'}] {c}")

    return tr, te


if __name__ == "__main__":
    tr, te = run_feature_engineering_da()
    fc = get_feature_columns(tr)
    print(f"\nTrain: {tr.shape} | Test: {te.shape} | Features: {len(fc)}")
    for i, c in enumerate(fc, 1):
        print(f"  {i:2d}. {c}")
