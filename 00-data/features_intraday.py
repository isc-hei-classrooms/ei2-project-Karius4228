"""
src/processing/features_intraday.py
────────────────────────────────────
Construction de la matrice X et du vecteur Y pour le modèle Intraday.

v2 : Séparation PV local vs remote + correction des disponibilités

Contraintes temporelles (prédiction à l'instant t) :
  - Charge Oiken   : livrée le lendemain à 2h → lags ≥ 96 seulement
  - PV local       : dispo ~15 min après → LAGS COURTS DISPONIBLES (t-1, t-4)
  - PV remote      : livré le lendemain à 2h → lags ≥ 96 seulement
  - Mesures météo  : délai 1h → shift(4) = t-4
  - NWP            : shift(-12) = horizon t+12

Changement majeur vs v1 :
  Le PV local (central+sierre) est le SEUL signal temps réel disponible
  pour l'Intraday (avec la météo à t-4). La charge reste inaccessible
  en temps réel. C'est ce qui distingue vraiment l'Intraday du DA.

Sortie  : data/features/train_intraday_v2.parquet, test_intraday_v2.parquet
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
    PV_LOCAL_DELAY, METEO_DELAY,
    TRAIN_RATIO,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HORIZON_MAX = 12  # 12 pas = 3h
LAG_1D = 96
LAG_7D = 672

# Lags charge : UNIQUEMENT longs (charge livrée le lendemain à 2h)
LOAD_LAGS = [LAG_1D, LAG_7D]

# Lags PV local : courts DISPONIBLES (dispo ~15 min après)
PV_LOCAL_LAGS_SHORT = [1, 4]          # t-1 (15min), t-4 (1h)
PV_LOCAL_LAGS_LONG = [LAG_1D, LAG_7D]  # J-1, J-7

# Lags PV remote : UNIQUEMENT longs (livré le lendemain à 2h)
PV_REMOTE_LAGS = [LAG_1D, LAG_7D]

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


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 1 — LAGS DE CHARGE (longs uniquement)
# ══════════════════════════════════════════════════════════════════════════════

def add_load_lags(df):
    """
    Charge livrée le lendemain à 2h → AUCUN lag court disponible.
    Seuls J-1 (96) et J-7 (672).
    """
    log.info("  [Groupe 1] Lags charge (J-1, J-7 — pas de lags courts)...")
    return df.with_columns([
        pl.col(COL_LOAD).shift(lag).alias(f"load_lag_{lag}")
        for lag in LOAD_LAGS
    ])


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 2 — ROLLING STATISTIQUES (sur données J-1)
# ══════════════════════════════════════════════════════════════════════════════

def add_rolling_features(df):
    """Rolling calculés sur données J-1 (pas de charge temps réel)."""
    log.info("  [Groupe 2] Rolling mean/std (24h, 7j) — sur données J-1...")
    return df.with_columns([
        pl.col(COL_LOAD).shift(LAG_1D).rolling_mean(window_size=LAG_1D).alias("rolling_mean_24h"),
        pl.col(COL_LOAD).shift(LAG_1D).rolling_std(window_size=LAG_1D).alias("rolling_std_24h"),
        pl.col(COL_LOAD).shift(LAG_1D).rolling_mean(window_size=LAG_7D).alias("rolling_mean_7d"),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 3 — LAGS PV (LOCAL courts + longs, REMOTE longs uniquement)
# ══════════════════════════════════════════════════════════════════════════════

def add_pv_lags(df):
    """
    PV local (central+sierre) : dispo ~15 min après → lags courts OK !
      - t-1 (15 min)  : production PV il y a 15 min — TRÈS prédictif
      - t-4 (1h)      : tendance PV récente
      - t-96 (J-1)    : référence journalière
      - t-672 (J-7)   : même jour semaine dernière

    PV remote : livré le lendemain à 2h → lags longs uniquement
      - t-96 (J-1)
      - t-672 (J-7)

    C'est LE signal temps réel qui distingue l'Intraday du Day-Ahead.
    """
    log.info("  [Groupe 3] PV local (t-1, t-4, J-1, J-7) + PV remote (J-1, J-7)...")
    exprs = []

    # PV local — lags courts + longs
    if COL_PV_LOCAL in df.columns:
        for lag in PV_LOCAL_LAGS_SHORT:
            exprs.append(pl.col(COL_PV_LOCAL).shift(lag).alias(f"pv_local_lag_{lag}"))
        for lag in PV_LOCAL_LAGS_LONG:
            exprs.append(pl.col(COL_PV_LOCAL).shift(lag).alias(f"pv_local_lag_{lag}"))

        # Rolling PV local sur dernière heure (4 pas) — disponible en temps réel
        exprs.append(
            pl.col(COL_PV_LOCAL).shift(PV_LOCAL_DELAY)
            .rolling_mean(window_size=4).alias("pv_local_rolling_mean_1h")
        )

    # PV remote — lags longs uniquement
    if COL_PV_REMOTE in df.columns:
        for lag in PV_REMOTE_LAGS:
            exprs.append(pl.col(COL_PV_REMOTE).shift(lag).alias(f"pv_remote_lag_{lag}"))

    return df.with_columns(exprs) if exprs else df


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 4 — ENCODAGE CYCLIQUE
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 5 — CALENDAIRE
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 6 — MÉTÉO RÉCENTE (t-4, délai réel 1h)
# ══════════════════════════════════════════════════════════════════════════════

def add_meteo_recent(df):
    """Mesures météo avec délai réel de 1h (shift 4, pas shift 1)."""
    log.info(f"  [Groupe 6] Météo récente (t-{METEO_DELAY}, délai 1h)...")
    exprs = [pl.col(c).shift(METEO_DELAY).alias(f"{c}_recent")
             for c in METEO_REAL_COLS if c in df.columns]
    return df.with_columns(exprs) if exprs else df


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 7 — NWP (horizon t+12)
# ══════════════════════════════════════════════════════════════════════════════

def add_nwp_features(df):
    """
    NWP conservées BRUTES (sans shift) dans le DataFrame.

    Le shift(-h) sera appliqué au moment de l'entraînement, par le code
    model_intraday.py, pour chaque horizon h (1 à 12). Ainsi :
      - Le modèle pour target_1 reçoit la NWP de t+1
      - Le modèle pour target_6 reçoit la NWP de t+6
      - Le modèle pour target_12 reçoit la NWP de t+12

    Ça évite de pré-calculer 12 colonnes par variable NWP (5 × 12 = 60)
    ou de forcer un seul horizon arbitraire (ex: t+12).

    Les colonnes NWP sont déjà présentes après la jointure. On vérifie
    juste qu'elles existent.
    """
    log.info("  [Groupe 7] NWP brutes (shift par horizon fait à l'entraînement)...")
    available = [c for c in NWP_COLS if c in df.columns]
    if len(available) < len(NWP_COLS):
        log.warning(f"  NWP manquantes : {set(NWP_COLS) - set(available)}")
    else:
        log.info(f"  {len(available)} NWP disponibles : {available}")
    return df  # colonnes déjà présentes, rien à ajouter


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 8 — INTERACTIONS (sur mesures récentes t-4)
# ══════════════════════════════════════════════════════════════════════════════

def add_interactions(df):
    """temp² et temp×rad sur mesures météo récentes (t-4)."""
    log.info("  [Groupe 8] Interactions sur mesures récentes (t-4)...")
    t = f"{COL_TEMP}_recent"
    r = f"{COL_GLOB}_recent"
    exprs = []
    if t in df.columns:
        exprs.append((pl.col(t)**2).alias("temp_squared"))
    if t in df.columns and r in df.columns:
        exprs.append((pl.col(t)*pl.col(r)).alias("temp_x_rad"))
    return df.with_columns(exprs) if exprs else df


# ══════════════════════════════════════════════════════════════════════════════
# CIBLES Y (multi-horizon)
# ══════════════════════════════════════════════════════════════════════════════

def build_targets(df):
    log.info(f"  Cibles Y : target_1 à target_{HORIZON_MAX}...")
    return df.with_columns([
        pl.col(COL_LOAD).shift(-h).alias(f"target_{h}")
        for h in range(1, HORIZON_MAX + 1)
    ])


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLAGE
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df_oiken, df_meteo_real, df_meteo_pred):
    log.info("\n" + "="*60 + "\nFEATURES INTRADAY (v2)\n" + "="*60)
    log.info("\nJointure des 3 sources...")
    mr = [COL_TIMESTAMP] + [c for c in df_meteo_real.columns if not c.endswith("_gap") and c != COL_TIMESTAMP]
    mp = [COL_TIMESTAMP] + [c for c in df_meteo_pred.columns if not c.endswith("_gap") and c != COL_TIMESTAMP]
    df = (df_oiken
          .join(df_meteo_real.select(mr), on=COL_TIMESTAMP, how="left")
          .join(df_meteo_pred.select(mp), on=COL_TIMESTAMP, how="left"))
    log.info(f"  Shape après jointure : {df.shape}")

    for fn in [add_load_lags, add_rolling_features, add_pv_lags,
               add_cyclical_features, add_calendar_features,
               add_meteo_recent, add_nwp_features, add_interactions, build_targets]:
        df = fn(df)

    n = df.height
    df = df.filter(pl.col(f"target_{HORIZON_MAX}").is_not_null())
    log.info(f"\n  Lignes sans target : {n - df.height}")
    check = [c for c in df.columns if "lag" in c or "rolling" in c]
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
    """
    Liste des colonnes X (tout sauf timestamp, targets, brutes, flags, vent).

    NOTE : les NWP (pred_temperature_ctrl, pred_radiation_ctrl, etc.) sont
    gardées comme features BRUTES. Le shift(-h) pour les aligner sur chaque
    horizon sera fait par le code d'entraînement, PAS ici.
    """
    exclude = {COL_TIMESTAMP, COL_FORECAST_LOAD, "net_load",
               COL_PV_LOCAL, COL_PV_REMOTE, COL_PV_CENTRAL, COL_PV_SIERRE,
               COL_TEMP, COL_GLOB, COL_PRECIP, COL_HUMIDITY,
               "sunshine_min", "wind_speed_ms", COL_LOAD,
               # Vent exclu (r=-0.252, pas d'éolien)
               "pred_wind_ctrl", "pred_wind_std"}
    # NWP_COLS ne sont PAS exclues : elles restent comme features brutes
    # Le shift par horizon sera fait à l'entraînement
    return sorted([c for c in df.columns
                   if c not in exclude and not c.endswith("_gap") and not c.startswith("target_")])

def get_target_columns():
    return [f"target_{h}" for h in range(1, HORIZON_MAX + 1)]


def get_nwp_columns():
    """Retourne la liste des colonnes NWP brutes présentes dans le parquet."""
    return list(NWP_COLS)


def build_nwp_for_horizon(df: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """
    Utilitaire pour le code d'entraînement.

    Pour un horizon donné (1 à 12), crée les colonnes NWP shiftées
    de -horizon et renommées en {col}_h{horizon}.

    Exemple pour horizon=4 :
      pred_temperature_ctrl → shift(-4) → pred_temperature_ctrl_h4

    Usage dans model_intraday.py :
        from features_intraday import build_nwp_for_horizon
        df_train = build_nwp_for_horizon(df_train, h=4)
        # Puis entraîner le modèle pour target_4 avec les colonnes _h4

    Retourne le DataFrame avec les nouvelles colonnes ajoutées.
    """
    available = [c for c in NWP_COLS if c in df.columns]
    exprs = [pl.col(c).shift(-horizon).alias(f"{c}_h{horizon}") for c in available]
    return df.with_columns(exprs) if exprs else df

def run_feature_engineering_intraday():
    log.info("Chargement données nettoyées (v2)...")
    df_o = pl.read_parquet(PROCESSED_DIR / "oiken_clean_v2.parquet")
    df_r = pl.read_parquet(PROCESSED_DIR / "meteo_real_clean.parquet")
    df_p = pl.read_parquet(PROCESSED_DIR / "meteo_pred_clean.parquet")
    df = build_feature_matrix(df_o, df_r, df_p)
    tr, te = train_test_split(df)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    tr.write_parquet(FEATURES_DIR / "train_intraday_v2.parquet")
    te.write_parquet(FEATURES_DIR / "test_intraday_v2.parquet")
    log.info("✓ Sauvegardé train_intraday_v2 / test_intraday_v2")
    fc = get_feature_columns(df)
    log.info(f"\n── Intraday v2 : {len(fc)} features, {HORIZON_MAX} cibles ──")
    for c in fc:
        n = df[c].null_count()
        log.info(f"  [{'✓' if n==0 else f'! {100*n/df.height:.1f}%'}] {c}")
    return tr, te

if __name__ == "__main__":
    tr, te = run_feature_engineering_intraday()
    fc = get_feature_columns(tr)
    tc = get_target_columns()
    print(f"\nTrain: {tr.shape} | Test: {te.shape} | Features: {len(fc)} | Targets: {len(tc)}")
    for i, c in enumerate(fc, 1):
        print(f"  {i:2d}. {c}")