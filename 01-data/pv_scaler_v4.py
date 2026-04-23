"""
pv_scaler_v4.py — Multiplicateur de signal PV (proxy capacité installée)
─────────────────────────────────────────────────────────────────────────
Calcule un facteur de normalisation mensuel basé sur l'évolution de la
charge nette minimale en heures solaires (10h–14h en été).

Principe physique :
  load_net = load_brut - production_PV
  Quand le PV augmente, le minimum de load_net (heures solaires) descend.
  Ce minimum mensuel est donc un proxy fidèle de la capacité PV installée.

Utilisation :
  - Appelé UNE FOIS avant features_da_v4.py pour créer pv_scaler_v4.parquet
  - Le scaler est ensuite joint aux features et utilisé pour normaliser/dénormaliser

Sortie :
  data/processed/pv_scaler_v4.parquet
    colonnes : year (int), month (int), pv_scaler (float32)

Auteur : Marius Fabbri
"""

import polars as pl
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR    = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"

# ── Paramètres du scaler ──────────────────────────────────────────────────────
# Heures solaires : on ne prend que les pas où le soleil peut produire
SOLAR_HOUR_MIN = 10
SOLAR_HOUR_MAX = 14

# Quantile bas de la charge nette → proxy du pic PV
# q=0.05 au lieu du strict minimum pour éviter les outliers (jours très nuageux ignorés)
QUANTILE_LOW = 0.05

# Smoothing : moyenne glissante sur N mois pour éviter les sauts brutaux
# (ex: un mois avec beaucoup de nuages donnera un faux creux)
SMOOTH_WINDOW = 3

# Valeur de référence : on ancre le scaler sur le PREMIER mois du dataset
# → scaler[premier_mois] = 1.0 (valeur de référence)
# → scaler[mois_suivant] = ratio par rapport au premier mois
# Ainsi un scaler=1.3 signifie "30% plus de PV qu'au début du dataset"
ANCHOR_TO_FIRST = True


def compute_pv_scaler(df_oiken: pl.DataFrame) -> pl.DataFrame:
    """
    Calcule le scaler mensuel à partir de la charge Oiken nettoyée.

    Retourne un DataFrame avec colonnes : year, month, pv_scaler
    """
    log.info("Calcul du proxy capacité PV (quantile bas charge nette heures solaires)")

    # ── Étape 1 : filtrer les heures solaires ────────────────────────────────
    df_solar = (
        df_oiken
        .with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.hour().alias("hour"),
        ])
        .filter(
            (pl.col("hour") >= SOLAR_HOUR_MIN) &
            (pl.col("hour") < SOLAR_HOUR_MAX)
        )
    )

    log.info(f"  Pas en heures solaires ({SOLAR_HOUR_MIN}h–{SOLAR_HOUR_MAX}h) : {df_solar.height}")

    # ── Étape 2 : quantile bas par mois ──────────────────────────────────────
    # Le quantile bas de la charge nette ≈ −capacité_PV_peak × irradiance_max
    # Plus la capacité augmente, plus ce quantile descend (devient plus négatif)
    df_monthly = (
        df_solar
        .group_by(["year", "month"])
        .agg(
            pl.col("load").quantile(QUANTILE_LOW).alias("load_q05"),
            pl.col("load").count().alias("n_obs"),
        )
        .sort(["year", "month"])
    )

    log.info(f"  Mois calculés : {df_monthly.height}")

    # ── Étape 3 : vérification de la couverture ───────────────────────────────
    # On vérifie que chaque mois a assez d'observations (min 4 jours × 16 pas = 64)
    n_sparse = df_monthly.filter(pl.col("n_obs") < 64).height
    if n_sparse > 0:
        log.warning(f"  {n_sparse} mois avec < 64 observations → scaler moins fiable")

    # ── Étape 4 : la charge nette minimale est négative en été (PV dominant)
    # Pour créer un multiplicateur positif et croissant avec le PV :
    #   raw_signal = -load_q05  (on inverse pour que ça croisse avec le PV)
    #   Si le minimum mensuel est -1.5 → raw_signal = 1.5
    #   Si le minimum mensuel est -2.0 → raw_signal = 2.0 (plus de PV)
    #
    # En hiver, load_q05 est positif (pas de PV) → raw_signal négatif
    # On clippe à 1.0 pour les mois sans PV significatif
    df_monthly = df_monthly.with_columns(
        pl.col("load_q05").neg().clip(lower_bound=0.5).alias("raw_signal")
    )

    # ── Étape 5 : smoothing temporel ─────────────────────────────────────────
    # Moyenne glissante pour éviter les sauts dus à la météo mensuelle
    df_monthly = df_monthly.with_columns(
        pl.col("raw_signal")
          .rolling_mean(window_size=SMOOTH_WINDOW, min_periods=1)
          .alias("raw_signal_smooth")
    )

    # ── Étape 6 : normalisation relative au premier mois ─────────────────────
    if ANCHOR_TO_FIRST:
        first_val = df_monthly["raw_signal_smooth"][0]
        if first_val == 0:
            log.warning("  Premier mois = 0, ancrage à 1.0 par défaut")
            first_val = 1.0
        df_monthly = df_monthly.with_columns(
            (pl.col("raw_signal_smooth") / first_val).alias("pv_scaler")
        )
    else:
        # Normalisation globale [0,1] → moins interprétable
        vmin = df_monthly["raw_signal_smooth"].min()
        vmax = df_monthly["raw_signal_smooth"].max()
        df_monthly = df_monthly.with_columns(
            ((pl.col("raw_signal_smooth") - vmin) / (vmax - vmin + 1e-8)).alias("pv_scaler")
        )

    # Sécurité : scaler toujours > 0 (évite division par zéro à la dénorm)
    df_monthly = df_monthly.with_columns(
        pl.col("pv_scaler").clip(lower_bound=0.1).alias("pv_scaler")
    )

    # ── Affichage diagnostic ──────────────────────────────────────────────────
    log.info("\n── Scaler mensuel (extrait) ──")
    log.info(f"  {'Année':>6} {'Mois':>5} {'Q05 charge':>12} {'Signal brut':>12} {'Scaler':>8}")
    for row in df_monthly.iter_rows(named=True):
        log.info(
            f"  {row['year']:>6} {row['month']:>5} "
            f"{row['load_q05']:>12.4f} {row['raw_signal']:>12.4f} "
            f"{row['pv_scaler']:>8.4f}"
        )

    # ── Vérification : le scaler doit croître globalement ────────────────────
    # (la capacité PV ne peut pas diminuer)
    scalers = df_monthly["pv_scaler"].to_numpy()
    # On vérifie la tendance sur les mois d'été seulement (3-9) car hiver = pas de signal PV
    summer_mask = df_monthly.with_columns(
        ((pl.col("month") >= 4) & (pl.col("month") <= 9)).alias("is_summer")
    ).filter(pl.col("is_summer"))["pv_scaler"].to_numpy()

    if len(summer_mask) > 2:
        trend = np.polyfit(np.arange(len(summer_mask)), summer_mask, 1)[0]
        log.info(f"\n  Tendance été (doit être > 0) : {trend:+.4f}/mois")
        if trend < 0:
            log.warning("  ATTENTION : tendance négative sur le scaler été — vérifier les données !")

    return df_monthly.select(["year", "month", "pv_scaler",
                               "raw_signal", "load_q05", "n_obs"])


def build_scaler_lookup(df_scaler: pl.DataFrame, df_full: pl.DataFrame) -> pl.DataFrame:
    """
    Joint le scaler mensuel sur le DataFrame complet (une valeur par pas de 15min).
    Retourne df_full avec une colonne 'pv_scaler' ajoutée.

    Pour les mois futurs non vus à l'entraînement : extrapolation par
    la tendance linéaire des 6 derniers mois d'été.
    """
    df_with_ym = df_full.with_columns([
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
    ])

    df_joined = df_with_ym.join(
        df_scaler.select(["year", "month", "pv_scaler"]),
        on=["year", "month"],
        how="left"
    )

    # Extrapolation pour les mois non couverts (forward fill + tendance)
    n_null = df_joined["pv_scaler"].null_count()
    if n_null > 0:
        log.warning(f"  {n_null} pas sans scaler → forward fill")
        df_joined = df_joined.with_columns(
            pl.col("pv_scaler").forward_fill().backward_fill()
        )

    return df_joined.drop(["year", "month"])


def run():
    log.info("=" * 60)
    log.info("PV SCALER v4 — Calcul du multiplicateur de signal")
    log.info("=" * 60)

    df_oiken = pl.read_parquet(PROCESSED_DIR / "oiken_clean_v2.parquet")
    log.info(f"Données Oiken : {df_oiken.shape} | {df_oiken['timestamp'].min()} → {df_oiken['timestamp'].max()}")

    df_scaler = compute_pv_scaler(df_oiken)

    # Sauvegarde
    out_path = PROCESSED_DIR / "pv_scaler_v4.parquet"
    df_scaler.write_parquet(out_path)
    log.info(f"\n✓ Scaler sauvegardé : {out_path}")
    log.info(f"  {df_scaler.height} mois | scaler min={df_scaler['pv_scaler'].min():.4f} max={df_scaler['pv_scaler'].max():.4f}")

    return df_scaler


if __name__ == "__main__":
    run()
