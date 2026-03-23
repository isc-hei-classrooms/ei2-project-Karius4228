"""
src/acquisition/load_meteo.py
──────────────────────────────
Ingestion des données météo depuis InfluxDB (timeseries.hevs.ch).

Deux types de données disponibles dans le bucket MeteoSuisse :

  1. MESURES RÉELLES — données historiques mesurées par les stations
     Utilisées comme features de lag dans X (valeurs passées connues)
     Variables : T_2M, GLOB, TOT_PREC, RELHUM_2M, DURSUN, FF_10M

  2. PRÉVISIONS NWP (Numerical Weather Prediction) — PRED_*
     Disponibles AVANT le gate closure J+1 → utilisables pour Day-Ahead
     Chaque variable a 4 versions : ctrl (centrale), stde, q10, q90
     On récupère ctrl (valeur prévue) + stde (incertitude de prévision)
     Variables : PRED_T_2M, PRED_GLOB, PRED_TOT_PREC, PRED_FF_10M,
                 PRED_DURSUN, PRED_RELHUM_2M

Pourquoi séparer mesures et prévisions ?
  Pour le modèle Day-Ahead (prédiction à 11h pour J+1) :
  - On ne peut PAS utiliser les mesures réelles de J+1 (pas encore disponibles)
  - On PEUT utiliser les prévisions NWP de J+1 (publiées la veille)
  - On PEUT utiliser les mesures réelles de J-1, J-7 etc. (lags passés)
  Mélanger les deux sans distinction = data leakage.

Auteur : Marius Fabbri
"""

import polars as pl
import certifi
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import (
    INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET,
    INFLUX_MEASUREMENTS_REAL, INFLUX_MEASUREMENTS_PRED,
    METEO_SITE_PRIMARY, PROCESSED_DIR, TIMEZONE, FREQ, COL_TIMESTAMP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONNEXION
# ─────────────────────────────────────────────────────────────────────────────

def get_client() -> InfluxDBClient:
    """
    Crée un client InfluxDB avec SSL (ssl_ca_cert=certifi.where()).
    certifi fournit le bundle de certificats CA nécessaire pour valider
    le certificat HTTPS de timeseries.hevs.ch.
    """
    return InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG,
        ssl_ca_cert=certifi.where(),
        timeout=1_000_000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REQUÊTE FLUX — un measurement / un site
# ─────────────────────────────────────────────────────────────────────────────

def query_measurement(
    client: InfluxDBClient,
    measurement: str,
    site: str,
    start: str,
    stop: str,
    agg_window: str = "15m",
) -> pl.Series:
    """
    Récupère un measurement pour un site sur une période et retourne
    une pl.Series avec index temporel UTC.

    Paramètres
    ----------
    measurement : nom exact du measurement InfluxDB
    site        : tag Site (ex: "Sion")
    start/stop  : RFC3339 ou relatif (ex: "-7d", "now()")
    agg_window  : fenêtre d'agrégation (défaut "15m" pour aligner sur Oiken)

    La requête Flux :
      range()          → filtre sur la période
      filter()         → sélectionne le measurement et le site
      aggregateWindow() → moyenne sur fenêtres de 15min
                          createEmpty:false → pas de lignes vides
    """
    flux = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start}, stop: {stop})
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["Site"] == "{site}")
      |> aggregateWindow(every: {agg_window}, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """

    log.info(f"  [{site}] {measurement[:50]}")

    try:
        tables = client.query_api().query(org=INFLUX_ORG, query=flux)
    except InfluxDBError as e:
        log.error(f"  Erreur InfluxDB : {e}")
        raise

    times, values = [], []
    for table in tables:
        for record in table.records:
            times.append(record["_time"])
            values.append(record["_value"])

    if not times:
        log.warning(f"  → Aucune donnée")
        return None

    # Construction d'un DataFrame temporaire puis conversion en Series
    df_tmp = pl.DataFrame({
        COL_TIMESTAMP: times,
        "value": values,
    }).with_columns(
        pl.col(COL_TIMESTAMP).dt.convert_time_zone("UTC")
    ).sort(COL_TIMESTAMP)

    log.info(f"  → {df_tmp.height} points")
    return df_tmp


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT D'UN GROUPE DE MEASUREMENTS
# ─────────────────────────────────────────────────────────────────────────────

def load_measurement_group(
    measurements_map: dict,
    start: str,
    stop: str,
    site: str,
) -> pl.DataFrame:
    """
    Charge les measurements un par un et les joint progressivement.
    Chaque colonne est d'abord sauvegardée en Parquet temporaire pour
    libérer la mémoire avant de charger la suivante.
    """
    client = get_client()
    df_all = None

    for col_name, measurement in measurements_map.items():
        try:
            df_tmp = query_measurement(
                client=client,
                measurement=measurement,
                site=site,
                start=start,
                stop=stop,
            )
            if df_tmp is None:
                continue

            df_tmp = df_tmp.rename({"value": col_name})

            # Agrégation en 15min AVANT le join pour réduire la taille
            # 310 550 points → ~105 000 après agrégation 15min
            df_tmp = (
                df_tmp
                .sort(COL_TIMESTAMP)
                .group_by_dynamic(COL_TIMESTAMP, every="15m")
                .agg(pl.col(col_name).mean())
            )

            if df_all is None:
                df_all = df_tmp
            else:
                # Join incrémental : on n'a jamais plus de 2 colonnes en RAM
                df_all = df_all.join(df_tmp, on=COL_TIMESTAMP, how="full", coalesce=True)
                df_all = df_all.sort(COL_TIMESTAMP)

            log.info(f"  ✓ {col_name} joint | shape courant : {df_all.shape}")

        except Exception as e:
            log.error(f"  Échec {col_name} : {e}")

    client.close()

    if df_all is None:
        raise RuntimeError("Aucune donnée récupérée.")

    # Dédupliquer
    df_all = df_all.unique(subset=[COL_TIMESTAMP], keep="last", maintain_order=True)
    return df_all.sort(COL_TIMESTAMP)


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────

def load_meteo_real(
    start: str,
    stop: str,
    site: str = None,
    output_name: str = "meteo_real.parquet",
) -> pl.DataFrame:
    """
    Charge les 6 variables météo mesurées réelles sur la période.
    Ces données sont utilisées comme features de lag (passé connu).

    Variables : T_2M, GLOB, TOT_PREC, RELHUM_2M, DURSUN, FF_10M
    """
    if site is None:
        site = METEO_SITE_PRIMARY

    log.info(f"\n═══ Mesures réelles | site={site} | {start} → {stop} ═══")

    df = load_measurement_group(
        measurements_map=INFLUX_MEASUREMENTS_REAL,
        start=start,
        stop=stop,
        site=site,
    )

    # Reconstruction index continu 15min (trous → null)
    df = _reindex_continuous(df)

    _log_nulls(df, "Mesures réelles")

    output_path = PROCESSED_DIR / output_name
    df.write_parquet(output_path)
    log.info(f"✓ Sauvegardé : {output_path} | shape : {df.shape}")
    return df


def load_meteo_pred(
    start: str,
    stop: str,
    site: str = None,
    output_name: str = "meteo_pred.parquet",
) -> pl.DataFrame:
    """
    Charge les prévisions NWP (PRED_*) sur la période.
    Ces données sont disponibles avant le gate closure et peuvent donc
    être utilisées comme features pour la prédiction Day-Ahead J+1.

    Variables : PRED_T_2M, PRED_GLOB, PRED_TOT_PREC, PRED_FF_10M,
                PRED_DURSUN, PRED_RELHUM_2M (ctrl + stde)
    """
    if site is None:
        site = METEO_SITE_PRIMARY

    log.info(f"\n═══ Prévisions NWP | site={site} | {start} → {stop} ═══")

    df = load_measurement_group(
        measurements_map=INFLUX_MEASUREMENTS_PRED,
        start=start,
        stop=stop,
        site=site,
    )

    df = _reindex_continuous(df)
    _log_nulls(df, "Prévisions NWP")

    output_path = PROCESSED_DIR / output_name
    df.write_parquet(output_path)
    log.info(f"✓ Sauvegardé : {output_path} | shape : {df.shape}")
    return df


def load_meteo_all(
    start: str,
    stop: str,
    site: str = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Charge les mesures réelles ET les prévisions NWP en un seul appel.
    Retourne un tuple (df_real, df_pred).
    """
    df_real = load_meteo_real(start, stop, site)
    df_pred = load_meteo_pred(start, stop, site)
    return df_real, df_pred


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES INTERNES
# ─────────────────────────────────────────────────────────────────────────────

def _reindex_continuous(df: pl.DataFrame) -> pl.DataFrame:
    """Reconstruit un index UTC 15min continu, trous → null."""
    grid = pl.DataFrame({
        COL_TIMESTAMP: pl.datetime_range(
            start=df[COL_TIMESTAMP].min(),
            end=df[COL_TIMESTAMP].max(),
            interval=FREQ,
            time_zone=TIMEZONE,
            eager=True,
        )
    })
    return grid.join(df, on=COL_TIMESTAMP, how="left")


def _log_nulls(df: pl.DataFrame, label: str) -> None:
    """Affiche le taux de nulls par colonne."""
    log.info(f"\n  Nulls [{label}] :")
    for col in df.columns:
        if col == COL_TIMESTAMP:
            continue
        n = df[col].null_count()
        pct = 100 * n / df.height
        symbol = "✓" if n == 0 else "!"
        log.info(f"  [{symbol}] {col:35s}: {n:5d} ({pct:.1f}%)")


def load_parquet(filename: str) -> pl.DataFrame:
    """Recharge un Parquet depuis data/processed/ (évite de requêter InfluxDB)."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {path}")
    df = pl.read_parquet(path)
    log.info(f"Chargé : {path} | shape : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST : python load_meteo.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Période alignée sur les données Oiken
    START = "2022-09-30T22:00:00Z"
    STOP  = "2025-09-29T22:15:00Z"

    log.info("Chargement mesures réelles + prévisions NWP...")
    df_real, df_pred = load_meteo_all(start=START, stop=STOP)

    print("\n── Mesures réelles ─────────────────────────────────")
    print(df_real.head(5))
    print(f"Shape : {df_real.shape}")

    print("\n── Prévisions NWP ──────────────────────────────────")
    print(df_pred.head(5))
    print(f"Shape : {df_pred.shape}")