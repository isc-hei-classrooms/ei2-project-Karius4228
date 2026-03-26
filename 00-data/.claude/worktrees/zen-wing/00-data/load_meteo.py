"""
load_meteo.py — Ingestion des données météo depuis InfluxDB
────────────────────────────────────────────────────────────
Récupère 2 types de données du bucket MeteoSuisse (timeseries.hevs.ch) :

  1. Mesures réelles (T_2M, GLOB, TOT_PREC, RELHUM_2M, DURSUN, FF_10M)
     → utilisées comme lags (valeurs passées connues)

  2. Prévisions NWP (PRED_*_ctrl / _stde)
     → disponibles AVANT gate closure J+1, utilisables pour Day-Ahead

Séparation obligatoire : utiliser les mesures réelles de J+1 serait du leakage.

Entrée  : InfluxDB (timeseries.hevs.ch / MeteoSuisse)
Sortie  : data/processed/meteo_real.parquet + meteo_pred.parquet

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


# ── Connexion InfluxDB ────────────────────────────────────────────────────────

def get_client() -> InfluxDBClient:
    """Client InfluxDB avec certificat SSL (certifi pour HTTPS hevs.ch)."""
    return InfluxDBClient(
        url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG,
        ssl_ca_cert=certifi.where(), timeout=1_000_000,
    )


# ── Requête Flux : 1 variable / 1 site ───────────────────────────────────────

def query_measurement(
    client: InfluxDBClient, measurement: str, site: str,
    start: str, stop: str, agg_window: str = "15m",
) -> pl.DataFrame | None:
    """
    Requête Flux pour un measurement sur un site et une période.
    Agrège en fenêtres de 15 min (moyenne) pour s'aligner sur Oiken.
    Retourne un DataFrame (timestamp, value) ou None si pas de données.
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

    # Extraction des records en listes Python puis conversion Polars
    times, values = [], []
    for table in tables:
        for record in table.records:
            times.append(record["_time"])
            values.append(record["_value"])

    if not times:
        log.warning(f"  → Aucune donnée")
        return None

    df = (
        pl.DataFrame({COL_TIMESTAMP: times, "value": values})
        .with_columns(pl.col(COL_TIMESTAMP).dt.convert_time_zone("UTC"))
        .sort(COL_TIMESTAMP)
    )
    log.info(f"  → {df.height} points")
    return df


# ── Chargement d'un groupe de variables ───────────────────────────────────────

def load_measurement_group(
    measurements_map: dict, start: str, stop: str, site: str,
) -> pl.DataFrame:
    """
    Charge chaque variable une par une, agrège en 15 min,
    et les joint progressivement sur le timestamp.
    """
    client = get_client()
    df_all = None

    for col_name, measurement in measurements_map.items():
        try:
            df_tmp = query_measurement(client, measurement, site, start, stop)
            if df_tmp is None:
                continue

            # Renomme "value" → nom de la variable, puis agrège en 15 min
            df_tmp = (
                df_tmp.rename({"value": col_name})
                .sort(COL_TIMESTAMP)
                .group_by_dynamic(COL_TIMESTAMP, every="15m")
                .agg(pl.col(col_name).mean())
            )

            # Join incrémental : jamais plus de 2 DataFrames en RAM
            if df_all is None:
                df_all = df_tmp
            else:
                df_all = df_all.join(df_tmp, on=COL_TIMESTAMP, how="full", coalesce=True).sort(COL_TIMESTAMP)

            log.info(f"  ✓ {col_name} | shape : {df_all.shape}")

        except Exception as e:
            log.error(f"  Échec {col_name} : {e}")

    client.close()
    if df_all is None:
        raise RuntimeError("Aucune donnée récupérée.")

    return df_all.unique(subset=[COL_TIMESTAMP], keep="last", maintain_order=True).sort(COL_TIMESTAMP)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _reindex_continuous(df: pl.DataFrame) -> pl.DataFrame:
    """Grille 15 min complète entre min et max timestamp. Trous → null."""
    grid = pl.DataFrame({
        COL_TIMESTAMP: pl.datetime_range(
            start=df[COL_TIMESTAMP].min(), end=df[COL_TIMESTAMP].max(),
            interval=FREQ, time_zone=TIMEZONE, eager=True,
        )
    })
    return grid.join(df, on=COL_TIMESTAMP, how="left")


def _log_nulls(df: pl.DataFrame, label: str):
    """Affiche le taux de nulls par colonne."""
    log.info(f"\n  Nulls [{label}] :")
    for col in df.columns:
        if col == COL_TIMESTAMP:
            continue
        n = df[col].null_count()
        log.info(f"  {'✓' if n == 0 else '!'} {col:35s}: {n:5d} ({100*n/df.height:.1f}%)")


# ── Fonctions principales ────────────────────────────────────────────────────

def load_meteo_real(start: str, stop: str, site: str = None, output_name: str = "meteo_real.parquet") -> pl.DataFrame:
    """Charge les 6 variables météo réelles → Parquet."""
    site = site or METEO_SITE_PRIMARY
    log.info(f"\n═══ Mesures réelles | {site} | {start} → {stop} ═══")

    df = load_measurement_group(INFLUX_MEASUREMENTS_REAL, start, stop, site)
    df = _reindex_continuous(df)
    _log_nulls(df, "Mesures réelles")

    output_path = PROCESSED_DIR / output_name
    df.write_parquet(output_path)
    log.info(f"✓ Sauvegardé : {output_path} | {df.shape}")
    return df


def load_meteo_pred(start: str, stop: str, site: str = None, output_name: str = "meteo_pred.parquet") -> pl.DataFrame:
    """Charge les prévisions NWP (PRED_*) → Parquet."""
    site = site or METEO_SITE_PRIMARY
    log.info(f"\n═══ Prévisions NWP | {site} | {start} → {stop} ═══")

    df = load_measurement_group(INFLUX_MEASUREMENTS_PRED, start, stop, site)
    df = _reindex_continuous(df)
    _log_nulls(df, "Prévisions NWP")

    output_path = PROCESSED_DIR / output_name
    df.write_parquet(output_path)
    log.info(f"✓ Sauvegardé : {output_path} | {df.shape}")
    return df


def load_meteo_all(start: str, stop: str, site: str = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Charge mesures réelles + prévisions NWP. Retourne (df_real, df_pred)."""
    return load_meteo_real(start, stop, site), load_meteo_pred(start, stop, site)


def load_parquet(filename: str) -> pl.DataFrame:
    """Recharge un Parquet depuis processed/ (évite de requêter InfluxDB)."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable")
    df = pl.read_parquet(path)
    log.info(f"Chargé : {path} | {df.shape}")
    return df


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    START = "2022-09-30T22:00:00Z"
    STOP  = "2025-09-29T22:15:00Z"

    df_real, df_pred = load_meteo_all(start=START, stop=STOP)

    print(f"\nMesures réelles : {df_real.shape}")
    print(df_real.head(5))
    print(f"\nPrévisions NWP : {df_pred.shape}")
    print(df_pred.head(5))