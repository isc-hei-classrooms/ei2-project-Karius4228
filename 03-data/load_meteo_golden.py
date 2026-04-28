"""
load_meteo_golden.py — Extraction météo pour le golden dataset
───────────────────────────────────────────────────────────────
Adapté de load_meteo.py pour couvrir la période du golden dataset :
  Sept 2023 → Avril 2026

On extrait uniquement les données NOUVELLES par rapport à ce qui existe
déjà dans 01-data (qui s'arrête au 29 sept 2025).

Stratégie :
  - On re-extrait toute la période golden (sept 2023 → avr 2026)
  - Cela couvre le chevauchement ET les nouveaux mois post-sept 2025
  - Les fichiers sont sauvegardés dans 03-data/data/processed/

Sortie :
  03-data/data/processed/meteo_real_clean.parquet   (remplace celui copié)
  03-data/data/processed/meteo_pred_clean.parquet   (remplace celui copié)

Usage :
  python load_meteo_golden.py

Auteur : Marius Fabbri
"""

import polars as pl
import certifi
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from pathlib import Path
import logging
import sys

# ── Import config depuis 01-data (où elle est définie) ───────────────────────
SRC_DIR = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\01-data")
sys.path.insert(0, str(SRC_DIR))
from config import (
    INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET,
    INFLUX_MEASUREMENTS_REAL, INFLUX_MEASUREMENTS_PRED,
    METEO_SITE_PRIMARY, TIMEZONE, FREQ, COL_TIMESTAMP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Chemins de sortie : 03-data ───────────────────────────────────────────────
PROJECT_ROOT  = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228")
PROCESSED_DIR = PROJECT_ROOT / "03-data" / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Période du golden dataset ─────────────────────────────────────────────────
# Golden : 01/09/2023 00:15 → 22/04/2026 00:00
# On prend un peu de marge avant pour avoir les lags J-7 complets dès le début
# (les features ont besoin de 7 jours de passé → démarrer le 24 août 2023)
START = "2023-08-24T22:00:00Z"  # marge J-7 avant le 01/09/2023
STOP  = "2026-04-22T00:15:00Z"  # dernier pas du golden dataset


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS (identiques à load_meteo.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_client() -> InfluxDBClient:
    return InfluxDBClient(
        url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG,
        ssl_ca_cert=certifi.where(), timeout=1_000_000,
    )


def query_measurement(client, measurement, site, start, stop, agg_window="15m"):
    flux = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start}, stop: {stop})
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["Site"] == "{site}")
      |> aggregateWindow(every: {agg_window}, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    """
    log.info(f"  [{site}] {measurement[:55]}")
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
        log.warning(f"  → Aucune donnée pour {measurement}")
        return None

    df = (
        pl.DataFrame({COL_TIMESTAMP: times, "value": values})
        .with_columns(pl.col(COL_TIMESTAMP).dt.convert_time_zone("UTC"))
        .sort(COL_TIMESTAMP)
    )
    log.info(f"  → {df.height} points | {df[COL_TIMESTAMP].min()} → {df[COL_TIMESTAMP].max()}")
    return df


def load_measurement_group(measurements_map, start, stop, site):
    client = get_client()
    df_all = None

    for col_name, measurement in measurements_map.items():
        try:
            df_tmp = query_measurement(client, measurement, site, start, stop)
            if df_tmp is None:
                continue

            df_tmp = (
                df_tmp.rename({"value": col_name})
                .sort(COL_TIMESTAMP)
                .group_by_dynamic(COL_TIMESTAMP, every="15m")
                .agg(pl.col(col_name).mean())
            )

            if df_all is None:
                df_all = df_tmp
            else:
                df_all = df_all.join(df_tmp, on=COL_TIMESTAMP, how="full", coalesce=True).sort(COL_TIMESTAMP)

            log.info(f"  ✓ {col_name} | shape cumulé : {df_all.shape}")

        except Exception as e:
            log.error(f"  Échec {col_name} : {e}")

    client.close()
    if df_all is None:
        raise RuntimeError("Aucune donnée récupérée depuis InfluxDB.")

    return df_all.unique(subset=[COL_TIMESTAMP], keep="last", maintain_order=True).sort(COL_TIMESTAMP)


def reindex_continuous(df):
    """Grille 15min complète entre min et max timestamp. Trous → null."""
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


def log_nulls(df, label):
    log.info(f"\n  Nulls [{label}] :")
    for col in df.columns:
        if col == COL_TIMESTAMP:
            continue
        n = df[col].null_count()
        pct = 100 * n / df.height
        icon = "✓" if n == 0 else ("!" if pct > 5 else "~")
        log.info(f"  {icon} {col:40s}: {n:6d} ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 65)
    log.info("EXTRACTION MÉTÉO — Golden Dataset")
    log.info(f"Période : {START}  →  {STOP}")
    log.info(f"Site    : {METEO_SITE_PRIMARY}")
    log.info(f"Sortie  : {PROCESSED_DIR}")
    log.info("=" * 65)

    # ── Mesures réelles ───────────────────────────────────────────────────────
    log.info("\n── Mesures réelles (T_2M, GLOB, TOT_PREC, RELHUM_2M...) ──")
    df_real = load_measurement_group(INFLUX_MEASUREMENTS_REAL, START, STOP, METEO_SITE_PRIMARY)
    df_real = reindex_continuous(df_real)
    log_nulls(df_real, "Mesures réelles")

    out_real = PROCESSED_DIR / "meteo_real_clean.parquet"
    df_real.write_parquet(out_real)
    log.info(f"\n✓ Sauvegardé : {out_real}")
    log.info(f"  {df_real.shape} | {df_real[COL_TIMESTAMP].min()} → {df_real[COL_TIMESTAMP].max()}")

    # ── Prévisions NWP ────────────────────────────────────────────────────────
    log.info("\n── Prévisions NWP (PRED_*_ctrl / _stde) ──")
    df_pred = load_measurement_group(INFLUX_MEASUREMENTS_PRED, START, STOP, METEO_SITE_PRIMARY)
    df_pred = reindex_continuous(df_pred)
    log_nulls(df_pred, "Prévisions NWP")

    out_pred = PROCESSED_DIR / "meteo_pred_clean.parquet"
    df_pred.write_parquet(out_pred)
    log.info(f"\n✓ Sauvegardé : {out_pred}")
    log.info(f"  {df_pred.shape} | {df_pred[COL_TIMESTAMP].min()} → {df_pred[COL_TIMESTAMP].max()}")

    # ── Résumé ────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("✓ Extraction terminée")
    log.info(f"  meteo_real_clean.parquet : {df_real.height:,} lignes")
    log.info(f"  meteo_pred_clean.parquet : {df_pred.height:,} lignes")
    log.info("\nProchaine étape : lancer build_golden_parquet.py")
    log.info("=" * 65)

    return df_real, df_pred


if __name__ == "__main__":
    run()
