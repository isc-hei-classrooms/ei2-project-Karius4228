"""
load_oiken.py — Ingestion des données de charge Oiken
─────────────────────────────────────────────────────────
Lit le CSV brut Oiken (~105 120 lignes, 3 ans à 15 min),
le nettoie, et produit un Parquet avec index temporel continu.

Entrée  : data/raw/oiken/oiken-data.csv
Sortie  : data/processed/oiken_raw.parquet (105 120 × 9)

Auteur : Marius Fabbri
"""

import polars as pl
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import (
    RAW_OIKEN_DIR, PROCESSED_DIR,
    OIKEN_COL_TIMESTAMP, OIKEN_COL_LOAD, OIKEN_COL_FORECAST_LOAD,
    OIKEN_COL_PV_CENTRAL, OIKEN_COL_PV_SION,
    OIKEN_COL_PV_SIERRE, OIKEN_COL_PV_REMOTE,
    COL_TIMESTAMP, COL_LOAD, COL_FORECAST_LOAD,
    COL_PV_CENTRAL, COL_PV_SION, COL_PV_SIERRE, COL_PV_REMOTE,
    COL_PV_TOTAL,
    TIMEZONE, FREQ,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Mapping colonnes CSV brutes → noms projet ────────────────────────────────
RENAME_MAP = {
    OIKEN_COL_TIMESTAMP:     COL_TIMESTAMP,
    OIKEN_COL_LOAD:          COL_LOAD,
    OIKEN_COL_FORECAST_LOAD: COL_FORECAST_LOAD,
    OIKEN_COL_PV_CENTRAL:    COL_PV_CENTRAL,
    OIKEN_COL_PV_SION:       COL_PV_SION,
    OIKEN_COL_PV_SIERRE:     COL_PV_SIERRE,
    OIKEN_COL_PV_REMOTE:     COL_PV_REMOTE,
}

PV_COLS = [COL_PV_CENTRAL, COL_PV_SION, COL_PV_SIERRE, COL_PV_REMOTE]


def load_oiken_csv(csv_path: Path | str) -> pl.DataFrame:
    """
    Lit UN fichier CSV Oiken et retourne un DataFrame propre.

    Étapes :
      1. Lecture CSV (gestion des #N/A et dates auto-parsées)
      2. Sélection des 7 colonnes utiles + renommage court
      3. Conversion timestamps heure locale Zurich → UTC
      4. Tri chronologique
      5. Ajout pv_total (somme 4 zones) et net_load (load − pv_total)
    """
    csv_path = Path(csv_path)
    log.info(f"Lecture CSV : {csv_path.name}")

    # 1. Lecture brute
    df = pl.read_csv(
        csv_path,
        try_parse_dates=True,                       # détection auto du timestamp
        null_values=["#N/A", "N/A", "", "null"],    # valeurs manquantes du CSV
        ignore_errors=True,                         # lignes malformées → null
    )

    # 2. Sélection + renommage
    df = df.select(list(RENAME_MAP.keys())).rename(RENAME_MAP)

    # 3. Timestamps → UTC
    #    Le CSV est en heure locale Zurich (CET/CEST).
    #    ambiguous="latest"  → au retour d'heure (2h existe 2×), on prend l'heure d'hiver
    #    non_existent="null" → au passage été (2h→3h sautée), on met null
    ts_dtype = df[COL_TIMESTAMP].dtype
    if ts_dtype == pl.Datetime:
        df = df.with_columns(
            pl.col(COL_TIMESTAMP)
              .dt.replace_time_zone("Europe/Zurich", ambiguous="latest", non_existent="null")
              .dt.convert_time_zone("UTC")
        )
    elif hasattr(ts_dtype, "time_zone") and ts_dtype.time_zone != "UTC":
        df = df.with_columns(pl.col(COL_TIMESTAMP).dt.convert_time_zone("UTC"))

    # 4. Tri
    df = df.sort(COL_TIMESTAMP)

   

    log.info(f"  → {df.height} lignes | {df[COL_TIMESTAMP].min()} → {df[COL_TIMESTAMP].max()}")
    return df


def build_continuous_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    Crée une grille 15 min complète entre le premier et le dernier timestamp,
    puis y joint les données. Les pas manquants deviennent des lignes null.

    Indispensable : sans index continu, les lags (J-1, J-7…) pointent
    vers les mauvais timestamps → erreurs silencieuses dans les features.
    """
    grid = pl.DataFrame({
        COL_TIMESTAMP: pl.datetime_range(
            start=df[COL_TIMESTAMP].min(),
            end=df[COL_TIMESTAMP].max(),
            interval=FREQ,
            time_zone=TIMEZONE,
            eager=True,
        )
    })

    df_full = grid.join(df, on=COL_TIMESTAMP, how="left")

    n_missing = df_full[COL_LOAD].null_count()
    log.info(f"  → Grille : {df_full.height} pas | trous : {n_missing} ({100*n_missing/df_full.height:.2f}%)")
    return df_full


def load_all_oiken(output_name: str = "oiken_raw.parquet") -> pl.DataFrame:
    """
    Pipeline complet : charge tous les CSV → concatène → déduplique →
    index continu → sauvegarde Parquet.
    """
    csv_files = sorted(RAW_OIKEN_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV dans {RAW_OIKEN_DIR}")

    log.info(f"Fichiers : {[f.name for f in csv_files]}")

    # Chargement + concaténation
    df = pl.concat([load_oiken_csv(f) for f in csv_files]).sort(COL_TIMESTAMP)

    # Dédoublonnage (si chevauchement entre fichiers)
    n_before = df.height
    df = df.unique(subset=[COL_TIMESTAMP], keep="last", maintain_order=True)
    if (n_dupes := n_before - df.height) > 0:
        log.warning(f"  → {n_dupes} doublons supprimés")

    # Index continu + sauvegarde
    df = build_continuous_index(df)
    output_path = PROCESSED_DIR / output_name
    df.write_parquet(output_path)
    log.info(f"  → Sauvegardé : {output_path} — {df.shape}")
    return df


def load_oiken_parquet(filename: str = "oiken_raw.parquet") -> pl.DataFrame:
    """Recharge depuis le Parquet (évite de relire le CSV)."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable — lancer load_all_oiken() d'abord")
    df = pl.read_parquet(path)
    log.info(f"Oiken chargé — {df.shape}")
    return df


# ── Point d'entrée : python load_oiken.py ────────────────────────────────────
if __name__ == "__main__":
    df = load_all_oiken()

    print(f"\n── Dimensions : {df.shape}")
    print(f"   Période    : {df[COL_TIMESTAMP].min()} → {df[COL_TIMESTAMP].max()}")

    print("\n── Nulls par colonne :")
    for col in df.columns:
        n = df[col].null_count()
        print(f"  {'✓' if n == 0 else '!'} {col:30s} : {n:5d} ({100*n/df.height:.2f}%)")

    print("\n── Statistiques :")
    print(df.describe())