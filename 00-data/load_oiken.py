"""
src/acquisition/load_oiken.py
──────────────────────────────
Ingestion des données de charge du sous-groupe bilan Oiken.

Structure réelle du CSV oiken-data.csv (105 120 lignes = ~3 ans à 15min) :
  timestamp                              → index temporel
  standardised load [-]                  → charge normalisée (variable cible)
  standardised forecast load [-]         → prévision de charge Oiken (feature)
  central valais solar production [kWh]  → PV zone Central Valais
  sion area solar production [kWh]       → PV zone Sion
  sierre area production [kWh]           → PV zone Sierre
  remote solar production [kWh]          → PV zone Remote

Note sur les unités :
  - La charge est DÉJÀ normalisée (sans dimension [-]).
    On la gardera telle quelle comme variable cible Y.
    La normalisation sera donc relative à cette échelle.
  - Les productions PV sont en kWh par quart d'heure.
    On les garde séparées par zone (utile pour le feature engineering)
    ET on calcule un total pour avoir un signal PV global.

Auteur : Marius Fabbri
"""

import polars as pl
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import (
    RAW_OIKEN_DIR, PROCESSED_DIR,
    # Noms bruts du CSV
    OIKEN_COL_TIMESTAMP, OIKEN_COL_LOAD, OIKEN_COL_FORECAST_LOAD,
    OIKEN_COL_PV_CENTRAL, OIKEN_COL_PV_SION,
    OIKEN_COL_PV_SIERRE, OIKEN_COL_PV_REMOTE, OIKEN_COLS_PV,
    # Noms standardisés du projet
    COL_TIMESTAMP, COL_LOAD, COL_FORECAST_LOAD,
    COL_PV_CENTRAL, COL_PV_SION, COL_PV_SIERRE, COL_PV_REMOTE,
    COL_PV_TOTAL, COL_NET_LOAD,
    # Paramètres
    TIMEZONE, FREQ,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT D'UN FICHIER CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_oiken_csv(csv_path: Path | str) -> pl.DataFrame:
    """
    Lit un fichier CSV Oiken et retourne un DataFrame Polars propre.

    Étapes :
      1. Lecture brute (toutes les colonnes, détection auto des dates)
      2. Sélection des 6 colonnes utiles + renommage convention projet
      3. Conversion timestamp en UTC
      4. Tri chronologique
      5. Calcul PV total (somme des 4 zones)
      6. Calcul charge nette (load - pv_total)

    Paramètres
    ----------
    csv_path : chemin vers le fichier CSV Oiken

    Retourne
    --------
    pl.DataFrame avec toutes les colonnes standardisées
    """
    csv_path = Path(csv_path)
    log.info(f"Lecture CSV : {csv_path.name}")

    # ── 1. Lecture brute ──────────────────────────────────────────────────────
    # try_parse_dates=True : Polars détecte et parse automatiquement
    # la colonne timestamp sans qu'on ait à spécifier le format
    # null_values : "#N/A" est le format de valeur nulle dans ce CSV
    # ignore_errors=True : lignes malformées → null plutôt que crash
    df = pl.read_csv(
        csv_path,
        try_parse_dates=True,
        null_values=["#N/A", "N/A", "", "null"],
        ignore_errors=True,
    )
    log.info(f"  → {df.height} lignes | colonnes brutes : {df.columns}")

    # ── 2. Sélection et renommage ─────────────────────────────────────────────
    # On sélectionne les 6 colonnes utiles (toutes sauf rien ici)
    # et on les renomme avec des noms courts sans espaces ni caractères spéciaux
    # → plus facile à manipuler dans tout le reste du code
    df = df.select([
        pl.col(OIKEN_COL_TIMESTAMP),
        pl.col(OIKEN_COL_LOAD),
        pl.col(OIKEN_COL_FORECAST_LOAD),
        pl.col(OIKEN_COL_PV_CENTRAL),
        pl.col(OIKEN_COL_PV_SION),
        pl.col(OIKEN_COL_PV_SIERRE),
        pl.col(OIKEN_COL_PV_REMOTE),
    ]).rename({
        OIKEN_COL_TIMESTAMP     : COL_TIMESTAMP,
        OIKEN_COL_LOAD          : COL_LOAD,
        OIKEN_COL_FORECAST_LOAD : COL_FORECAST_LOAD,
        OIKEN_COL_PV_CENTRAL    : COL_PV_CENTRAL,
        OIKEN_COL_PV_SION       : COL_PV_SION,
        OIKEN_COL_PV_SIERRE     : COL_PV_SIERRE,
        OIKEN_COL_PV_REMOTE     : COL_PV_REMOTE,
    })

    # ── 3. Conversion timestamp en UTC ───────────────────────────────────────
    # On inspecte le dtype pour savoir si le timestamp a déjà une TZ ou non
    ts_dtype = df[COL_TIMESTAMP].dtype

    if ts_dtype == pl.Datetime:
        # Timestamp naïf (pas de TZ dans le CSV) :
        # on suppose que les données sont en heure locale de Zurich (CET/CEST)
        # puis on convertit en UTC — ambiguous="latest" gère le retour d'heure
        # (la nuit du passage heure d'été → heure d'hiver où 2h existe deux fois)
        df = df.with_columns(
            pl.col(COL_TIMESTAMP)
                .dt.replace_time_zone(
                    "Europe/Zurich",
                    ambiguous="latest",      # heure d'hiver lors du retour (2h existe 2x)
                    non_existent="null"      # heure inexistante lors du passage été (2h→3h)
                )
                .dt.convert_time_zone("UTC")
        )
    elif hasattr(ts_dtype, "time_zone") and ts_dtype.time_zone != "UTC":
        # Timestamp avec une TZ différente d'UTC → conversion directe
        df = df.with_columns(
            pl.col(COL_TIMESTAMP).dt.convert_time_zone("UTC")
        )
    # Si déjà UTC : rien à faire

    # ── 4. Tri chronologique ──────────────────────────────────────────────────
    df = df.sort(COL_TIMESTAMP)

    # ── 5. PV total = somme des 4 zones ──────────────────────────────────────
    # On additionne les 4 colonnes de production PV pour avoir un signal global.
    # pl.sum_horizontal() fait la somme ligne par ligne sur plusieurs colonnes.
    # Avantage : si une zone est null, la somme vaut null aussi → pas de perte
    # silencieuse d'information (contrairement à fill_null(0) prématuré).
    df = df.with_columns(
        pl.sum_horizontal(
            COL_PV_CENTRAL, COL_PV_SION, COL_PV_SIERRE, COL_PV_REMOTE
        ).alias(COL_PV_TOTAL)
    )

    # ── 6. Charge nette ───────────────────────────────────────────────────────
    # Charge nette = ce que le sous-groupe bilan doit acheter sur le marché
    # = consommation mesurée − production PV totale injectée localement
    #
    # ATTENTION sur les unités :
    #   - load         : normalisé [-], sans unité physique
    #   - pv_total_kwh : en kWh
    # → charge_nette n'est pas directement interprétable physiquement.
    #   Elle sera utile comme feature relative, mais la cible principale
    #   du modèle Day-Ahead sera COL_LOAD (charge normalisée seule).
    df = df.with_columns(
        (pl.col(COL_LOAD) - pl.col(COL_PV_TOTAL)).alias(COL_NET_LOAD)
    )

    # ── Log de contrôle ───────────────────────────────────────────────────────
    log.info(f"  → Période  : {df[COL_TIMESTAMP].min()}  →  {df[COL_TIMESTAMP].max()}")
    log.info(f"  → Nulls charge   : {df[COL_LOAD].null_count()}")
    log.info(f"  → Nulls PV total : {df[COL_PV_TOTAL].null_count()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDEX TEMPOREL CONTINU
# ─────────────────────────────────────────────────────────────────────────────

def build_continuous_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    Construit une grille UTC continue au pas de 15 minutes et
    réindexe le DataFrame dessus via un join left.

    Les pas de temps manquants dans le CSV original apparaissent
    comme des lignes avec null → traités par clean.py ensuite.

    Pourquoi c'est indispensable pour le ML ?
    Les features de lag (ex: valeur d'il y a 24h) sont calculées par
    décalage d'index. Si l'index a des trous, le décalage pointe vers
    la mauvaise heure → erreurs silencieuses dans X et Y.
    """
    t_min = df[COL_TIMESTAMP].min()
    t_max = df[COL_TIMESTAMP].max()

    # Génération de la grille complète
    grid = pl.DataFrame({
        COL_TIMESTAMP: pl.datetime_range(
            start=t_min,
            end=t_max,
            interval=FREQ,        # "15m" défini dans config.py
            time_zone=TIMEZONE,   # "UTC"
            eager=True,
        )
    })

    # Join left : tous les pas de la grille sont conservés
    # Les pas manquants dans df → null sur toutes les colonnes
    df_full = grid.join(df, on=COL_TIMESTAMP, how="left")

    n_missing = df_full[COL_LOAD].null_count()
    pct = 100 * n_missing / df_full.height
    log.info(f"  → Grille continue : {df_full.height} pas | "
             f"trous détectés : {n_missing} ({pct:.2f}%)")

    return df_full


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT COMPLET + SAUVEGARDE PARQUET
# ─────────────────────────────────────────────────────────────────────────────

def load_all_oiken(output_name: str = "oiken_raw.parquet") -> pl.DataFrame:
    """
    Charge et concatène tous les CSV dans RAW_OIKEN_DIR,
    construit l'index continu, et sauvegarde en Parquet.

    Paramètres
    ----------
    output_name : nom du fichier Parquet dans data/processed/

    Retourne
    --------
    pl.DataFrame complet, index UTC 15min continu
    """
    # Recherche de tous les CSV dans le dossier raw/oiken/
    csv_files = sorted(RAW_OIKEN_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Aucun CSV trouvé dans {RAW_OIKEN_DIR}\n"
            "→ Copier oiken-data.csv dans ce dossier."
        )

    log.info(f"Fichiers trouvés : {[f.name for f in csv_files]}")

    # Chargement individuel de chaque fichier puis concaténation verticale
    frames = [load_oiken_csv(f) for f in csv_files]
    df_all = pl.concat(frames).sort(COL_TIMESTAMP)

    # Suppression des doublons temporels (si deux fichiers se chevauchent)
    n_before = df_all.height
    df_all = df_all.unique(subset=[COL_TIMESTAMP], keep="last", maintain_order=True)
    n_dupes = n_before - df_all.height
    if n_dupes > 0:
        log.warning(f"  → {n_dupes} doublons temporels supprimés")

    # Index continu
    df_all = build_continuous_index(df_all)

    # Sauvegarde Parquet : format binaire compressé, beaucoup plus rapide
    # à relire que le CSV pour toutes les étapes suivantes du pipeline
    output_path = PROCESSED_DIR / output_name
    df_all.write_parquet(output_path)
    log.info(f"  → Sauvegardé : {output_path}")
    log.info(f"  → Dimensions finales : {df_all.shape}")

    return df_all


def load_oiken_parquet(filename: str = "oiken_raw.parquet") -> pl.DataFrame:
    """
    Recharge depuis le Parquet local (évite de relire le CSV à chaque fois).
    """
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet non trouvé : {path}\n"
            "→ Lancer d'abord load_all_oiken()."
        )
    df = pl.read_parquet(path)
    log.info(f"Oiken chargé depuis Parquet — shape : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST RAPIDE : python load_oiken.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_all_oiken()

    print("\n── Aperçu des 5 premières lignes ───────────────────────────────")
    print(df.head(5))

    print("\n── Aperçu des 5 dernières lignes ───────────────────────────────")
    print(df.tail(5))

    print(f"\n── Dimensions : {df.shape}")
    print(f"   Période    : {df[COL_TIMESTAMP].min()}  →  {df[COL_TIMESTAMP].max()}")

    print("\n── Nulls par colonne ───────────────────────────────────────────")
    for col in df.columns:
        n = df[col].null_count()
        pct = 100 * n / df.height
        status = "✓" if n == 0 else "!"
        print(f"  [{status}] {col:35s} : {n:5d} nulls ({pct:.2f}%)")

    print("\n── Statistiques descriptives ───────────────────────────────────")
    print(df.describe())