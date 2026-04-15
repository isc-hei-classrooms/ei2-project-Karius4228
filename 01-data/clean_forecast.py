"""
clean_forecast.py — Détection et exclusion des prévisions Oiken corrompues
───────────────────────────────────────────────────────────────────────────
Détecte les périodes où forecast_load est figé (même valeur pendant >8h)
et retourne les dates à exclure des métriques.

Auteur : Marius Fabbri
"""

import polars as pl
import numpy as np
from datetime import timedelta
import logging

log = logging.getLogger(__name__)


def detect_frozen_forecast(df: pl.DataFrame, col: str = "forecast_load",
                           min_run: int = 32) -> set:
    """
    Détecte les jours où forecast_load est figé (même valeur > min_run pas = 8h).
    Retourne un set de dates à exclure (inclut le jour précédent pour forecast_load_target).
    """
    fc = df[col].to_numpy()
    ts = df["timestamp"].to_list()
    suspect = set()

    run_start, run_val = 0, fc[0]
    for i in range(1, len(fc)):
        if fc[i] != run_val:
            if (i - run_start) >= min_run:
                for j in range(run_start, i):
                    suspect.add(ts[j].date())
            run_start, run_val = i, fc[i]
    if (len(fc) - run_start) >= min_run:
        for j in range(run_start, len(fc)):
            suspect.add(ts[j].date())

    # Étendre : forecast_load_target[t] = forecast_load[t+24h]
    # donc si forecast_load est figé le jour D, forecast_load_target est affecté le jour D-1
    extended = set()
    for d in suspect:
        extended.add(d)
        extended.add(d - timedelta(days=1))

    if extended:
        log.warning(f"  Détecté {len(suspect)} jours avec {col} figé → {len(extended)} jours exclus (avec shift)")

    return extended


def filter_frozen_days(df: pl.DataFrame, suspect_dates: set) -> pl.DataFrame:
    """Exclut les lignes dont la date est dans suspect_dates."""
    if not suspect_dates:
        return df
    mask = ~pl.Series([ts.date() in suspect_dates for ts in df["timestamp"].to_list()])
    n_before = df.height
    df_clean = df.filter(mask)
    log.info(f"  Nettoyage : {n_before - df_clean.height} pas exclus ({n_before} → {df_clean.height})")
    return df_clean
