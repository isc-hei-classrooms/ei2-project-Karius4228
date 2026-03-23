import polars as pl
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")
df = pl.read_parquet(BASE / "oiken_clean.parquet")

# Statistiques de la charge brute
print("Statistiques load :")
print(df["load"].describe())

# Comparer load avec pv_total sur quelques lignes en journée
print("\nExemple journée estivale (été, midi) :")
exemple = df.filter(
    (pl.col("timestamp").dt.month() == 7) &
    (pl.col("timestamp").dt.hour() == 11)
).select(["timestamp", "load", "pv_total_kwh"]).head(5)
print(exemple)

# Comparer load avec pv_total sur quelques lignes en hiver
print("\nExemple journée hivernale (hiver, midi) :")
exemple2 = df.filter(
    (pl.col("timestamp").dt.month() == 1) &
    (pl.col("timestamp").dt.hour() == 11)
).select(["timestamp", "load", "pv_total_kwh"]).head(5)
print(exemple2)