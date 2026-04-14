import polars as pl
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data")
test = pl.read_parquet(BASE / "data" / "features" / "test_da_v2.parquet")

print("=== 5 premières lignes du test — rolling features ===")
print(test.select([
    "timestamp", "load_lag_1d", "load_lag_7d",
    "rolling_mean_24h", "rolling_mean_7d"
]).head(10))

print("\n=== NaN dans les rolling au début du test ===")
print(f"rolling_mean_7d nulls : {test['rolling_mean_7d'].null_count()}")
print(f"rolling_mean_24h nulls: {test['rolling_mean_24h'].null_count()}")