"""
normalize_intraday.py — Normalisation Intraday (v2)
Auteur : Marius Fabbri
"""

import polars as pl
import joblib
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import FEATURES_DIR, SCALERS_DIR

# ─── Colonnes à scaler ────────────────────────────────────────────────────────
# NWP gardées BRUTES (sans shift) — le shift(-h) est fait à l'entraînement
# via build_nwp_for_horizon(). On les scale ici sur leurs valeurs brutes.
COLS_TO_SCALE = [
    # PV (kWh)
    "pv_local_lag_1", "pv_local_lag_4",
    "pv_local_lag_96", "pv_local_lag_672",
    "pv_local_rolling_mean_1h",
    "pv_remote_lag_96", "pv_remote_lag_672",
    # Météo récente (t-4)
    "temperature_c_recent", "radiation_wm2_recent",
    "humidity_pct_recent",
    # NWP brutes
    "pred_temperature_ctrl", "pred_radiation_ctrl",
    "pred_humidity_ctrl", "pred_sunshine_ctrl",
    # Interactions sur météo récente
    "temp_squared", "temp_x_rad",
]
# precipitation_mm_recent et pred_precipitation_ctrl exclus : IQR ≈ 0


def run_normalization_intraday():
    df_train = pl.read_parquet(FEATURES_DIR / "train_intraday_v2.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_intraday_v2.parquet")

    # Imputation médiane sur train uniquement (anti-leakage)
    medians = {}
    train_exprs, test_exprs = [], []
    for col in COLS_TO_SCALE:
        if col not in df_train.columns:
            continue
        if df_train[col].null_count() == 0 and df_test[col].null_count() == 0:
            continue
        m = float(df_train[col].drop_nulls().median())
        medians[col] = m
        train_exprs.append(pl.col(col).fill_null(m).alias(col))
        if col in df_test.columns:
            test_exprs.append(pl.col(col).fill_null(m).alias(col))

    if train_exprs:
        df_train = df_train.with_columns(train_exprs)
    if test_exprs:
        df_test  = df_test.with_columns(test_exprs)

    # Scaling — fit sur train, transform train + test
    cols = [c for c in COLS_TO_SCALE if c in df_train.columns]
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(df_train.select(cols).to_numpy())
    X_test_scaled  = scaler.transform(df_test.select(cols).to_numpy())

    df_train = df_train.drop(cols).hstack(
        pl.DataFrame(X_train_scaled, schema={c: pl.Float64 for c in cols})
    )
    df_test = df_test.drop(cols).hstack(
        pl.DataFrame(X_test_scaled, schema={c: pl.Float64 for c in cols})
    )

    # Sauvegarde
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    df_train.write_parquet(FEATURES_DIR / "train_intraday_v2_norm.parquet")
    df_test.write_parquet(FEATURES_DIR  / "test_intraday_v2_norm.parquet")
    joblib.dump(scaler,  SCALERS_DIR / "scaler_intraday.joblib")
    joblib.dump(medians, SCALERS_DIR / "medians_intraday.joblib")

    print(f"✓ Intraday — Train : {df_train.shape} | Test : {df_test.shape}")
    return df_train, df_test


if __name__ == "__main__":
    run_normalization_intraday()