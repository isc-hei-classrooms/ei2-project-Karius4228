"""
normalize_da.py — Normalisation Day-Ahead (v2)
Auteur : Marius Fabbri
"""

import polars as pl
import joblib
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config import FEATURES_DIR, SCALERS_DIR

# ─── Colonnes à scaler (kWh + météo + NWP — pas encore normalisées) ───────────
# Exclues implicitement : load/lags charge (z-score), sin/cos ([-1,1]),
# booléens, timestamp, target, pred_wind/std (exclus des features)
COLS_TO_SCALE = [
    # PV (kWh, distribution asymétrique)
    "pv_local_lag_1d", "pv_local_lag_7d",
    "pv_remote_lag_1d", "pv_remote_lag_7d",
    # Météo lag J-1
    "temperature_c_lag_1d", "radiation_wm2_lag_1d",
    "humidity_pct_lag_1d",
    # NWP cibles (shift -96, colonnes _target)
    "pred_temperature_ctrl_target", "pred_radiation_ctrl_target",
    "pred_humidity_ctrl_target", "pred_sunshine_ctrl_target",
    # Interactions NWP
    "nwp_temp_squared", "nwp_temp_x_rad",
]
# precipitation_mm_lag_1d et pred_precipitation_ctrl_target exclus :
# IQR ≈ 0 (70-90% de zéros) → division par ~0 avec RobustScaler


def run_normalization_da():
    df_train = pl.read_parquet(FEATURES_DIR / "train_da_v2.parquet")
    df_test  = pl.read_parquet(FEATURES_DIR / "test_da_v2.parquet")

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
    df_train.write_parquet(FEATURES_DIR / "train_da_v2_norm.parquet")
    df_test.write_parquet(FEATURES_DIR  / "test_da_v2_norm.parquet")
    joblib.dump(scaler,  SCALERS_DIR / "scaler_da.joblib")
    joblib.dump(medians, SCALERS_DIR / "medians_da.joblib")

    print(f"✓ DA — Train : {df_train.shape} | Test : {df_test.shape}")
    return df_train, df_test


if __name__ == "__main__":
    run_normalization_da()