"""
config.py — Configuration centrale du projet Energy Informatics 2
Auteur : Marius Fabbri
Date   : 2026
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS DU PROJET
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR      = Path(__file__).resolve().parent
RAW_DIR       = ROOT_DIR / "data" / "raw"
RAW_OIKEN_DIR = RAW_DIR  / "oiken"          # ← maintenant après RAW_DIR
RAW_METEO_DIR = RAW_DIR  / "meteo"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR  = ROOT_DIR / "data" / "features"
SCALERS_DIR   = ROOT_DIR / "scalers"
MODELS_DIR    = ROOT_DIR / "models_saved"

for _dir in [RAW_METEO_DIR, PROCESSED_DIR, FEATURES_DIR, SCALERS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES TEMPORELS
# ─────────────────────────────────────────────────────────────────────────────

TIMEZONE   = "UTC"
FREQ       = "15m"
FREQ_PD    = "15min"

HORIZON_DA = 96    # Day-Ahead  : 24h × 4 = 96 pas de 15min
HORIZON_ID = 12    # Intra-Day  : 3h  × 4 = 12 pas
HORIZON_IQ = 4     # Intra-QH   : 1h  × 4 = 4  pas

# ─────────────────────────────────────────────────────────────────────────────
# COLONNES BRUTES CSV OIKEN
# ─────────────────────────────────────────────────────────────────────────────

OIKEN_COL_TIMESTAMP     = "timestamp"
OIKEN_COL_LOAD          = "standardised load [-]"
OIKEN_COL_FORECAST_LOAD = "standardised forecast load [-]"
OIKEN_COL_PV_CENTRAL    = "central valais solar production [kWh]"
OIKEN_COL_PV_SION       = "sion area solar production [kWh]"
OIKEN_COL_PV_SIERRE     = "sierre area production [kWh]"
OIKEN_COL_PV_REMOTE     = "remote solar production [kWh]"

# ─────────────────────────────────────────────────────────────────────────────
# NOMS STANDARDISÉS — convention interne au projet
# ─────────────────────────────────────────────────────────────────────────────

COL_TIMESTAMP       = "timestamp"

# Charge Oiken
COL_LOAD            = "load"              # variable cible Y (déjà normalisée)
COL_FORECAST_LOAD   = "forecast_load"     # prévision Oiken (feature dans X)

# Production PV par zone
COL_PV_CENTRAL      = "pv_central_kwh"
COL_PV_SION         = "pv_sion_kwh"      # exclue après nettoyage (capteur dégradé)
COL_PV_SIERRE       = "pv_sierre_kwh"
COL_PV_REMOTE       = "pv_remote_kwh"
COL_PV_TOTAL        = "pv_total_kwh"     # somme des 3 zones fiables
COL_NET_LOAD        = "net_load"

# ── Mesures météo réelles (disponibles en J-1 comme lags) ────────────────────
COL_TEMP            = "temperature_c"      # T_2M
COL_GLOB            = "radiation_wm2"      # GLOB
COL_PRECIP          = "precipitation_mm"   # TOT_PREC
COL_HUMIDITY        = "humidity_pct"       # RELHUM_2M
COL_SUNSHINE        = "sunshine_min"       # DURSUN
COL_WIND_SPEED      = "wind_speed_ms"      # FF_10M

# ── Prévisions NWP (disponibles avant gate closure J+1) ──────────────────────
# _ctrl = valeur centrale, _std = incertitude, _q10/_q90 = fourchette
COL_PRED_TEMP_CTRL  = "pred_temperature_ctrl"
COL_PRED_TEMP_STD   = "pred_temperature_std"
COL_PRED_GLOB_CTRL  = "pred_radiation_ctrl"
COL_PRED_GLOB_STD   = "pred_radiation_std"
COL_PRED_PREC_CTRL  = "pred_precipitation_ctrl"
COL_PRED_WIND_CTRL  = "pred_wind_ctrl"
COL_PRED_WIND_STD   = "pred_wind_std"
COL_PRED_SUN_CTRL   = "pred_sunshine_ctrl"
COL_PRED_HUM_CTRL   = "pred_humidity_ctrl"

# ─────────────────────────────────────────────────────────────────────────────
# MAPPING INFLUXDB — measurements réels
# clé = nom colonne projet, valeur = nom exact du measurement dans InfluxDB
# ─────────────────────────────────────────────────────────────────────────────

INFLUX_MEASUREMENTS_REAL = {
    COL_TEMP      : "Air temperature 2m above ground (current value)",
    COL_GLOB      : "Global radiation (ten minutes mean)",
    COL_PRECIP    : "Precipitation (ten minutes total)",
    COL_HUMIDITY  : "Relative air humidity 2m above ground (current value)",
    COL_SUNSHINE  : "Sunshine duration (ten minutes total)",
    COL_WIND_SPEED: "Wind speed scalar (ten minutes mean)",
}

# ── Mapping prévisions NWP ────────────────────────────────────────────────────
# Chaque variable NWP a 4 measurements : ctrl, q10, q90, stde
# On récupère ctrl (valeur centrale) et stde (incertitude)
INFLUX_MEASUREMENTS_PRED = {
    COL_PRED_TEMP_CTRL : "PRED_T_2M_ctrl",
    COL_PRED_TEMP_STD  : "PRED_T_2M_stde",
    COL_PRED_GLOB_CTRL : "PRED_GLOB_ctrl",
    COL_PRED_GLOB_STD  : "PRED_GLOB_stde",
    COL_PRED_PREC_CTRL : "PRED_TOT_PREC_ctrl",
    COL_PRED_WIND_CTRL : "PRED_FF_10M_ctrl",
    COL_PRED_WIND_STD  : "PRED_FF_10M_stde",
    COL_PRED_SUN_CTRL  : "PRED_DURSUN_ctrl",
    COL_PRED_HUM_CTRL  : "PRED_RELHUM_2M_ctrl",
}

# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIALS INFLUXDB
# ─────────────────────────────────────────────────────────────────────────────

INFLUX_URL    = "https://timeseries.hevs.ch"
INFLUX_ORG    = "HESSOVS"
INFLUX_BUCKET = "MeteoSuisse"
INFLUX_TOKEN  = "ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0YmVrk7hZGPpvb_5aaA-ZxhIw=="

METEO_SITES        = ["Sion", "Simplon-Dorf"]
METEO_SITE_PRIMARY = "Sion"

# ─────────────────────────────────────────────────────────────────────────────
# NETTOYAGE & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

MAX_INTERP_STEPS = 4      # max 4 pas consécutifs interpolés = 1 heure
IQR_FACTOR       = 5.0    # seuil outlier : médiane ± 5 × IQR
TRAIN_RATIO      = 0.70   # 70% train, 30% test — split chronologique
METRICS          = ["MAE", "RMSE", "MAPE"]



