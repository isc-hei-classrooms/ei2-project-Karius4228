"""
corr_7_nwp_predictions.py
Visualisation des prévisions NWP utilisées comme features Day-Ahead.
  - Fig 1 : pred_temperature_ctrl vs charge (même logique que corr_1 mais avec NWP)
  - Fig 2 : pred_radiation_ctrl vs PV total (même logique que corr_2 mais avec NWP)
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

oiken = pl.read_parquet(BASE / "oiken_clean.parquet").select(["timestamp", "load", "pv_total_kwh"])
nwp   = pl.read_parquet(BASE / "meteo_pred_clean.parquet").select([
    "timestamp", "pred_temperature_ctrl", "pred_radiation_ctrl",
])

df = oiken.join(nwp, on="timestamp").drop_nulls()

# ── Fig 1 : pred_temperature → charge ────────────────────────────────────────
temp = df["pred_temperature_ctrl"].to_numpy()
load = df["load"].to_numpy()
r1   = np.corrcoef(temp, load)[0, 1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(temp, load, s=1, alpha=0.15, color="#e74c3c")
ax.set_xlabel("Prévision température NWP (°C)", fontsize=12)
ax.set_ylabel("Charge normalisée [-]", fontsize=12)
ax.set_title(f"NWP Température → Charge   (r = {r1:.3f}, n = {len(temp):,})",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.text(0.02, 0.95, "À comparer avec corr_1 (mesure réelle)\n→ NWP doit capturer la même relation en U",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_7a_nwp_temp_charge.png", dpi=150)
plt.show()

# ── Fig 2 : pred_radiation → PV total ────────────────────────────────────────
rad = df["pred_radiation_ctrl"].to_numpy()
pv  = df["pv_total_kwh"].to_numpy()
r2  = np.corrcoef(rad, pv)[0, 1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(rad, pv, s=1, alpha=0.15, color="#f39c12")
ax.set_xlabel("Prévision radiation NWP (W/m²)", fontsize=12)
ax.set_ylabel("Production PV totale (kWh)", fontsize=12)
ax.set_title(f"NWP Radiation → PV total   (r = {r2:.3f}, n = {len(rad):,})",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.text(0.02, 0.95, "Relation linéaire attendue :\nplus de radiation prévue → plus de PV",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_7b_nwp_rad_pv.png", dpi=150)
plt.show()

