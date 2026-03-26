"""
corr_8_meteo_complementaire.py
Corrélations des variables météo réelles non encore analysées :
  - Fig 1 : humidité vs charge
  - Fig 2 : ensoleillement + vent vs charge (double subplot)
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

oiken = pl.read_parquet(BASE / "oiken_clean.parquet").select(["timestamp", "load"])
meteo = pl.read_parquet(BASE / "meteo_real_clean.parquet").select([
    "timestamp", "humidity_pct", "sunshine_min", "wind_speed_ms",
])

df = oiken.join(meteo, on="timestamp").drop_nulls()

# ── Fig 1 : humidité → charge ────────────────────────────────────────────────
hum  = df["humidity_pct"].to_numpy()
load = df["load"].to_numpy()
r1   = np.corrcoef(hum, load)[0, 1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(hum, load, s=1, alpha=0.15, color="#1abc9c")
ax.set_xlabel("Humidité relative (%)", fontsize=12)
ax.set_ylabel("Charge normalisée [-]", fontsize=12)
ax.set_title(f"Humidité → Charge   (r = {r1:.3f}, n = {len(hum):,})",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.text(0.02, 0.95, "Humidité élevée → temps couvert/froid\n→ corrélation positive attendue",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_8a_humidity_charge.png", dpi=150)
plt.show()

# ── Fig 2 : ensoleillement + vent → charge ───────────────────────────────────
sun  = df["sunshine_min"].to_numpy()
wind = df["wind_speed_ms"].to_numpy()
r_sun  = np.corrcoef(sun, load)[0, 1]
r_wind = np.corrcoef(wind, load)[0, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(sun, load, s=1, alpha=0.15, color="#f1c40f")
ax1.set_xlabel("Ensoleillement (min / 15min)", fontsize=12)
ax1.set_ylabel("Charge normalisée [-]", fontsize=12)
ax1.set_title(f"Ensoleillement → Charge   (r = {r_sun:.3f})", fontsize=13, fontweight="bold")
ax1.grid(alpha=0.3)

ax2.scatter(wind, load, s=1, alpha=0.15, color="#9b59b6")
ax2.set_xlabel("Vitesse du vent (m/s)", fontsize=12)
ax2.set_ylabel("Charge normalisée [-]", fontsize=12)
ax2.set_title(f"Vent → Charge   (r = {r_wind:.3f})", fontsize=13, fontweight="bold")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_8b_sun_wind_charge.png", dpi=150)
plt.show()

