"""
corr_1_temperature_charge.py
Corrélation entre température extérieure et charge du réseau.
Hypothèse : forte consommation par temps froid (chauffage) et chaud (climatisation)
→ relation en U (non-linéaire)
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

oiken = pl.read_parquet(BASE / "oiken_clean.parquet").select(["timestamp", "load"])
meteo = pl.read_parquet(BASE / "meteo_real_clean.parquet").select(["timestamp", "temperature_c"])

df = oiken.join(meteo, on="timestamp").drop_nulls()

temp = df["temperature_c"].to_numpy()
load = df["load"].to_numpy()
r    = np.corrcoef(temp, load)[0, 1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(temp, load, s=1, alpha=0.15, color="#3498db")
ax.set_xlabel("Température (°C)", fontsize=12)
ax.set_ylabel("Charge normalisée [-]", fontsize=12)
ax.set_title(f"Température → Charge   (r = {r:.3f}, n = {len(temp):,})",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.text(0.02, 0.95, "Relation en U attendue :\nfroid → chauffage, chaud → climatisation",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\viz\corr_1_temp_charge.png", dpi=150)
plt.show()
