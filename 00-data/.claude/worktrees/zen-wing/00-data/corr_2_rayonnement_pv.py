"""
corr_2_rayonnement_pv.py
Corrélation entre rayonnement solaire et production PV totale.
Hypothèse : relation quasi-linéaire directe (physique photovoltaïque)
→ justifie l'inclusion de radiation_wm2 et des prévisions NWP GLOB
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

oiken = pl.read_parquet(BASE / "oiken_clean.parquet").select(["timestamp", "pv_total_kwh"])
meteo = pl.read_parquet(BASE / "meteo_real_clean.parquet").select(["timestamp", "radiation_wm2"])

# Heures diurnes uniquement (rayonnement > 0)
df = oiken.join(meteo, on="timestamp").filter(
    pl.col("radiation_wm2") > 10
).drop_nulls()

rad = df["radiation_wm2"].to_numpy()
pv  = df["pv_total_kwh"].to_numpy()
r   = np.corrcoef(rad, pv)[0, 1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(rad, pv, s=1, alpha=0.15, color="#f39c12")

# Droite de régression
m, b = np.polyfit(rad, pv, 1)
x_line = np.linspace(rad.min(), rad.max(), 100)
ax.plot(x_line, m * x_line + b, "r-", linewidth=2,
        label=f"y = {m:.1f}x + {b:.0f}")

ax.set_xlabel("Rayonnement solaire (W/m²)", fontsize=12)
ax.set_ylabel("Production PV totale (kWh)", fontsize=12)
ax.set_title(f"Rayonnement → Production PV   (r = {r:.3f})",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\viz\corr_2_radiation_pv.png", dpi=150)
plt.show()
