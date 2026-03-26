"""
corr_3_precipitations_pv.py
Corrélation inverse entre précipitations et production PV.
Hypothèse : quand il pleut → nuages → moins de rayonnement → moins de PV
→ justifie l'inclusion de precipitation_mm comme feature
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

oiken = pl.read_parquet(BASE / "oiken_clean.parquet").select(["timestamp", "pv_total_kwh"])
meteo = pl.read_parquet(BASE / "meteo_real_clean.parquet").select(["timestamp", "precipitation_mm"])

# Heures diurnes uniquement
df = oiken.join(meteo, on="timestamp").filter(
    pl.col("timestamp").dt.hour().is_between(7, 19)
).drop_nulls()

# Grouper par niveau de précipitation pour un boxplot lisible
precip = df["precipitation_mm"].to_numpy()
pv     = df["pv_total_kwh"].to_numpy()
r      = np.corrcoef(precip, pv)[0, 1]

# Catégories de pluie
labels = ["Sec\n(0 mm)", "Pluie légère\n(0–1 mm)", "Pluie mod.\n(1–5 mm)", "Forte pluie\n(>5 mm)"]
groups = [
    pv[precip == 0],
    pv[(precip > 0)  & (precip <= 1)],
    pv[(precip > 1)  & (precip <= 5)],
    pv[precip > 5],
]
sizes = [len(g) for g in groups]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(groups, labels=labels, patch_artist=True, showfliers=False)

colors = ["#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Annoter le nombre d'observations
for i, (label, n) in enumerate(zip(labels, sizes)):
    ax.text(i + 1, ax.get_ylim()[0] - 300, f"n={n:,}", ha="center", fontsize=8, color="gray")

ax.set_ylabel("Production PV totale (kWh)", fontsize=12)
ax.set_title(f"Précipitations → Production PV   (r = {r:.3f})\n"
             f"Plus il pleut, moins le PV produit",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\viz\corr_3_precip_pv.png", dpi=150)
plt.show()
