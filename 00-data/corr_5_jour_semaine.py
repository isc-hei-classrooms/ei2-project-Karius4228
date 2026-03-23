"""
corr_5_jour_semaine.py
Distribution de la charge par jour de la semaine.
Hypothèse : semaine > weekend (activité industrielle/commerciale)
→ justifie is_weekend et weekday_sin/cos comme features
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

df = pl.read_parquet(BASE / "oiken_clean.parquet").select(["timestamp", "load"])
df = df.with_columns(pl.col("timestamp").dt.weekday().alias("weekday"))

jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
groups = [
    df.filter(pl.col("weekday") == i)["load"].drop_nulls().to_numpy()
    for i in range(7)
]

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(groups, labels=jours, patch_artist=True, showfliers=False)

# Semaine en bleu, weekend en orange
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor("#3498db" if i < 5 else "#e67e22")
    patch.set_alpha(0.75)

# Ligne de moyenne globale
mean_global = df["load"].mean()
ax.axhline(mean_global, color="gray", linestyle="--", linewidth=1,
           label=f"Moyenne globale ({mean_global:.3f})")

# Séparer visuellement semaine / weekend
ax.axvline(5.5, color="black", linestyle=":", linewidth=1.5, alpha=0.5)
ax.text(5.7, ax.get_ylim()[1] * 0.95, "Weekend", fontsize=9,
        color="#e67e22", fontweight="bold")

ax.set_ylabel("Charge normalisée [-]", fontsize=12)
ax.set_title("Charge par jour de la semaine\n"
             "→ justifie is_weekend et weekday_sin/cos",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\viz\corr_5_jour_semaine.png", dpi=150)
plt.show()
