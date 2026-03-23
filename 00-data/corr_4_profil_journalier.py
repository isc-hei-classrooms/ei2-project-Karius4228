"""
corr_4_profil_journalier.py
Profil moyen de la charge par heure de la journée.
Hypothèse : deux pics (matin ~8h, soir ~19h), creux nocturne
→ justifie l'encodage cyclique heure_sin/cos comme feature
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

df = pl.read_parquet(BASE / "oiken_clean.parquet").select(["timestamp", "load"])

# Heure locale approximative (UTC+1)
df = df.with_columns(
    ((pl.col("timestamp").dt.hour() + 1) % 24).alias("heure_locale")
)

# Moyenne et écart-type par heure
profil = df.group_by("heure_locale").agg([
    pl.col("load").mean().alias("mean"),
    pl.col("load").std().alias("std"),
]).sort("heure_locale")

heures = profil["heure_locale"].to_numpy()
mean   = profil["mean"].to_numpy()
std    = profil["std"].to_numpy()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(heures, mean, "o-", color="#2980b9", linewidth=2.5, markersize=6, label="Charge moyenne")
ax.fill_between(heures, mean - std, mean + std, alpha=0.2, color="#2980b9", label="±1 écart-type")

# Annoter les pics
idx_max1 = np.argmax(mean[:12])
idx_max2 = np.argmax(mean[12:]) + 12
for idx in [idx_max1, idx_max2]:
    ax.annotate(f"{mean[idx]:.2f}", xy=(heures[idx], mean[idx]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=9, color="#e74c3c", fontweight="bold")

ax.set_xlabel("Heure locale (approximative)", fontsize=12)
ax.set_ylabel("Charge normalisée [-]", fontsize=12)
ax.set_title("Profil journalier moyen de la charge\n"
             "→ justifie l'encodage cyclique hour_sin/cos",
             fontsize=13, fontweight="bold")
ax.set_xticks(range(0, 24))
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\viz\corr_4_profil_journalier.png", dpi=150)
plt.show()
