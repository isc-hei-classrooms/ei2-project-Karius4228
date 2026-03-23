"""
corr_6_saison_charge.py
Distribution de la charge par saison.
Hypothèse : hiver > été (chauffage électrique en Valais)
→ justifie is_winter/spring/summer/autumn et month_sin/cos
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

df = pl.read_parquet(BASE / "oiken_clean.parquet").select(["timestamp", "load"])
df = df.with_columns(pl.col("timestamp").dt.month().alias("mois"))

# Assigner une saison
df = df.with_columns(
    pl.when((pl.col("mois") == 12) | (pl.col("mois") <= 2)).then(pl.lit("Hiver"))
    .when((pl.col("mois") >= 3) & (pl.col("mois") <= 5)).then(pl.lit("Printemps"))
    .when((pl.col("mois") >= 6) & (pl.col("mois") <= 8)).then(pl.lit("Été"))
    .otherwise(pl.lit("Automne"))
    .alias("saison")
)

saisons = ["Hiver", "Printemps", "Été", "Automne"]
colors  = ["#3498db", "#2ecc71", "#f39c12", "#e67e22"]
groups  = [df.filter(pl.col("saison") == s)["load"].drop_nulls().to_numpy()
           for s in saisons]
means   = [g.mean() for g in groups]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Boxplot
bp = axes[0].boxplot(groups, labels=saisons, patch_artist=True, showfliers=False)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
axes[0].set_ylabel("Charge normalisée [-]")
axes[0].set_title("Distribution par saison", fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)

# Profil mensuel moyen
profil_mens = df.group_by("mois").agg(
    pl.col("load").mean().alias("mean")
).sort("mois")

mois_noms = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
             "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
mois_vals = profil_mens["mois"].to_numpy()
mean_vals = profil_mens["mean"].to_numpy()

season_colors_month = (
    ["#3498db"] * 2 + ["#2ecc71"] * 3 + ["#f39c12"] * 3 + ["#e67e22"] * 3 + ["#3498db"]
)
axes[1].bar(range(12), mean_vals[np.argsort(mois_vals)],
            color=season_colors_month, edgecolor="white", alpha=0.85)
axes[1].set_xticks(range(12))
axes[1].set_xticklabels(mois_noms, fontsize=9)
axes[1].set_ylabel("Charge moyenne normalisée [-]")
axes[1].set_title("Profil mensuel moyen", fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Charge par saison → justifie is_winter/spring/summer/autumn et month_sin/cos",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\viz\corr_6_saison_charge.png", dpi=150)
plt.show()
