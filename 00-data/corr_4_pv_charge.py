"""
corr_4_pv_charge.py
Lien entre production PV et charge du réseau.
  - Fig 1 : scatter PV total vs charge (corrélation directe)
  - Fig 2 : profil journalier moyen PV et charge superposés
    → montre le décalage : PV max à midi, charge max matin/soir
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

df = pl.read_parquet(BASE / "oiken_clean.parquet").select([
    "timestamp", "load", "pv_total_kwh",
]).drop_nulls()

pv   = df["pv_total_kwh"].to_numpy()
load = df["load"].to_numpy()
r    = np.corrcoef(pv, load)[0, 1]

# ── Fig 1 : scatter PV total → charge ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(pv, load, s=1, alpha=0.15, color="#e67e22")
ax.set_xlabel("Production PV totale (kWh)", fontsize=12)
ax.set_ylabel("Charge normalisée [-]", fontsize=12)
ax.set_title(f"PV total → Charge   (r = {r:.3f}, n = {len(pv):,})",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.text(0.02, 0.95, "Corrélation négative attendue :\nplus de PV → moins de charge nette",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_4a_pv_charge.png", dpi=150)
plt.show()

# ── Fig 2 : profil journalier moyen PV vs charge ─────────────────────────────
# Heure fractionnaire (ex: 14h30 = 14.5) pour un axe X continu
df2 = df.with_columns(
    (pl.col("timestamp").dt.hour() + pl.col("timestamp").dt.minute() / 60)
    .alias("hour_frac")
)

hourly = (
    df2.group_by("hour_frac").agg([
        pl.col("load").mean().alias("load_mean"),
        pl.col("pv_total_kwh").mean().alias("pv_mean"),
    ]).sort("hour_frac")
)

h    = hourly["hour_frac"].to_numpy()
l_m  = hourly["load_mean"].to_numpy()
pv_m = hourly["pv_mean"].to_numpy()

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(h, l_m, color="#3498db", linewidth=2, label="Charge moyenne")
ax1.set_xlabel("Heure UTC", fontsize=12)
ax1.set_ylabel("Charge normalisée [-]", fontsize=12, color="#3498db")
ax1.tick_params(axis="y", labelcolor="#3498db")

ax2 = ax1.twinx()
ax2.plot(h, pv_m, color="#e67e22", linewidth=2, label="PV moyen")
ax2.set_ylabel("PV total moyen (kWh)", fontsize=12, color="#e67e22")
ax2.tick_params(axis="y", labelcolor="#e67e22")

ax1.set_title("Profil journalier moyen — Charge vs PV", fontsize=13, fontweight="bold")
ax1.grid(alpha=0.3)
ax1.set_xticks(range(0, 25, 3))

# Légende combinée
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

ax1.text(0.98, 0.95, "PV max à midi\nCharge max matin/soir\n→ décalage offre/demande",
         transform=ax1.transAxes, fontsize=9, va="top", ha="right",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_4b_profil_pv_charge.png", dpi=150)
plt.show()

