"""
corr_9_nwp_vs_reel.py
Comparaison prévisions NWP vs mesures réelles.
Vérifie la qualité des NWP : si les prévisions sont bonnes,
elles sont fiables comme features Day-Ahead.
  - Fig 1 : pred_temperature_ctrl vs temperature_c (scatter + diagonale)
  - Fig 2 : pred_radiation_ctrl vs radiation_wm2 (scatter + diagonale)
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

real = pl.read_parquet(BASE / "meteo_real_clean.parquet").select([
    "timestamp", "temperature_c", "radiation_wm2",
])
pred = pl.read_parquet(BASE / "meteo_pred_clean.parquet").select([
    "timestamp", "pred_temperature_ctrl", "pred_radiation_ctrl",
])

df = real.join(pred, on="timestamp").drop_nulls()

# ── Fig 1 : température NWP vs réelle ────────────────────────────────────────
t_real = df["temperature_c"].to_numpy()
t_pred = df["pred_temperature_ctrl"].to_numpy()
r1     = np.corrcoef(t_real, t_pred)[0, 1]
mae1   = np.mean(np.abs(t_real - t_pred))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(t_real, t_pred, s=1, alpha=0.1, color="#2980b9")
lims = [min(t_real.min(), t_pred.min()), max(t_real.max(), t_pred.max())]
ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="Diagonale parfaite")
ax.set_xlabel("Température mesurée (°C)", fontsize=12)
ax.set_ylabel("Température NWP (°C)", fontsize=12)
ax.set_title(f"NWP vs Réel — Température   (r = {r1:.3f}, MAE = {mae1:.2f}°C)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_aspect("equal")
ax.text(0.02, 0.95, "Points proches de la diagonale\n→ prévision fiable comme feature",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_9a_nwp_vs_reel_temp.png", dpi=150)
plt.show()

# ── Fig 2 : radiation NWP vs réelle ──────────────────────────────────────────
r_real = df["radiation_wm2"].to_numpy()
r_pred = df["pred_radiation_ctrl"].to_numpy()
r2     = np.corrcoef(r_real, r_pred)[0, 1]
mae2   = np.mean(np.abs(r_real - r_pred))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(r_real, r_pred, s=1, alpha=0.1, color="#e67e22")
lims = [0, max(r_real.max(), r_pred.max())]
ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="Diagonale parfaite")
ax.set_xlabel("Radiation mesurée (W/m²)", fontsize=12)
ax.set_ylabel("Radiation NWP (W/m²)", fontsize=12)
ax.set_title(f"NWP vs Réel — Radiation   (r = {r2:.3f}, MAE = {mae2:.1f} W/m²)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_aspect("equal")
ax.text(0.02, 0.95, "Dispersion plus forte en radiation\n→ nuages difficiles à prévoir",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_9b_nwp_vs_reel_rad.png", dpi=150)
plt.show()

