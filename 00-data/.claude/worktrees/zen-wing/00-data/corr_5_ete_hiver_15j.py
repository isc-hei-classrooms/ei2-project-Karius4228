"""
corr_5_ete_hiver_15j.py
Comparaison visuelle été vs hiver sur 15 jours.
Triple subplot : charge, PV total, température.
→ Montre les patterns saisonniers du réseau Oiken.
"""
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\Projet\00-data\data\processed")

oiken = pl.read_parquet(BASE / "oiken_clean.parquet").select(["timestamp", "load", "pv_total_kwh"])
meteo = pl.read_parquet(BASE / "meteo_real_clean.parquet").select(["timestamp", "temperature_c"])
df = oiken.join(meteo, on="timestamp").drop_nulls()

# 15 jours été (juillet 2024) et hiver (janvier 2024)

ete = df.filter(pl.col("timestamp").is_between(
    datetime(2024, 7, 1, tzinfo=timezone.utc),
    datetime(2024, 7, 16, tzinfo=timezone.utc),
))
hiver = df.filter(pl.col("timestamp").is_between(
    datetime(2024, 1, 5, tzinfo=timezone.utc),
    datetime(2024, 1, 20, tzinfo=timezone.utc),
))

def plot_15j(data: pl.DataFrame, saison: str, color_load: str, color_pv: str, color_temp: str):
    """Triple subplot pour une période de 15 jours."""
    t    = data["timestamp"].to_numpy()
    load = data["load"].to_numpy()
    pv   = data["pv_total_kwh"].to_numpy()
    temp = data["temperature_c"].to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    ax1.plot(t, load, color=color_load, linewidth=0.8)
    ax1.set_ylabel("Charge [-]", fontsize=11)
    ax1.set_title(f"15 jours {saison} — Charge, PV, Température", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)

    ax2.fill_between(t, pv, color=color_pv, alpha=0.7)
    ax2.set_ylabel("PV total (kWh)", fontsize=11)
    ax2.grid(alpha=0.3)

    ax3.plot(t, temp, color=color_temp, linewidth=0.8)
    ax3.set_ylabel("Température (°C)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.grid(alpha=0.3)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    return fig

# ── Été ───────────────────────────────────────────────────────────────────────
fig_ete = plot_15j(ete, "été (juil. 2024)", "#3498db", "#f39c12", "#e74c3c")
fig_ete.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_5a_ete_15j.png", dpi=150)
plt.show()

# ── Hiver ─────────────────────────────────────────────────────────────────────
fig_hiver = plot_15j(hiver, "hiver (janv. 2024)", "#3498db", "#f39c12", "#e74c3c")
fig_hiver.savefig(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\00-data\viz\corr_5b_hiver_15j.png", dpi=150)
plt.show()


