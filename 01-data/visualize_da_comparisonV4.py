"""
visualize_da_comparison.py — Comparaison v3 vs v4 (normalisation PV) vs Oiken
───────────────────────────────────────────────────────────────────────────────
Charge les prédictions des deux versions et génère des figures comparatives :

  1. Courbes semaine hiver   — là où ML v3 dominait déjà
  2. Courbes semaine été     — là où v3 échouait (concept drift PV)
  3. Courbes semaine printemps
  4. MAE globale : barre groupée v3 / v4 / Oiken
  5. MAE saisonnière : hiver vs été pour les 3 modèles
  6. Scatter erreur v3 vs v4  — chaque point = un pas 15min
  7. Biais mensuel : évolution du biais moyen mois par mois
  8. Feature importance XGBoost v4 (top 15)

Auteur : Marius Fabbri
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime, timezone

# ── Chemins ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
MODELS_DIR  = SCRIPT_DIR / "models_saved"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette cohérente sur toutes les figures ──────────────────────────────────
C = {
    "real":    "#2c3e50",   # gris foncé — charge réelle
    "oiken":   "#e74c3c",   # rouge      — Oiken
    "v3_xgb":  "#95a5a6",   # gris clair — XGB v3
    "v3_lgb":  "#bdc3c7",   # gris très clair — LGB v3
    "v4_xgb":  "#3498db",   # bleu       — XGB v4
    "v4_lgb":  "#2ecc71",   # vert       — LGB v4
}

SEASON_WINDOWS = [
    # (start, end, label_court, filename_suffix)
    (datetime(2025, 1, 13, tzinfo=timezone.utc),
     datetime(2025, 1, 20, tzinfo=timezone.utc),
     "Hiver (13–19 jan 2025)", "winter"),
    (datetime(2025, 6, 2, tzinfo=timezone.utc),
     datetime(2025, 6, 9, tzinfo=timezone.utc),
     "Été (2–8 juin 2025)", "summer"),
    (datetime(2025, 4, 7, tzinfo=timezone.utc),
     datetime(2025, 4, 14, tzinfo=timezone.utc),
     "Printemps (7–13 avr 2025)", "spring"),
]


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Charge les prédictions v3 et v4.
    Retourne deux dicts avec les mêmes clés normalisées.
    """
    d3 = joblib.load(MODELS_DIR / "da_v3_predictions.joblib")
    d4 = joblib.load(MODELS_DIR / "da_v4_predictions.joblib")

    # v3 : les valeurs sont déjà dans l'espace z-score Oiken (pas de dénorm nécessaire)
    # v4 : les valeurs "raw" sont dans le même espace z-score après dénorm
    # On aligne les deux sur les timestamps communs
    ts3 = set(d3["timestamps"])
    ts4 = set(d4["timestamps"])
    common = sorted(ts3 & ts4)

    def align(d, key, common_set):
        ts = d["timestamps"]
        idx = [i for i, t in enumerate(ts) if t in common_set]
        return np.array([d[key][i] for i in idx])

    # v4 utilise y_test (valeurs raw dénormalisées) pour être comparable avec v3
    v3 = {
        "timestamps": common,
        "y_test":  align(d3, "y_test", ts4),
        "y_oiken": align(d3, "y_oiken", ts4),
        "y_xgb":   align(d3, "y_xgb", ts4),
        "y_lgb":   align(d3, "y_lgb", ts4),
        "results": d3["results"],
        "xgb_importance": d3.get("xgb_importance", {}),
    }
    v4 = {
        "timestamps": common,
        "y_test":  align(d4, "y_test", ts3),   # y_test raw = même espace que v3
        "y_oiken": align(d4, "y_oiken", ts3),
        "y_xgb":   align(d4, "y_xgb", ts3),
        "y_lgb":   align(d4, "y_lgb", ts3),
        "results": d4["results"],
        "seasonal": d4.get("seasonal", {}),
        "xgb_importance": d4.get("xgb_importance", {}),
    }
    return v3, v4


def slice_window(data, start, end):
    """Extrait les indices correspondant à une fenêtre temporelle."""
    idx = [i for i, ts in enumerate(data["timestamps"]) if start <= ts < end]
    if not idx:
        return None
    return {
        "ts":     [data["timestamps"][i] for i in idx],
        "real":   data["y_test"][idx],
        "oiken":  data["y_oiken"][idx],
        "v3_xgb": data["y_xgb"][idx],
        "v3_lgb": data["y_lgb"][idx],
    }


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1–3 : COURBES SEMAINE
# ─────────────────────────────────────────────────────────────────────────────

def plot_week_comparison(v3, v4, start, end, season_label, suffix):
    """
    Courbe principale + résidus pour une semaine donnée.
    Affiche v3 XGB / v4 XGB / Oiken / réel.
    LGB omis pour ne pas surcharger (seule la meilleure version est montrée).
    """
    idx3 = [i for i, ts in enumerate(v3["timestamps"]) if start <= ts < end]
    idx4 = [i for i, ts in enumerate(v4["timestamps"]) if start <= ts < end]
    if not idx3 or not idx4:
        print(f"  Aucune donnée pour {season_label}")
        return

    ts = [v3["timestamps"][i] for i in idx3]

    real   = v3["y_test"][idx3]
    oiken  = v3["y_oiken"][idx3]
    xgb_v3 = v3["y_xgb"][idx3]
    xgb_v4 = v4["y_xgb"][idx4]
    lgb_v4 = v4["y_lgb"][idx4]

    mae_oiken  = mae(real, oiken)
    mae_v3     = mae(real, xgb_v3)
    mae_v4_xgb = mae(real, xgb_v4)
    mae_v4_lgb = mae(real, lgb_v4)

    fig = plt.figure(figsize=(17, 11))
    gs  = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[3, 1.2, 1.2], hspace=0.08)

    # ── Panneau principal : courbes ───────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(ts, real,    color=C["real"],   lw=2.0, label="Charge réelle", zorder=6)
    ax0.plot(ts, oiken,   color=C["oiken"],  lw=1.4, ls="--", alpha=0.85,
             label=f"Oiken  (MAE={mae_oiken:.3f})")
    ax0.plot(ts, xgb_v3,  color=C["v3_xgb"], lw=1.2, alpha=0.75,
             label=f"XGB v3 (MAE={mae_v3:.3f})")
    ax0.plot(ts, xgb_v4,  color=C["v4_xgb"], lw=1.4, alpha=0.9,
             label=f"XGB v4 (MAE={mae_v4_xgb:.3f})")
    ax0.plot(ts, lgb_v4,  color=C["v4_lgb"], lw=1.2, alpha=0.8,
             label=f"LGB v4 (MAE={mae_v4_lgb:.3f})")

    ax0.set_title(f"Day-Ahead — Comparaison v3 vs v4 vs Oiken | {season_label}",
                  fontsize=13, fontweight="bold", pad=10)
    ax0.set_ylabel("Charge nette (z-score)")
    ax0.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax0.grid(True, alpha=0.25)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    ax0.xaxis.set_major_locator(mdates.DayLocator())
    ax0.set_xticklabels([])

    # ── Panneau erreur absolue v3 ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    err_oiken = np.abs(real - oiken)
    err_v3    = np.abs(real - xgb_v3)
    ax1.fill_between(ts, err_oiken, color=C["oiken"],  alpha=0.25, label="Oiken")
    ax1.plot(ts, err_oiken, color=C["oiken"],  lw=0.8, alpha=0.6)
    ax1.fill_between(ts, err_v3,    color=C["v3_xgb"], alpha=0.35, label="XGB v3")
    ax1.plot(ts, err_v3,    color=C["v3_xgb"], lw=0.8, alpha=0.7)
    ax1.set_ylabel("|Erreur| v3", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.set_xticklabels([])

    # ── Panneau erreur absolue v4 ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    err_v4_xgb = np.abs(real - xgb_v4)
    err_v4_lgb = np.abs(real - lgb_v4)
    ax2.fill_between(ts, err_v4_xgb, color=C["v4_xgb"], alpha=0.3, label="XGB v4")
    ax2.plot(ts, err_v4_xgb, color=C["v4_xgb"], lw=0.9)
    ax2.plot(ts, err_v4_lgb, color=C["v4_lgb"], lw=0.9, alpha=0.8, label="LGB v4")
    ax2.set_ylabel("|Erreur| v4", fontsize=9)
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha="right")

    out = FIGURES_DIR / f"da_compare_{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 : MAE GLOBALE GROUPÉE
# ─────────────────────────────────────────────────────────────────────────────

def plot_mae_global(v3, v4):
    """
    Barres groupées : Oiken / XGB v3 / LGB v3 / XGB v4 / LGB v4
    Avec annotation du gain relatif par rapport à Oiken.
    """
    r3 = v3["results"]
    r4 = v4["results"]

    labels  = ["Oiken", "XGB v3", "LGB v3", "XGB v4", "LGB v4"]
    maes    = [
        r3["Oiken"]["MAE"],
        r3["XGBoost"]["MAE"],
        r3["LightGBM"]["MAE"],
        r4["XGBoost"]["MAE"],
        r4["LightGBM"]["MAE"],
    ]
    colors  = [C["oiken"], C["v3_xgb"], C["v3_lgb"], C["v4_xgb"], C["v4_lgb"]]
    oiken_mae = r3["Oiken"]["MAE"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, maes, color=colors, edgecolor="#2c3e50",
                  linewidth=0.6, width=0.55)

    for bar, m, lbl in zip(bars, maes, labels):
        # Valeur MAE au-dessus de la barre
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{m:.4f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
        # Gain vs Oiken en dessous de la valeur
        if lbl != "Oiken":
            gain = 100 * (oiken_mae - m) / oiken_mae
            color_gain = "#27ae60" if gain > 0 else "#c0392b"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"({gain:+.1f}%)", ha="center", va="bottom",
                    fontsize=8.5, color=color_gain)

    # Ligne de référence Oiken
    ax.axhline(oiken_mae, color=C["oiken"], lw=1.2, ls="--", alpha=0.6,
               label=f"Oiken = {oiken_mae:.4f}")

    ax.set_ylabel("MAE (z-score Oiken)", fontsize=11)
    ax.set_title("MAE globale — Comparaison v3 vs v4 vs Oiken\n(test set nettoyé)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(maes) * 1.18)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # Séparateur visuel entre v3 et v4
    ax.axvline(2.5, color="#7f8c8d", lw=1, ls=":")
    ax.text(1.5, max(maes) * 1.10, "v3", ha="center", fontsize=10,
            color="#7f8c8d", style="italic")
    ax.text(3.5, max(maes) * 1.10, "v4 (norm. PV)", ha="center", fontsize=10,
            color="#2980b9", style="italic")

    plt.tight_layout()
    out = FIGURES_DIR / "da_mae_global_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 : MAE SAISONNIÈRE
# ─────────────────────────────────────────────────────────────────────────────

def plot_mae_seasonal(v3, v4):
    """
    Double barplot hiver/été pour chaque modèle.
    C'est LA figure clé : montre si v4 corrige le problème été.
    """
    ts3  = np.array(v3["timestamps"])
    months = np.array([t.month for t in ts3])
    winter_mask = np.isin(months, [11, 12, 1, 2, 3])
    summer_mask = ~winter_mask

    def smae(real, pred, mask):
        return float(np.mean(np.abs(real[mask] - pred[mask])))

    real3 = v3["y_test"]
    entries = [
        ("Oiken",   v3["y_oiken"], v3["y_oiken"]),
        ("XGB v3",  v3["y_xgb"],   v3["y_xgb"]),
        ("LGB v3",  v3["y_lgb"],   v3["y_lgb"]),
        ("XGB v4",  v4["y_xgb"],   v4["y_xgb"]),
        ("LGB v4",  v4["y_lgb"],   v4["y_lgb"]),
    ]
    real4 = v4["y_test"]

    labels = []
    mae_w  = []
    mae_s  = []
    cols   = []

    for lbl, pred3, pred4 in entries:
        labels.append(lbl)
        if "v4" in lbl:
            mae_w.append(smae(real4, pred4, winter_mask))
            mae_s.append(smae(real4, pred4, summer_mask))
        else:
            mae_w.append(smae(real3, pred3, winter_mask))
            mae_s.append(smae(real3, pred3, summer_mask))
        cols.append(C["oiken"] if lbl == "Oiken" else
                    C["v3_xgb"] if "v3" in lbl and "XGB" in lbl else
                    C["v3_lgb"] if "v3" in lbl else
                    C["v4_xgb"] if "XGB" in lbl else C["v4_lgb"])

    x = np.arange(len(labels))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # ── Hiver ──
    bars1 = ax1.bar(x, mae_w, width=w * 2, color=cols, edgecolor="#2c3e50",
                    linewidth=0.5)
    for bar, m in zip(bars1, mae_w):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
    oiken_w = mae_w[0]
    ax1.axhline(oiken_w, color=C["oiken"], lw=1.2, ls="--", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("MAE (z-score)")
    ax1.set_title("❄  Hiver (nov–mars)", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.set_ylim(0, max(mae_w) * 1.2)

    # ── Été ──
    bars2 = ax2.bar(x, mae_s, width=w * 2, color=cols, edgecolor="#2c3e50",
                    linewidth=0.5)
    for bar, m in zip(bars2, mae_s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
    oiken_s = mae_s[0]
    ax2.axhline(oiken_s, color=C["oiken"], lw=1.2, ls="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_title("☀  Été (avr–oct)", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.set_ylim(0, max(mae_s) * 1.2)

    # Annotation sur le gain été v4 vs v3
    gain_xgb = 100 * (mae_s[1] - mae_s[3]) / mae_s[1]
    gain_lgb = 100 * (mae_s[2] - mae_s[4]) / mae_s[2]
    ax2.text(0.97, 0.97,
             f"Gain v4 vs v3 :\n XGB {gain_xgb:+.1f}%\n LGB {gain_lgb:+.1f}%",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bdc3c7"))

    fig.suptitle("MAE saisonnière — v3 vs v4 vs Oiken", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "da_mae_seasonal_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 : SCATTER ERREUR v3 vs v4
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_scatter(v3, v4):
    """
    Scatter plot : erreur absolue XGB v3 (x) vs XGB v4 (y).
    Points au-dessus de la diagonale = v4 plus mauvais que v3 sur ce pas.
    Points en-dessous = v4 meilleur.
    Coloré par saison (hiver bleu, été rouge).
    """
    real3  = v3["y_test"]
    real4  = v4["y_test"]
    err_v3 = np.abs(real3 - v3["y_xgb"])
    err_v4 = np.abs(real4 - v4["y_xgb"])

    ts     = np.array(v3["timestamps"])
    months = np.array([t.month for t in ts])
    winter = np.isin(months, [11, 12, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(8, 8))

    sc1 = ax.scatter(err_v3[winter], err_v4[winter], s=3, alpha=0.3,
                     color="#3498db", label="Hiver", rasterized=True)
    sc2 = ax.scatter(err_v3[~winter], err_v4[~winter], s=3, alpha=0.3,
                     color="#e74c3c", label="Été", rasterized=True)

    # Diagonale y=x (v3 = v4)
    lim = max(err_v3.max(), err_v4.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#7f8c8d", lw=1.5, ls="--",
            label="v3 = v4")

    # Zones
    ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.04, color="#2ecc71")
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color="#e74c3c")
    ax.text(lim * 0.82, lim * 0.05, "v4 meilleur",
            color="#27ae60", fontsize=9, style="italic")
    ax.text(lim * 0.02, lim * 0.92, "v3 meilleur",
            color="#c0392b", fontsize=9, style="italic")

    # Stats résumées
    n_v4_better = int((err_v4 < err_v3).sum())
    pct = 100 * n_v4_better / len(err_v3)
    ax.text(0.03, 0.97,
            f"v4 meilleur sur {pct:.1f}% des pas\n"
            f"MAE v3={err_v3.mean():.4f}  v4={err_v4.mean():.4f}",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bdc3c7"))

    ax.set_xlabel("Erreur absolue XGB v3")
    ax.set_ylabel("Erreur absolue XGB v4")
    ax.set_title("Erreur pas à pas : v3 vs v4 (XGBoost)\ncoloré par saison",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    out = FIGURES_DIR / "da_scatter_v3_v4.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 : BIAIS MENSUEL
# ─────────────────────────────────────────────────────────────────────────────

def plot_monthly_bias(v3, v4):
    """
    Biais moyen (pred - réel) mois par mois pour Oiken / XGB v3 / XGB v4.
    Un biais positif = sur-estimation, négatif = sous-estimation.
    La correction du concept drift PV doit réduire le biais négatif en été pour v4.
    """
    ts     = np.array(v3["timestamps"])
    months = np.array([t.month for t in ts])
    years  = np.array([t.year  for t in ts])

    # Étiquettes mois-année uniques triées
    ym_list = sorted(set(zip(years, months)))
    labels  = [f"{m:02d}/{str(y)[2:]}" for y, m in ym_list]

    def monthly_bias(pred, real):
        biases = []
        for y, m in ym_list:
            mask = (years == y) & (months == m)
            if mask.sum() > 0:
                biases.append(float(np.mean(pred[mask] - real[mask])))
            else:
                biases.append(np.nan)
        return biases

    bias_oiken = monthly_bias(v3["y_oiken"], v3["y_test"])
    bias_v3    = monthly_bias(v3["y_xgb"],   v3["y_test"])
    bias_v4    = monthly_bias(v4["y_xgb"],   v4["y_test"])

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axhline(0, color="#2c3e50", lw=1.2, alpha=0.7)

    ax.plot(x, bias_oiken, color=C["oiken"],  lw=1.8, marker="o", ms=4,
            alpha=0.85, label="Oiken")
    ax.plot(x, bias_v3,    color=C["v3_xgb"], lw=1.6, marker="s", ms=4,
            alpha=0.75, label="XGB v3")
    ax.plot(x, bias_v4,    color=C["v4_xgb"], lw=1.8, marker="^", ms=5,
            alpha=0.9,  label="XGB v4 (norm. PV)")

    # Zones saisonnières
    for i, (_, m) in enumerate(ym_list):
        if m in [6, 7, 8]:
            ax.axvspan(i - 0.5, i + 0.5, color="#f39c12", alpha=0.07)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Biais moyen (pred − réel)")
    ax.set_title("Biais mensuel — Oiken vs XGB v3 vs XGB v4\n"
                 "(zones orangées = mois d'été plein ; biais négatif = sous-estimation PV)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIGURES_DIR / "da_monthly_bias.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 : FEATURE IMPORTANCE v4
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance_v4(v4, top_n=15):
    imp = v4.get("xgb_importance", {})
    if not imp:
        print("  Pas de feature importance v4 disponible")
        return
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])[:top_n]
    names  = [x[0] for x in sorted_imp][::-1]
    values = [x[1] for x in sorted_imp][::-1]

    # Colorier différemment la feature pv_scaler_target (nouvelle en v4)
    bar_colors = [
        "#e67e22" if "pv_scaler" in n else "#3498db"
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(names, values, color=bar_colors, edgecolor="#2c3e50",
                   linewidth=0.4)
    ax.set_xlabel("Importance (gain XGBoost)")
    ax.set_title("Top 15 Features — XGBoost v4\n"
                 "(orange = features liées au scaler PV, nouvelle en v4)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(fc="#3498db", label="Features standard"),
        Patch(fc="#e67e22", label="pv_scaler (nouveau v4)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    out = FIGURES_DIR / "da_feature_importance_v4.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("VISUALISATION COMPARATIVE v3 vs v4 vs Oiken")
    print("=" * 60)

    print("\nChargement des prédictions...")
    v3, v4 = load_data()
    print(f"  v3 : {len(v3['timestamps'])} pas | v4 : {len(v4['timestamps'])} pas")
    print(f"  Commun : {len(v3['timestamps'])} pas")

    print("\n── Courbes semaines ──")
    for start, end, label, suffix in SEASON_WINDOWS:
        plot_week_comparison(v3, v4, start, end, label, suffix)

    print("\n── MAE globale ──")
    plot_mae_global(v3, v4)

    print("\n── MAE saisonnière ──")
    plot_mae_seasonal(v3, v4)

    print("\n── Scatter erreur v3 vs v4 ──")
    plot_error_scatter(v3, v4)

    print("\n── Biais mensuel ──")
    plot_monthly_bias(v3, v4)

    print("\n── Feature importance v4 ──")
    plot_feature_importance_v4(v4)

    print(f"\n✓ Toutes les figures dans {FIGURES_DIR}")
    print(f"  8 figures générées :")
    for suffix in ["winter", "summer", "spring"]:
        print(f"    da_compare_{suffix}.png")
    for name in ["da_mae_global_comparison", "da_mae_seasonal_comparison",
                 "da_scatter_v3_v4", "da_monthly_bias",
                 "da_feature_importance_v4"]:
        print(f"    {name}.png")


if __name__ == "__main__":
    main()