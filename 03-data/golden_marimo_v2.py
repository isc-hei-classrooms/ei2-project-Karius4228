import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", app_title="Oiken — Évaluation Golden Dataset v3")


@app.cell
def _():
    import marimo as mo
    return (mo,)


# ── Imports & config ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import joblib
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    PROJECT_ROOT  = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228")
    DST_ROOT      = PROJECT_ROOT / "03-data"
    DST_PROCESSED = DST_ROOT / "data" / "processed"
    DST_MODELS    = DST_ROOT / "models"
    DST_RESULTS   = DST_ROOT / "results"

    PRED_PATH    = DST_RESULTS   / "golden_predictions.joblib"
    METRICS_PATH = DST_RESULTS   / "golden_metrics.json"
    SCALER_PATH  = DST_PROCESSED / "pv_scaler_v4_extended.parquet"

    COLORS = {
        "real"  : "#2b2b2b",
        "oiken" : "#1f3a68",
        "xgb"   : "#c85a17",
        "lgb"   : "#6b8e23",
        "naive" : "#888888",
        "scaler": "#7b3fa0",
        "accent": "#8b0000",
        "pos"   : "#2a7a2a",
        "neg"   : "#8b0000",
    }

    plt.rcParams.update({
        "figure.dpi"      : 110,
        "font.family"     : "DejaVu Sans",
        "font.size"       : 10.5,
        "axes.titlesize"  : 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize"  : 10,
        "axes.edgecolor"  : "#555555",
        "axes.linewidth"  : 0.8,
        "axes.grid"       : True,
        "axes.axisbelow"  : True,
        "grid.color"      : "#d8d8d8",
        "grid.linewidth"  : 0.6,
        "legend.frameon"  : False,
        "legend.fontsize" : 9.5,
        "xtick.labelsize" : 9,
        "ytick.labelsize" : 9,
    })

    return (
        COLORS, METRICS_PATH, PRED_PATH, SCALER_PATH,
        DST_MODELS, joblib, json, mdates, np, pl, plt, Path,
    )


# ── Chargement & préparation données ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(METRICS_PATH, PRED_PATH, SCALER_PATH, DST_MODELS, joblib, json, np, pl):
    from datetime import datetime, timezone

    _pred = joblib.load(PRED_PATH)

    def _arr(keys):
        for k in keys:
            if k in _pred:
                return np.asarray(_pred[k])
        return None

    _ts      = _arr(["timestamps", "timestamp"])
    _y_true  = _arr(["y_true_raw",  "y_test_raw", "y_true"])
    _y_xgb   = _arr(["y_xgb_raw",  "xgb_raw"])
    _y_lgb   = _arr(["y_lgb_raw",  "lgb_raw"])
    _y_oiken = _arr(["y_oiken",    "forecast_load_target"])
    _y_naive = _arr(["y_naive"])
    _scaler  = _arr(["scaler_target", "pv_scaler_target"])

    # Convertir timestamps en datetime Python naïfs (UTC strip)
    def _to_naive(ts_arr):
        out = []
        for t in ts_arr:
            if hasattr(t, "tzinfo") and t.tzinfo is not None:
                t = t.replace(tzinfo=None)
            out.append(t)
        return out

    _ts_naive = _to_naive(_ts)

    # Construction DataFrame Polars
    df = pl.DataFrame({
        "timestamp"     : _ts_naive,
        "real"          : _y_true.astype(float),
        "oiken"         : _y_oiken.astype(float),
        "xgb"           : _y_xgb.astype(float),
        "lgb"           : _y_lgb.astype(float),
        "naive"         : _y_naive.astype(float),
        "scaler_target" : _scaler.astype(float),
    }).with_columns([
        pl.col("timestamp").cast(pl.Datetime("us")),
    ]).sort("timestamp")

    # Features temporelles
    df = df.with_columns([
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("weekday"),
        # saison : hiver = nov dec jan fev mar
        pl.when(pl.col("timestamp").dt.month().is_in([11, 12, 1, 2, 3]))
          .then(pl.lit("hiver")).otherwise(pl.lit("été")).alias("season"),
        # mois tronqué pour agrégation
        (pl.col("timestamp").dt.truncate("1mo")).alias("month_ts"),
    ])

    # Erreurs
    df = df.with_columns([
        (pl.col("xgb")   - pl.col("real")).alias("err_xgb"),
        (pl.col("lgb")   - pl.col("real")).alias("err_lgb"),
        (pl.col("oiken") - pl.col("real")).alias("err_oiken"),
    ]).with_columns([
        pl.col("err_xgb").abs().alias("ae_xgb"),
        pl.col("err_lgb").abs().alias("ae_lgb"),
        pl.col("err_oiken").abs().alias("ae_oiken"),
    ])

    # Bornes des zones (naïf, sans tz)
    TRAIN_END = datetime(2024, 11, 6, 19, 30)
    TEST_END  = datetime(2025, 9, 29, 22, 0)
    B_START   = datetime(2025, 3, 31, 22, 0)

    ZONE_MASKS = {
        "in_train" : pl.col("timestamp") <= pl.lit(TRAIN_END).cast(pl.Datetime("us")),
        "in_test"  : (pl.col("timestamp") > pl.lit(TRAIN_END).cast(pl.Datetime("us"))) &
                     (pl.col("timestamp") <= pl.lit(TEST_END).cast(pl.Datetime("us"))),
        "unseen"   : pl.col("timestamp") > pl.lit(TEST_END).cast(pl.Datetime("us")),
        "B_12mois" : pl.col("timestamp") >= pl.lit(B_START).cast(pl.Datetime("us")),
    }

    # Métriques JSON
    with open(METRICS_PATH) as _f:
        metrics = json.load(_f)

    # Scaler PV
    scaler_df = pl.read_parquet(SCALER_PATH).with_columns([
        pl.date(pl.col("year"), pl.col("month"), pl.lit(15))
          .cast(pl.Datetime("us")).alias("timestamp")
    ])

    # Modèles
    _xgb_p = DST_MODELS / "xgb_da_v4.joblib"
    _lgb_p = DST_MODELS / "lgb_da_v4.joblib"
    xgb_model = joblib.load(_xgb_p) if _xgb_p.exists() else None
    lgb_model  = joblib.load(_lgb_p) if _lgb_p.exists() else None

    return (
        df, metrics, scaler_df, xgb_model, lgb_model,
        ZONE_MASKS, TRAIN_END, TEST_END, B_START,
        datetime,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TITRE
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Évaluation sur le Golden Dataset — Oiken Load Prediction v4
    **Marius Fabbri** · Energy Informatics 2 · HES-SO Valais-Wallis · Avril 2026

    Modèles : **XGBoost v4** + **LightGBM v4** — normalisation proxy capacité PV
    Période : **sept 2023 → avr 2026** · pas 15 min · horizon +24 h · **aucun réentraînement**
    Pipeline golden calibré v3 : `target_raw` et `forecast_load` dans l'espace z-score train.

    ---
    """)
    return


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — TABLEAU COMPARATIF 4 ZONES
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    mo.md("## 0 · Tableau comparatif — 4 zones")
    return


@app.cell(hide_code=True)
def _(metrics, mo, pl):
    _zones  = ["in_train", "in_test", "unseen", "B_12mois"]
    _labels = {
        "in_train" : "In-train (sept 2023 → nov 2024)",
        "in_test"  : "In-test  (nov 2024 → sept 2025)",
        "unseen"   : "Unseen   (oct 2025 → avr 2026)",
        "B_12mois" : "12 mois  (avr 2025 → avr 2026)",
    }

    _rows = {
        "Zone"       : [],
        "n"          : [],
        "Naïf MAE"   : [],
        "Oiken MAE"  : [],
        "XGB MAE"    : [],
        "LGB MAE"    : [],
        "Gain XGB"   : [],
        "Biais XGB"  : [],
        "Biais Oiken": [],
    }

    for _z in _zones:
        if _z not in metrics["zones"]:
            continue
        _g = metrics["zones"][_z]["global"]
        _rows["Zone"].append(_labels[_z])
        _rows["n"].append(metrics["zones"][_z]["n"])
        _rows["Naïf MAE"].append(round(_g["naive"]["MAE"],   4))
        _rows["Oiken MAE"].append(round(_g["oiken"]["MAE"],  4))
        _rows["XGB MAE"].append(round(_g["xgb_v4"]["MAE"],   4))
        _rows["LGB MAE"].append(round(_g["lgb_v4"]["MAE"],   4))
        _rows["Gain XGB"].append(f"{_g['gain_xgb_vs_oiken_pct']:+.1f}%")
        _rows["Biais XGB"].append(f"{_g['xgb_v4']['bias']:+.4f}")
        _rows["Biais Oiken"].append(f"{_g['oiken']['bias']:+.4f}")

    _tbl = pl.DataFrame(_rows)
    mo.ui.table(_tbl, selection=None)


# ══════════════════════════════════════════════════════════════════════════════
# SÉLECTEUR DE ZONE (partagé par toutes les sections suivantes)
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    zone_selector = mo.ui.dropdown(
        options={
            "In-train  (sept 2023 → nov 2024)" : "in_train",
            "In-test   (nov 2024 → sept 2025)" : "in_test",
            "Unseen    (oct 2025 → avr 2026)"  : "unseen",
            "12 mois   (avr 2025 → avr 2026)"  : "B_12mois",
        },
        value="In-test   (nov 2024 → sept 2025)",
        label="**Zone d'évaluation (toutes les sections) :**",
    )
    zone_selector
    return (zone_selector,)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PERFORMANCE PAR ZONE
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    mo.md("## 1 · Performance par zone")
    return


@app.cell(hide_code=True)
def _(COLORS, metrics, np, plt, zone_selector):
    _z  = zone_selector.value
    _g  = metrics["zones"].get(_z, {}).get("global", {})
    _ok = bool(_g)
    _models = ["Naïf J-1", "Oiken", "XGBoost v4", "LightGBM v4"]
    _maes  = [_g["naive"]["MAE"],  _g["oiken"]["MAE"],  _g["xgb_v4"]["MAE"],  _g["lgb_v4"]["MAE"]]  if _ok else [0,0,0,0]
    _rmses = [_g["naive"]["RMSE"], _g["oiken"]["RMSE"], _g["xgb_v4"]["RMSE"], _g["lgb_v4"]["RMSE"]] if _ok else [0,0,0,0]
    _clrs  = [COLORS["naive"], COLORS["oiken"], COLORS["xgb"], COLORS["lgb"]]

    fig1, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for _ax, _vals, _ylabel, _ttl in [
        (axes[0], _maes,  "MAE (z-score)",  f"MAE — {_z}"),
        (axes[1], _rmses, "RMSE (z-score)", f"RMSE — {_z}"),
    ]:
        _x = np.arange(len(_models))
        _bars = _ax.bar(_x, _vals, color=_clrs, edgecolor="white", width=0.55)
        for _b, _v in zip(_bars, _vals):
            _ax.text(_b.get_x() + _b.get_width()/2, _v + 0.003,
                    f"{_v:.4f}", ha="center", fontsize=9.5, fontweight="semibold")
        _ax.set_xticks(_x); _ax.set_xticklabels(_models)
        _ax.set_ylabel(_ylabel); _ax.set_title(_ttl, loc="left")
        _ax.set_ylim(0, max(_vals) * 1.20 if _ok else 1)
        _ax.grid(axis="x", visible=False)
    plt.tight_layout()
    fig1


@app.cell(hide_code=True)
def _(COLORS, metrics, np, plt, zone_selector):
    _z = zone_selector.value
    _s = metrics["zones"].get(_z, {}).get("seasonal", {})

    def _get(mk, season):
        return _s.get(mk, {}).get(season, {}).get("MAE", float("nan")) if _s else float("nan")

    _labels = ["Oiken", "XGBoost v4", "LightGBM v4"]
    _clrs   = [COLORS["oiken"], COLORS["xgb"], COLORS["lgb"]]
    _hiver  = [_get("oiken","winter"), _get("xgb_v4","winter"), _get("lgb_v4","winter")]
    _ete    = [_get("oiken","summer"), _get("xgb_v4","summer"), _get("lgb_v4","summer")]

    fig_seas, _ax = plt.subplots(figsize=(9, 4.5))
    _x = np.arange(len(_labels)); _w = 0.32
    _b1 = _ax.bar(_x - _w/2, _hiver, _w, color=_clrs, alpha=0.6,
                 edgecolor="white", label="Hiver (nov–mars)")
    _b2 = _ax.bar(_x + _w/2, _ete,   _w, color=_clrs, alpha=1.0,
                 edgecolor="white", hatch="//", label="Été (avr–oct)")
    for _b, _v in list(zip(_b1, _hiver)) + list(zip(_b2, _ete)):
        if not np.isnan(_v):
            _ax.text(_b.get_x() + _b.get_width()/2, _v + 0.004,
                    f"{_v:.3f}", ha="center", fontsize=8.5)
    _ax.set_xticks(_x); _ax.set_xticklabels(_labels)
    _ax.set_ylabel("MAE (z-score)")
    _ax.set_title(f"MAE saisonnière — {_z}", loc="left")
    _ymax = np.nanmax(_hiver + _ete)
    _ax.set_ylim(0, _ymax * 1.25 if not np.isnan(_ymax) else 1)
    _ax.legend(loc="upper right"); _ax.grid(axis="x", visible=False)
    plt.tight_layout()
    fig_seas


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DÉRIVE TEMPORELLE
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    mo.md("## 2 · Dérive temporelle — MAE et biais mensuels")
    return


@app.cell(hide_code=True)
def _(COLORS, TRAIN_END, TEST_END, df, mdates, np, plt, scaler_df):
    # MAE mensuelle globale + zones colorées + scaler PV
    _agg = (
        df.group_by("month_ts")
        .agg([
            pl.col("ae_oiken").mean().alias("mae_oiken"),
            pl.col("ae_xgb").mean().alias("mae_xgb"),
            pl.col("ae_lgb").mean().alias("mae_lgb"),
        ])
        .sort("month_ts")
    )

    _months = _agg["month_ts"].to_list()
    _mae_oiken = _agg["mae_oiken"].to_numpy()
    _mae_xgb   = _agg["mae_xgb"].to_numpy()
    _mae_lgb   = _agg["mae_lgb"].to_numpy()

    _sc_ts  = scaler_df["timestamp"].to_list()
    _sc_val = scaler_df["pv_scaler"].to_numpy()

    fig_drift, (_ax1, _ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    _ax1.fill_betweenx(
        [0, max(_mae_xgb.max(), _mae_oiken.max()) * 1.3],
        _months[0], TRAIN_END, color="#e8f5e9", alpha=0.4, label="in_train",
    )
    _ax1.fill_betweenx(
        [0, max(_mae_xgb.max(), _mae_oiken.max()) * 1.3],
        TRAIN_END, TEST_END, color="#fff3e0", alpha=0.4, label="in_test",
    )
    _ax1.fill_betweenx(
        [0, max(_mae_xgb.max(), _mae_oiken.max()) * 1.3],
        TEST_END, _months[-1], color="#fce4ec", alpha=0.4, label="unseen",
    )
    _ax1.plot(_months, _mae_oiken, color=COLORS["oiken"],
             linewidth=2, marker="o", markersize=3.5, label="Oiken")
    _ax1.plot(_months, _mae_xgb,   color=COLORS["xgb"],
             linewidth=2, marker="s", markersize=3.5, label="XGBoost v4")
    _ax1.plot(_months, _mae_lgb,   color=COLORS["lgb"],
             linewidth=1.5, marker="^", markersize=3.5, linestyle="--",
             label="LightGBM v4", alpha=0.85)
    _ax1.set_ylabel("MAE (z-score)")
    _ax1.set_title("MAE mensuelle globale · sept 2023 → avr 2026", loc="left")
    _ax1.legend(loc="upper left", ncol=3, fontsize=9)
    _ax1.set_ylim(bottom=0)

    _ax2.fill_between(_sc_ts, _sc_val, alpha=0.25, color=COLORS["scaler"])
    _ax2.plot(_sc_ts, _sc_val, color=COLORS["scaler"], linewidth=2,
             label="pv_scaler (proxy capacité PV)")
    _ax2.set_ylabel("Scaler PV")
    _ax2.set_title("Croissance proxy capacité PV", loc="left")
    _ax2.legend(loc="upper left")

    for _ax in [_ax1, _ax2]:
        _ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        _ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    fig_drift


@app.cell(hide_code=True)
def _(COLORS, ZONE_MASKS, df, mdates, plt, zone_selector):
    # Biais mensuel — zone sélectionnée
    _z    = zone_selector.value
    _df_z = df.filter(ZONE_MASKS[_z])

    _agg = (
        _df_z.group_by("month_ts")
        .agg([
            pl.col("err_oiken").mean().alias("bias_oiken"),
            pl.col("err_xgb").mean().alias("bias_xgb"),
            pl.col("err_lgb").mean().alias("bias_lgb"),
        ])
        .sort("month_ts")
    )

    _months     = _agg["month_ts"].to_list()
    _bias_oiken = _agg["bias_oiken"].to_numpy()
    _bias_xgb   = _agg["bias_xgb"].to_numpy()
    _bias_lgb   = _agg["bias_lgb"].to_numpy()

    fig_bias, _ax = plt.subplots(figsize=(13, 4.2))
    _ax.axhline(0, color="#444", linewidth=1.0)
    _ax.fill_between(_months, 0, _bias_xgb,
                    where=_bias_xgb > 0, color=COLORS["xgb"], alpha=0.15)
    _ax.fill_between(_months, 0, _bias_xgb,
                    where=_bias_xgb < 0, color=COLORS["lgb"], alpha=0.15)
    _ax.plot(_months, _bias_oiken, color=COLORS["oiken"],
            linewidth=1.8, marker="o", markersize=3.5, label="Oiken")
    _ax.plot(_months, _bias_xgb,   color=COLORS["xgb"],
            linewidth=1.8, marker="s", markersize=3.5, label="XGBoost v4")
    _ax.plot(_months, _bias_lgb,   color=COLORS["lgb"],
            linewidth=1.4, marker="^", markersize=3.5, linestyle="--",
            label="LightGBM v4", alpha=0.85)
    _ax.set_ylabel("Biais moyen (prédit − réel)")
    _ax.set_title(f"Biais mensuel — zone : {_z}", loc="left")
    _ax.legend(loc="upper left", ncol=3)
    _ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    _ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    fig_bias


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COURBES HEBDOMADAIRES
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    mo.md("## 3 · Courbes de charge hebdomadaires")
    return


@app.cell(hide_code=True)
def _(mo):
    week_selector = mo.ui.dropdown(
        options={
            "[in_train]  Hiver jan 2025  (13–19 jan 2025)"  : "2025-01-13",
            "[in_train]  Été jul 2024    (14–20 jul 2024)"  : "2024-07-14",
            "[in_test]   Hiver déc 2024  (16–22 déc 2024)"  : "2024-12-16",
            "[in_test]   Été jul 2025    (14–20 jul 2025)"  : "2025-07-14",
            "[unseen]    Automne oct 2025 (06–12 oct 2025)" : "2025-10-06",
            "[unseen]    Hiver jan 2026  (12–18 jan 2026)"  : "2026-01-12",
            "[unseen]    Avril 2026 sem1  (01–07 avr 2026)" : "2026-04-01",
            "[unseen]    Avril 2026 sem2  (08–14 avr 2026)" : "2026-04-08",
            "[unseen]    Avril 2026 sem3  (15–21 avr 2026)" : "2026-04-15",
        },
        value="[unseen]    Avril 2026 sem1  (01–07 avr 2026)",
        label="**Sélectionner une semaine :**",
    )
    week_selector
    return (week_selector,)


@app.cell(hide_code=True)
def _(COLORS, datetime, df, mdates, plt, week_selector, pl):
    from datetime import timedelta as _td
    _start = datetime.fromisoformat(week_selector.value)
    _end   = _start + _td(days=7)
    _end6  = _start + _td(days=6)

    _sub = df.filter(
        (pl.col("timestamp") >= pl.lit(_start).cast(pl.Datetime("us"))) &
        (pl.col("timestamp") <  pl.lit(_end).cast(pl.Datetime("us")))
    ).sort("timestamp")

    _ts    = _sub["timestamp"].to_list()
    _real  = _sub["real"].to_numpy()
    _oiken = _sub["oiken"].to_numpy()
    _xgb   = _sub["xgb"].to_numpy()
    _lgb   = _sub["lgb"].to_numpy()
    _ae_o  = _sub["ae_oiken"].to_numpy()
    _ae_x  = _sub["ae_xgb"].to_numpy()

    _mae_o = float(_sub["ae_oiken"].mean()) if _sub.height > 0 else 0.0
    _mae_x = float(_sub["ae_xgb"].mean())   if _sub.height > 0 else 0.0
    _mae_l = float(_sub["ae_lgb"].mean())   if _sub.height > 0 else 0.0

    fig_week, (_ax_top, _ax_bot) = plt.subplots(
        2, 1, figsize=(13, 6.5),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    if _sub.height == 0:
        _ax_top.text(0.5, 0.5, f"Aucune donnée pour la semaine du {_start.date()}",
                    ha="center", va="center", transform=_ax_top.transAxes)
    else:
        _ax_top.plot(_ts, _real,  color=COLORS["real"],  linewidth=2.0, label="Réel", zorder=5)
        _ax_top.plot(_ts, _oiken, color=COLORS["oiken"], linewidth=1.5,
                    label=f"Oiken  (MAE={_mae_o:.3f})", alpha=0.9, zorder=3)
        _ax_top.plot(_ts, _xgb,   color=COLORS["xgb"],   linewidth=1.5,
                    label=f"XGB v4 (MAE={_mae_x:.3f})", alpha=0.9, zorder=4)
        _ax_top.plot(_ts, _lgb,   color=COLORS["lgb"],   linewidth=1.2,
                    label=f"LGB v4 (MAE={_mae_l:.3f})", alpha=0.8, linestyle="--", zorder=4)
        _ax_bot.fill_between(_ts, 0, _ae_o, color=COLORS["oiken"], alpha=0.3, label="|err| Oiken")
        _ax_bot.fill_between(_ts, 0, _ae_x, color=COLORS["xgb"],   alpha=0.3, label="|err| XGB v4")
        _ax_bot.plot(_ts, _ae_o, color=COLORS["oiken"], linewidth=1.0)
        _ax_bot.plot(_ts, _ae_x, color=COLORS["xgb"],   linewidth=1.0)

    _ax_top.set_ylabel("Charge nette (z-score)")
    _ax_top.set_title(
        f"Semaine du {_start.strftime('%d %b %Y')} au {_end6.strftime('%d %b %Y')}",
        loc="left", fontsize=13,
    )
    _ax_top.legend(loc="upper right", ncol=4, fontsize=10)
    _ax_bot.set_ylabel("Erreur abs.")
    _ax_bot.legend(loc="upper right", ncol=2, fontsize=9.5)
    _ax_bot.xaxis.set_major_locator(mdates.DayLocator())
    _ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    plt.tight_layout()
    fig_week


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SCATTER RÉEL vs PRÉDIT
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    mo.md("## 4 · Scatter réel vs prédit — diagnostic du biais")
    return


@app.cell(hide_code=True)
def _(COLORS, ZONE_MASKS, df, np, plt, zone_selector, pl):
    _z    = zone_selector.value
    _df_z = df.filter(ZONE_MASKS[_z])

    # Sample aléatoire reproductible
    _n   = min(6000, _df_z.height)
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(_df_z.height, _n, replace=False)
    _sub = _df_z[_idx.tolist()]

    _real  = _sub["real"].to_numpy()
    _lim   = (min(_real.min(), _sub["xgb"].min(), _sub["oiken"].min()) - 0.15,
              max(_real.max(), _sub["xgb"].max(), _sub["oiken"].max()) + 0.15)
    _diag  = np.linspace(_lim[0], _lim[1], 100)

    fig_scatter, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for _ax, _col, _label, _color in [
        (_ax1, "xgb",   "XGBoost v4", COLORS["xgb"]),
        (_ax2, "oiken", "Oiken",      COLORS["oiken"]),
    ]:
        _pred_arr = _sub[_col].to_numpy()
        _win_mask = _sub["season"].to_numpy() == "hiver"
        _ete_mask = ~_win_mask

        _ax.scatter(_real[_win_mask], _pred_arr[_win_mask],
                   color=COLORS["oiken"], alpha=0.18, s=5, label="Hiver")
        _ax.scatter(_real[_ete_mask], _pred_arr[_ete_mask],
                   color=_color, alpha=0.28, s=5, label="Été")
        _ax.plot(_diag, _diag, color="#444", linewidth=1.2, linestyle="--", label="Parfait")

        _coeffs = np.polyfit(_real, _pred_arr, 1)
        _ax.plot(_diag, np.polyval(_coeffs, _diag),
                color="#c00", linewidth=1.2, label=f"Régression (a={_coeffs[0]:.3f})")

        _mae_v  = float(_df_z[f"ae_{_col}"].mean()) if f"ae_{_col}" in _df_z.columns else float("nan")
        _bias_v = float(_df_z[f"err_{_col}"].mean()) if f"err_{_col}" in _df_z.columns else float("nan")

        _ax.set_xlabel("Réel (z-score)"); _ax.set_ylabel("Prédit (z-score)")
        _ax.set_title(f"{_label} — {_z}\nMAE={_mae_v:.4f}  biais={_bias_v:+.4f}", loc="left")
        _ax.set_xlim(_lim); _ax.set_ylim(_lim)
        _ax.legend(markerscale=3, fontsize=9)
        _ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig_scatter


@app.cell(hide_code=True)
def _(COLORS, ZONE_MASKS, df, np, plt, zone_selector):
    # Erreur XGB vs scaler PV + distribution par saison
    _z    = zone_selector.value
    _df_z = df.filter(ZONE_MASKS[_z])
    _n    = min(6000, _df_z.height)
    _idx  = np.random.default_rng(42).choice(_df_z.height, _n, replace=False)
    _sub  = _df_z[_idx.tolist()]

    _sc  = _sub["scaler_target"].to_numpy()
    _err = _sub["err_xgb"].to_numpy()
    _sea = _sub["season"].to_numpy()

    fig_errsc, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    _ax1.scatter(_sc[_sea == "hiver"], _err[_sea == "hiver"],
                color=COLORS["oiken"], alpha=0.15, s=6, label="Hiver")
    _ax1.scatter(_sc[_sea == "été"],   _err[_sea == "été"],
                color=COLORS["xgb"],   alpha=0.25, s=6, label="Été")
    _ax1.axhline(0, color="#444", linewidth=1.0)
    _ax1.axvline(3.8, color=COLORS["accent"], linewidth=1.2, linestyle="--")
    _ax1.text(3.85, float(np.quantile(_err, 0.90)),
             "max train ≈ 3.8", fontsize=8.5, color=COLORS["accent"])
    _ax1.set_xlabel("pv_scaler_target")
    _ax1.set_ylabel("Erreur XGB v4 (prédit − réel)")
    _ax1.set_title(f"Erreur XGB vs scaler PV — {_z}", loc="left")
    _ax1.legend(markerscale=3)

    _err_full = _df_z["err_xgb"].to_numpy()
    _sea_full = _df_z["season"].to_numpy()
    _eh = _err_full[_sea_full == "hiver"]
    _ee = _err_full[_sea_full == "été"]
    _bh = float(_eh.mean()) if len(_eh) > 0 else float("nan")
    _be = float(_ee.mean()) if len(_ee) > 0 else float("nan")

    if len(_eh) > 10:
        _ax2.hist(_eh, bins=80, color=COLORS["oiken"], alpha=0.6,
                 label=f"Hiver (biais={_bh:+.3f})", density=True)
    if len(_ee) > 10:
        _ax2.hist(_ee, bins=80, color=COLORS["xgb"],   alpha=0.6,
                 label=f"Été   (biais={_be:+.3f})", density=True)
    _ax2.axvline(0, color="#444", linewidth=1.0)
    if not np.isnan(_bh): _ax2.axvline(_bh, color=COLORS["oiken"], linewidth=1.5, linestyle="--")
    if not np.isnan(_be): _ax2.axvline(_be, color=COLORS["xgb"],   linewidth=1.5, linestyle="--")
    _ax2.set_xlabel("Erreur XGB v4 (prédit − réel)")
    _ax2.set_ylabel("Densité")
    _ax2.set_title(f"Distribution erreurs XGB par saison — {_z}", loc="left")
    _ax2.legend()
    plt.tight_layout()
    fig_errsc


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PROFIL JOURNALIER (zone sélectionnée)
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    mo.md("## 5 · Profil journalier — MAE et biais par heure")
    return


@app.cell(hide_code=True)
def _(COLORS, ZONE_MASKS, df, np, plt, zone_selector):
    _z    = zone_selector.value
    _df_z = df.filter(ZONE_MASKS[_z])

    _by_h = (
        _df_z.group_by("hour")
        .agg([
            pl.col("ae_oiken").mean().alias("mae_oiken"),
            pl.col("ae_xgb").mean().alias("mae_xgb"),
            pl.col("err_xgb").mean().alias("bias_xgb"),
            pl.col("err_oiken").mean().alias("bias_oiken"),
        ])
        .sort("hour")
    )

    _h         = _by_h["hour"].to_numpy()
    _mae_o     = _by_h["mae_oiken"].to_numpy()
    _mae_x     = _by_h["mae_xgb"].to_numpy()
    _bias_x    = _by_h["bias_xgb"].to_numpy()
    _bias_o    = _by_h["bias_oiken"].to_numpy()

    fig_hour, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    _ax1.plot(_h, _mae_o, color=COLORS["oiken"], linewidth=2,
             marker="o", markersize=3.5, label="Oiken")
    _ax1.plot(_h, _mae_x, color=COLORS["xgb"],   linewidth=2,
             marker="s", markersize=3.5, label="XGB v4")
    _ax1.fill_between(_h, _mae_o, _mae_x,
                     where=_mae_x < _mae_o,
                     color=COLORS["pos"], alpha=0.2, label="ML meilleur")
    _ax1.fill_between(_h, _mae_o, _mae_x,
                     where=_mae_x >= _mae_o,
                     color=COLORS["neg"], alpha=0.15, label="ML moins bon")
    _ax1.axvspan(10, 14, color="#ffe0b2", alpha=0.4, label="Heures solaires")
    _ax1.set_xlabel("Heure (UTC)"); _ax1.set_ylabel("MAE moyenne")
    _ax1.set_title(f"MAE par heure — {_z}", loc="left")
    _ax1.set_xticks(range(0, 24, 2)); _ax1.legend(fontsize=9)

    _ax2.axhline(0, color="#444", linewidth=0.8)
    _ax2.fill_between(_h, 0, _bias_x,
                     where=_bias_x > 0, color=COLORS["xgb"], alpha=0.3)
    _ax2.fill_between(_h, 0, _bias_x,
                     where=_bias_x <= 0, color=COLORS["lgb"], alpha=0.3)
    _ax2.plot(_h, _bias_o, color=COLORS["oiken"], linewidth=1.8, label="Biais Oiken")
    _ax2.plot(_h, _bias_x, color=COLORS["xgb"],   linewidth=1.8, label="Biais XGB v4")
    _ax2.axvspan(10, 14, color="#ffe0b2", alpha=0.4, label="Heures solaires")
    _ax2.set_xlabel("Heure (UTC)"); _ax2.set_ylabel("Biais moyen (prédit − réel)")
    _ax2.set_title(f"Biais par heure — {_z}", loc="left")
    _ax2.set_xticks(range(0, 24, 2)); _ax2.legend(fontsize=9)

    plt.tight_layout()
    fig_hour


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    mo.md("## 6 · Importance des features")
    return


@app.cell(hide_code=True)
def _(COLORS, lgb_model, np, plt, xgb_model, PRED_PATH, joblib):
    from matplotlib.patches import Patch as _Patch

    # Récupérer les noms de features sauvegardés dans golden_predictions.joblib
    _pred_data   = joblib.load(PRED_PATH)
    _feat_names  = _pred_data.get("feature_cols", [])  # liste sauvegardée par predict_golden.py

    fig_fi, _axes_fi = plt.subplots(1, 2, figsize=(13, 6))
    for _ax, _model, _label, _color in [
        (_axes_fi[0], xgb_model, "XGBoost v4",  COLORS["xgb"]),
        (_axes_fi[1], lgb_model, "LightGBM v4", COLORS["lgb"]),
    ]:
        if _model is None or not hasattr(_model, "feature_importances_"):
            _ax.text(0.5, 0.5, "Modèle indisponible",
                    ha="center", va="center", transform=_ax.transAxes)
            continue
        _imp = _model.feature_importances_

        # Priorité : feature_cols du joblib → feature_names_in_ du modèle → indices
        if _feat_names and len(_feat_names) == len(_imp):
            _names = list(_feat_names)
        elif hasattr(_model, "feature_names_in_") and _model.feature_names_in_ is not None:
            _names = list(_model.feature_names_in_)
        else:
            _names = [f"f{i}" for i in range(len(_imp))]

        _idx = np.argsort(_imp)[-15:]
        _nt  = [_names[i] for i in _idx]
        _it  = _imp[_idx]
        _bc  = [
            COLORS["accent"] if "pv_scaler" in n
            else COLORS["xgb"] if ("pv_" in n or "rolling" in n)
            else _color
            for n in _nt
        ]
        _ax.barh(range(len(_nt)), _it, color=_bc, edgecolor="white")
        _ax.set_yticks(range(len(_nt))); _ax.set_yticklabels(_nt, fontsize=9)
        _ax.set_xlabel("Importance")
        _ax.set_title(f"Top 15 features — {_label}", loc="left")

    _axes_fi[1].legend(handles=[
        _Patch(facecolor=COLORS["accent"], label="pv_scaler (proxy PV)"),
        _Patch(facecolor=COLORS["xgb"],    label="Features PV / rolling"),
        _Patch(facecolor=COLORS["lgb"],    label="Autres features"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout()
    fig_fi


# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Conclusion

    | | in_train | in_test | unseen | B_12mois |
    |---|---|---|---|---|
    | **XGB MAE** | 0.103 | 0.226 | 0.251 | 0.241 |
    | **Oiken MAE** | 0.179 | 0.173 | 0.152 | 0.159 |
    | **Gain XGB** | **+42.5%** ✓ | −30.9% | −65.7% | −51.3% |
    | **Biais XGB** | +0.009 | +0.016 | +0.122 | +0.050 |

    **Interprétation** :
    - Sur `in_train`, XGB surpasse Oiken de +42% — les patterns temporels sont bien mémorisés
    - Sur `in_test` et `unseen`, Oiken reprend l'avantage : le modèle figé ne suit pas le drift PV post-entraînement
    - Le biais unseen (+0.122) est de l'extrapolation saisonnière hors distribution — confirmé par Oiken (MAE 0.152, biais +0.019)

    **Pistes prioritaires** : réentraînement glissant · architecture charge brute + PV séparés · ensemble ML hiver + Oiken été

    ---
    *Marius Fabbri · Energy Informatics 2 · HES-SO Valais-Wallis · Avril 2026*
    """)
    return


if __name__ == "__main__":
    app.run()