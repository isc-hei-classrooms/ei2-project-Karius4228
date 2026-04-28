import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", app_title="Oiken Load Prediction v4")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import joblib
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    from matplotlib.patches import Patch

    # ---------- chemins (Windows, raw strings) ----------
    # Dossier racine des données du projet
    DATA_DIR = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228\01-data")
    PROCESSED_DIR = DATA_DIR / "data" / "processed"
    MODELS_DIR = DATA_DIR / "models_saved"

    # Parquets de données (dans data/processed/)
    PV_SCALER_PATH = PROCESSED_DIR / "pv_scaler_v4.parquet"
    TEST_V4_PATH = PROCESSED_DIR / "test_da_v4.parquet"

    # Artefacts modèle (dans models_saved/)
    PRED_V4_PATH = MODELS_DIR / "da_v4_predictions.joblib"
    METRICS_V4_PATH = MODELS_DIR / "da_v4_metrics.json"

    # ---------- palette sobre ----------
    COLORS = {
        "real": "#2b2b2b",
        "oiken": "#1f3a68",
        "xgb": "#c85a17",
        "lgb": "#6b8e23",
        "scaler": "#1f3a68",
        "grid": "#d8d8d8",
        "band": "#ededed",
        "accent": "#8b0000",
    }

    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    return (
        COLORS,
        PRED_V4_PATH,
        PV_SCALER_PATH,
        Patch,
        joblib,
        mdates,
        np,
        pd,
        pl,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prédiction de la charge nette — Oiken
    ## Day-Ahead **v4** : normalisation par un proxy de capacité PV

    ---

    **Marius Fabbri** · Energy Informatics 2 · HES-SO Valais-Wallis · Avril 2026

    Sous-groupe bilan **Sion — Sierre — Valais central** · pas 15 min · horizon 24 h (96 pas)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Le problème que v4 cherche à résoudre

    Les modèles à base d'arbres (XGBoost, LightGBM) **ne peuvent pas extrapoler** au-delà des valeurs vues en entraînement.
    Or la capacité photovoltaïque raccordée sur le périmètre Oiken **croît d'année en année** :

    - max PV observé **été 2024** (train) : charge nette ≈ −1.52
    - max PV observé **été 2025** (test)  : charge nette ≈ −1.94

    → Le modèle v3 **sature** en milieu de journée d'été et sous-estime systématiquement l'effet PV.
    → v4 introduit une **normalisation de la charge par un scaler PV** pour ramener la cible dans une plage stationnaire.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Le scaler PV

    ### Principe en 6 étapes

    1. Filtrer les pas horaires **10 h – 14 h** (pic solaire)
    2. Par mois : **quantile 5 %** de la charge nette → proxy du pic PV
    3. **Inverser le signe** (le signal croît avec la capacité PV)
    4. **Lissage** sur 3 mois glissants (filtre la variabilité météo mensuelle)
    5. **Ancrage à 1.0** sur le premier mois → ratio relatif
    6. Clip à 0.1 minimum (protection contre la division par zéro)

    > Pourquoi pas Pronovo ou VESE ? Pas d'historique mensuel par gestionnaire de réseau en open data.
    > Le proxy interne capte en plus **l'autoconsommation non déclarée**.
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, PV_SCALER_PATH, mdates, pd, pl, plt):
    def _build_scaler_figure():
        df = pl.read_parquet(PV_SCALER_PATH).to_pandas()
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(
                dict(year=df["year"], month=df["month"], day=15)
            )

        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(
            df["timestamp"], df["pv_scaler"],
            color=COLORS["scaler"], linewidth=2.2, marker="o", markersize=4,
            label="pv_scaler (lissé 3 mois, ancré à 1.0)",
        )
        if "raw_signal" in df.columns:
            ax.plot(
                df["timestamp"],
                df["raw_signal"] / df["raw_signal"].iloc[0],
                color=COLORS["scaler"], linewidth=1.0, linestyle="--",
                alpha=0.45, label="signal brut (−Q05 charge, normalisé)",
            )

        ax.axhline(1.0, color="#888", linewidth=0.7, linestyle=":")
        ax.set_title("Évolution du proxy de capacité PV — sous-groupe Oiken", loc="left")
        ax.set_ylabel("Multiplicateur relatif (base = 1er mois)")
        ax.set_xlabel("")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(loc="upper left")

        first_v = float(df["pv_scaler"].iloc[0])
        last_v = float(df["pv_scaler"].iloc[-1])
        growth = (last_v / first_v - 1) * 100
        ax.annotate(
            f"+{growth:.0f} % sur la période",
            xy=(df["timestamp"].iloc[-1], last_v),
            xytext=(-130, -18), textcoords="offset points",
            fontsize=9.5, color=COLORS["accent"], fontweight="semibold",
        )
        plt.tight_layout()
        return fig, df, first_v, last_v, growth

    fig_scaler, scaler_df, scaler_first, scaler_last, scaler_growth = _build_scaler_figure()
    fig_scaler
    return scaler_df, scaler_first, scaler_growth, scaler_last


@app.cell(hide_code=True)
def _(mo, scaler_df, scaler_first, scaler_growth, scaler_last):
    mo.md(f"""
    **Lecture** — Le multiplicateur passe de **{scaler_first:.2f}** à **{scaler_last:.2f}** sur {len(scaler_df)} mois,
    soit **+{scaler_growth:.0f} %** de capacité PV effective. C'est exactement cette dérive qui sort la cible
    de la plage d'entraînement en v3.

    **Anti-leakage** — le scaler du mois M est construit à partir des observations historiques du mois M,
    joint par année-mois à la cible. Pas d'information future injectée.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Les features du modèle v4

    **35 features** au total — toutes calculées sur la **charge normalisée** `load / pv_scaler`.
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, Patch, plt):
    def _build_features_figure():
        fam = [
            ("Lags charge\n(J-1, J-2, J-7)", 6),
            ("Rolling stats\n(24 h, 7 j)", 4),
            ("Lags PV local\n& remote", 4),
            ("Cyclique\n(h, wd, mois)", 6),
            ("Calendaire\n(WE, fériés, saisons)", 5),
            ("Météo réelle\n(lag J-1)", 3),
            ("NWP cible\n+24 h", 4),
            ("Interactions\n(T², T×rad)", 2),
            ("Scaler PV\n(courant + cible)", 2),
        ]
        feat_labels = [f[0] for f in fam]
        sizes = [f[1] for f in fam]
        feat_colors = [COLORS["oiken"]] * (len(fam) - 1) + [COLORS["xgb"]]

        fig, ax = plt.subplots(figsize=(10, 4.2))
        bars = ax.barh(range(len(fam)), sizes, color=feat_colors,
                       edgecolor="white", linewidth=0.8)
        ax.set_yticks(range(len(fam)))
        ax.set_yticklabels(feat_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Nombre de features")
        ax.set_title("Composition des 35 features · v4", loc="left")
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)

        for bar, size in zip(bars, sizes):
            ax.text(bar.get_width() + 0.12,
                    bar.get_y() + bar.get_height() / 2,
                    str(size), va="center", fontsize=9.5, color="#333")

        legend_items = [
            Patch(facecolor=COLORS["oiken"], label="Feature v3 (reconduite)"),
            Patch(facecolor=COLORS["xgb"], label="Ajout v4 (normalisation PV)"),
        ]
        ax.legend(handles=legend_items, loc="lower right")
        ax.set_xlim(0, max(sizes) + 1.8)
        plt.tight_layout()
        return fig

    fig_feat = _build_features_figure()
    fig_feat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cible & dénormalisation

    $$\text{target\_norm} = \frac{\text{load}[t+24\text{h}]}{\text{pv\_scaler}[t+24\text{h}]}$$

    Le modèle apprend sur `target_norm`. Les prédictions sont **dénormalisées** avant toute métrique :

    $$\widehat{\text{load}}[t+24\text{h}] = \widehat{\text{target\_norm}} \times \text{pv\_scaler}[t+24\text{h}]$$

    **Ce qui garantit la comparabilité avec Oiken**, qui reste exprimé dans l'espace brut.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Pipeline d'entraînement

    | Étape | Contenu |
    |---|---|
    | **Modèles** | XGBoost + LightGBM (comparés) |
    | **Loss** | MSE (`reg:squarederror` / `regression`) |
    | **Scoring CV** | MAE (`neg_mean_absolute_error`) |
    | **Recherche** | RandomizedSearchCV · 40 itérations |
    | **Validation** | TimeSeriesSplit · 5 folds |
    | **Split** | Chronologique **70 / 30** · jamais de shuffle |
    | **Train / Test** | ~73 000 / ~31 000 pas · oct 2022 → sept 2025 |

    Ordre d'exécution : `pv_scaler_v4.py` → `features_da_v4.py` → `model_da_v4.py`
    """)
    return


@app.cell(hide_code=True)
def _(PRED_V4_PATH, joblib, np, pd):
    def _load_predictions():
        pred = joblib.load(PRED_V4_PATH)

        def to_series(d, keys):
            for k in keys:
                if k in d:
                    return np.asarray(d[k])
            return None

        timestamps = to_series(pred, ["timestamp", "timestamps", "ts"])
        y_true_raw = to_series(pred, ["y_test_raw", "y_true_raw", "y_true", "y_test"])
        y_xgb_raw = to_series(pred, ["y_xgb_raw", "xgb_raw", "y_pred_xgb", "pred_xgb"])
        y_lgb_raw = to_series(pred, ["y_lgb_raw", "lgb_raw", "y_pred_lgb", "pred_lgb"])
        y_oiken = to_series(pred, ["forecast_load_target", "oiken", "y_oiken"])
        pv_scaler_tgt = to_series(pred, ["pv_scaler_target", "scaler_target"])

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps),
            "real": y_true_raw,
            "oiken": y_oiken,
            "xgb_v4": y_xgb_raw,
            "lgb_v4": y_lgb_raw,
            "pv_scaler_target": pv_scaler_tgt,
        }).sort_values("timestamp").reset_index(drop=True)

        # --- normaliser en datetime naïf (retire tz si présent) ---
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert(None)

        df["saison"] = df["timestamp"].apply(
            lambda ts: "hiver" if ts.month in (11, 12, 1, 2, 3) else "été"
        )
        df["err_xgb"] = df["xgb_v4"] - df["real"]
        df["err_oiken"] = df["oiken"] - df["real"]
        df["err_lgb"] = df["lgb_v4"] - df["real"]
        return df

    df_pred = _load_predictions()
    return (df_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Comportement saisonnier — v4 vs Oiken

    Trois semaines représentatives : **hiver**, **printemps**, **été**.
    Pour chaque semaine : charge réelle, Oiken, v4 (XGB + LGB) — **panneau inférieur : erreur absolue pas-à-pas**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    season_selector = mo.ui.dropdown(
        options={
            "Hiver · 13–19 janvier 2025": "2025-01-13",
            "Printemps · 14–20 avril 2025": "2025-04-14",
            "Été · 14–20 juillet 2025": "2025-07-14",
        },
        value="Hiver · 13–19 janvier 2025",
        label="**Semaine à afficher :**",
    )
    season_selector
    return (season_selector,)


@app.cell(hide_code=True)
def _(COLORS, df_pred, mdates, pd, plt, season_selector):
    def _build_season_week_figure(start_date, title):
        start = pd.Timestamp(start_date)
        end = start + pd.Timedelta(days=7)
        sub = df_pred[
            (df_pred["timestamp"] >= start) & (df_pred["timestamp"] < end)
        ].copy()

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1,
            figsize=(12, 6.5),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # --- panneau principal : courbes ---
        ax_top.plot(sub["timestamp"], sub["real"],
                    color=COLORS["real"], linewidth=1.9, label="Réel", zorder=5)
        ax_top.plot(sub["timestamp"], sub["oiken"],
                    color=COLORS["oiken"], linewidth=1.5, label="Oiken",
                    alpha=0.9, zorder=3)
        ax_top.plot(sub["timestamp"], sub["xgb_v4"],
                    color=COLORS["xgb"], linewidth=1.5, label="XGB v4",
                    alpha=0.9, zorder=4)
        ax_top.plot(sub["timestamp"], sub["lgb_v4"],
                    color=COLORS["lgb"], linewidth=1.3, label="LGB v4",
                    alpha=0.85, linestyle="--", zorder=4)

        ax_top.set_title(title, loc="left", fontsize=13.5)
        ax_top.set_ylabel("Charge nette (z-score)")
        ax_top.legend(loc="upper right", ncol=4, fontsize=11)

        # --- panneau inférieur : erreurs absolues ---
        ax_bot.plot(sub["timestamp"], (sub["oiken"] - sub["real"]).abs(),
                    color=COLORS["oiken"], linewidth=1.2,
                    label="|err| Oiken", alpha=0.9)
        ax_bot.plot(sub["timestamp"], (sub["xgb_v4"] - sub["real"]).abs(),
                    color=COLORS["xgb"], linewidth=1.2,
                    label="|err| XGB v4", alpha=0.9)
        ax_bot.set_ylabel("Erreur abs.")
        ax_bot.set_xlabel("")
        ax_bot.axhline(0, color="#999", linewidth=0.6)
        ax_bot.xaxis.set_major_locator(mdates.DayLocator())
        ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
        ax_bot.legend(loc="upper right", ncol=2, fontsize=10)

        plt.tight_layout()
        return fig

    # mapping libellé → date + titre
    season_map = {
        "2025-01-13": "Hiver · semaine 13-19 janvier 2025",
        "2025-04-14": "Printemps · semaine 14-20 avril 2025",
        "2025-07-14": "Été · semaine 14-20 juillet 2025",
    }
    selected_date = season_selector.value
    fig_seasons = _build_season_week_figure(
        selected_date, season_map[selected_date]
    )
    fig_seasons
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lecture des trois semaines

    - **Hiver** · régime sans PV · v4 un peu en dessous d'Oiken en MAE mais suit bien la dynamique
    - **Printemps** · transition · les creux de midi apparaissent, v4 les capture déjà mieux
    - **Été** · régime PV dominant · **v4 ne sature plus** sur les creux profonds, le biais systématique d'Oiken est visible

    L'enjeu de v4 se lit visuellement sur la semaine d'été : les creux négatifs sont plus serrés contre la courbe réelle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Résultats — MAE / biais / saisons

    Test set nettoyé (exclusion des 8 jours de prévisions Oiken figées en sept 2025), **n ≈ 30 700 pas**.
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, Patch, np, plt):
    def _build_mae_figure():
        mae_labels = ["Oiken", "XGB v4", "LGB v4"]
        mae_global_v = [0.2186, 0.2269, 0.2243]
        mae_hiver_v = [0.2663, 0.2276, 0.2221]
        mae_ete_v = [0.1790, 0.2262, 0.2262]
        mae_colors = [COLORS["oiken"], COLORS["xgb"], COLORS["lgb"]]

        x = np.arange(len(mae_labels))
        w = 0.26

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x - w, mae_hiver_v, w, color=mae_colors,
               alpha=0.55, edgecolor="white")
        ax.bar(x, mae_global_v, w, color=mae_colors, edgecolor="white")
        ax.bar(x + w, mae_ete_v, w, color=mae_colors,
               alpha=0.85, edgecolor="white", hatch="//")

        for xi, v in zip(x, mae_global_v):
            ax.text(xi, v + 0.005, f"{v:.4f}", ha="center",
                    fontsize=9.5, fontweight="semibold")

        ax.set_xticks(x)
        ax.set_xticklabels(mae_labels)
        ax.set_ylabel("MAE (z-score)")
        ax.set_title("MAE globale et saisonnière · v4 vs Oiken", loc="left")
        ax.set_ylim(0, 0.32)

        mae_legend = [
            Patch(facecolor="#777", alpha=0.55, label="Hiver (nov–mars)"),
            Patch(facecolor="#777", label="Global"),
            Patch(facecolor="#777", alpha=0.85, hatch="//", label="Été (avr–oct)"),
        ]
        ax.legend(handles=mae_legend, loc="upper right")
        ax.grid(axis="x", visible=False)
        plt.tight_layout()
        return fig

    fig_mae = _build_mae_figure()
    fig_mae
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Ce que montre la figure**

    - **Hiver** : XGB v4 et LGB v4 **battent Oiken de ~15 %** (0.222–0.228 contre 0.266)
    - **Été** : Oiken reste **le meilleur** (0.179), v4 à 0.226 — le scaler mensuel est trop lent pour coller à la variabilité intra-mensuelle
    - **Global** : v4 termine à **−3 % d'Oiken**, mais pour les bonnes raisons (biais estival corrigé)
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, df_pred, mdates, plt):
    def _build_bias_figure():
        df = df_pred.copy()
        df["mois"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
        agg = df.groupby("mois").agg(
            bias_oiken=("err_oiken", "mean"),
            bias_xgb=("err_xgb", "mean"),
            bias_lgb=("err_lgb", "mean"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.axhline(0, color="#444", linewidth=0.9)
        ax.fill_between(agg["mois"], 0, agg["bias_oiken"],
                        color=COLORS["oiken"], alpha=0.15)
        ax.plot(agg["mois"], agg["bias_oiken"],
                color=COLORS["oiken"], linewidth=1.8, marker="o",
                markersize=3.5, label="Oiken")
        ax.plot(agg["mois"], agg["bias_xgb"],
                color=COLORS["xgb"], linewidth=1.8, marker="s",
                markersize=3.5, label="XGB v4")
        ax.plot(agg["mois"], agg["bias_lgb"],
                color=COLORS["lgb"], linewidth=1.4, marker="^",
                markersize=3.5, linestyle="--", label="LGB v4", alpha=0.85)

        ax.set_title(
            "Biais mensuel moyen (prédiction − réel) · v4 corrige la dérive estivale",
            loc="left",
        )
        ax.set_ylabel("Biais (z-score)")
        ax.legend(loc="upper right", ncol=3)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig

    fig_bias = _build_bias_figure()
    fig_bias
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lecture du biais mensuel

    - La courbe **Oiken** (bleu marine) plonge en-dessous de zéro sur les étés → **sous-estimation systématique** (prédit plus de soutirage que réel, manque l'effet PV)
    - La courbe **XGB v4** (orange) reste centrée autour de zéro **toute l'année**
    - Concrètement : **le biais estival mensuel passe de ~+0.17 (v3) à −0.10 (v4)** — la sous-estimation est corrigée
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Limites & perspectives

    ### Ce que v4 résout
    - Le modèle **ne sature plus** sur les creux PV profonds hors plage d'entraînement
    - Le **biais estival systématique est corrigé** (+0.17 → −0.10)
    - L'architecture pose les bases d'une robustesse au concept drift

    ### Ce que v4 ne résout pas
    - **Régression en hiver** (−21 % de MAE vs v3) : le scaler mensuel injecte du bruit là où il n'y a pas de PV à normaliser
    - Le **scaler mensuel est trop lent** pour capturer les variations intra-mensuelles de production

    ### Pistes suivantes
    1. **Scaler bi-mensuel ou hebdomadaire** avec projection tendance des 6 derniers mois
    2. **Architecture charge brute + PV séparés** (Modèle 1 = consommation stable, Modèle 2 = production, sortie = M1 − M2)
    3. **Ensemble adaptatif** : ML en hiver (v3 ou v4), Oiken en été, poids appris par saison
    4. **Réentraînement glissant** avec sample weights temporels (demi-vie ~1 an)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    **v4 n'est pas un remplacement de v3** — c'est une **preuve de concept** d'une architecture robuste au concept drift PV.

    Apports :

    - **Diagnostic** du saturation-effet sur les modèles à base d'arbres face à la croissance PV
    - **Méthode de normalisation reproductible**, basée uniquement sur les données Oiken
    - **Biais estival corrigé**, fondation posée pour itérations futures

    Le chemin critique pour une v5 productive : un scaler plus réactif **+** une séparation charge/PV.

    ---

    *Merci — questions ?*
    """)
    return


if __name__ == "__main__":
    app.run()
