# Oiken Load Prediction — Guide de reprise v3

## Diagnostic des bugs v2 (résolu)

### Bug 1 — Grille temporelle trouée (CRITIQUE)
Les sources brutes sont parfaitement régulières (105 120 pas à 15 min, 100 %).  
Mais `features_da.py` filtrait toutes les lignes où un lag ou une NWP était null :
```python
df = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in check]))
```
Résultat : **seulement 36 pas/jour en moyenne** au lieu de 96. Aucun jour complet.

### Bug 2 — Shifts par ligne au lieu de shifts temporels (CRITIQUE)
`pl.col("load").shift(96)` décale de 96 **lignes**, pas de 96 pas de 15 min.  
Avec 14 % de trous, `load_lag_1d` pointait vers **~50 h en arrière** au lieu de 24 h.  
Tous les lags (charge, PV, météo) et rolling stats étaient corrompus.

### Bug 3 — Benchmark Oiken mal aligné
`forecast_load[t]` = prévision Oiken **pour** le timestamp `t`.  
Le benchmark comparait `forecast_load[t]` vs `target = load[t+96]` → deux moments différents.  
MAE Oiken rapportée : 0.238 (gonflée). MAE réelle : **0.231** (correctement alignée).

---

## Résultats v3 corrigés — Day-Ahead

| Modèle     | MAE    | RMSE   | MAPE   | vs Oiken |
|------------|--------|--------|--------|----------|
| Naïf J-1   | 0.3334 | 0.5121 | 52.3 % | −44.5 %  |
| **Oiken**  | 0.2307 | 0.3488 | 37.1 % | ref      |
| XGBoost    | 0.2178 | 0.3245 | 30.5 % | +5.6 %   |
| LightGBM   | 0.2136 | 0.3178 | 30.0 % | +7.4 %   |

---

## Fichiers fournis

| Fichier | Rôle |
|---------|------|
| `features_da_v3.py` | Pipeline features DA corrigé (shifts temporels, NWP ffill, benchmark aligné) |
| `model_da_v3.py` | Entraînement XGBoost + LightGBM DA |
| `features_intraday_v3.py` | Pipeline features Intraday corrigé (À ADAPTER — template fourni) |
| `model_intraday_v3.py` | Entraînement Intraday multi-horizon (À ADAPTER) |
| `visualize_da.py` | Visualisation réel vs Oiken vs modèles |

---

## Étapes pour exécuter localement

### 1. Prérequis
```bash
pip install polars xgboost lightgbm scikit-learn joblib matplotlib
```

### 2. Placer les fichiers
Copier les 5 fichiers `.py` dans `00-data/`.  
Les données clean doivent exister :
- `data/processed/oiken_clean_v2.parquet`
- `data/processed/meteo_real_clean.parquet`
- `data/processed/meteo_pred_clean.parquet`

### 3. Exécuter le pipeline DA
```bash
cd 00-data
python features_da_v3.py      # → data/features_v3/train_da_v3.parquet + test_da_v3.parquet
python model_da_v3.py          # → models_saved/xgb_da_v3.joblib + lgb_da_v3.joblib + métriques
python visualize_da.py         # → figures/da_predictions_winter.png + da_predictions_summer.png
```

### 4. Exécuter le pipeline Intraday
```bash
python features_intraday_v3.py  # → data/features_v3/train_intraday_v3.parquet + test_intraday_v3.parquet
python model_intraday_v3.py     # → models_saved/xgb_id_v3_h1..h12.joblib + lgb_id_v3_h1..h12.joblib
```

---

## Changements clés v2 → v3

### Shifts temporels (le fix principal)
```python
# AVANT (v2) — FAUX avec trous
pl.col("load").shift(96)  # décale de 96 LIGNES

# APRÈS (v3) — CORRECT
def temporal_shift(df, col, delta, alias):
    lookup = df.select([
        (pl.col("timestamp") + delta).alias("timestamp"),
        pl.col(col).alias(alias),
    ])
    return df.join(lookup, on="timestamp", how="left")

temporal_shift(df, "load", timedelta(hours=24), "load_lag_1d")
```

### NWP forward-fill avant shift
```python
# Les NWP ont 30-60% de nulls (publiées toutes les 3h)
# Forward-fill AVANT le shift élimine 99.9% des nulls
df = df.with_columns([pl.col(c).forward_fill() for c in NWP_COLS])
```

### Benchmark Oiken aligné
```python
# AVANT (v2) — compare forecast_load[t] vs load[t+24h] → FAUX
y_oiken = df_test["forecast_load"]

# APRÈS (v3) — compare forecast_load[t+24h] vs load[t+24h] → CORRECT
temporal_shift_forward(df, "forecast_load", timedelta(hours=24), "forecast_load_target")
```

### Filtrage minimal au lieu de suppression agressive
```python
# AVANT (v2) — supprime toute ligne avec un null quelconque
df = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in check]))
# → 36 pas/jour, 14% de trous

# APRÈS (v3) — filtre uniquement target + lag J-1 + lag J-7, impute le reste
df = df.filter(pl.col("target").is_not_null())
df = df.filter(pl.col("load_lag_1d").is_not_null())
df = df.filter(pl.col("load_lag_7d").is_not_null())
# Imputation médiane pour les résidus (< 4% des lignes)
# → 96 pas/jour, grille complète
```

---

## Notes pour l'Intraday

Le même fix s'applique à `features_intraday_v3.py` :
- Shifts temporels au lieu de shifts par ligne
- NWP forward-fill
- Pour les lags courts PV local (t−1, t−4) : `shift(1)` et `shift(4)` restent corrects
  sur grille régulière, mais il vaut mieux utiliser `temporal_shift` par cohérence
- `build_nwp_for_horizon(df, h)` doit aussi utiliser `temporal_shift_forward`
  au lieu de `shift(-h)`
- Le benchmark Intraday = naïf `load_lag_96` (charge il y a 24h)

---

## Random Forest — Abandonné
RF trop gourmand en mémoire/temps pour 73 000 lignes × 33 features × 12 horizons.  
XGBoost et LightGBM suffisent pour la comparaison (boosting vs boosting, 2 implémentations).
