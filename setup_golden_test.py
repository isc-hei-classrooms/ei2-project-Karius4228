"""
setup_golden_test.py — Mise en place du dossier de test golden dataset (v2)
────────────────────────────────────────────────────────────────────────────
Usage :  python setup_golden_test.py
Auteur : Marius Fabbri
"""

import shutil, sys
from pathlib import Path

PROJECT_ROOT  = Path(r"C:\Users\mariu\OneDrive - HESSO\6_sem\Info\ei2-project-Karius4228")
SRC_PROCESSED = PROJECT_ROOT / "01-data" / "data" / "processed"
SRC_MODELS    = PROJECT_ROOT / "01-data" / "models_saved"
SRC_SCALERS   = PROJECT_ROOT / "01-data" / "scalers"
SRC_GOLDEN    = PROJECT_ROOT / "01-data" / "data" / "raw" / "oiken" / "oiken-golden-dataset.csv"

DST_ROOT      = PROJECT_ROOT / "03-data"
DST_PROCESSED = DST_ROOT / "data" / "processed"
DST_MODELS    = DST_ROOT / "models"
DST_RAW       = DST_ROOT / "data" / "raw"

PROCESSED_FILES = [
    "oiken_clean_v2.parquet",
    "meteo_real_clean.parquet",
    "meteo_pred_clean.parquet",
    "pv_scaler_v4.parquet",
]

# src -> (dst, required)
MODEL_FILES = {
    SRC_MODELS  / "xgb_da_v4.joblib"         : (DST_MODELS / "xgb_da_v4.joblib",         True),
    SRC_MODELS  / "lgb_da_v4.joblib"         : (DST_MODELS / "lgb_da_v4.joblib",         True),
    SRC_SCALERS / "medians_da.joblib"        : (DST_MODELS / "medians_da.joblib",        True),
    SRC_MODELS  / "da_v4_predictions.joblib" : (DST_MODELS / "da_v4_predictions.joblib", False),
}


def copy_file(src: Path, dst: Path, required: bool = True) -> bool:
    if not src.exists():
        print(f"  {'✗ MANQUANT (requis)' if required else '- Optionnel absent'} : {src.name}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"  ↩  Déjà présent    : {dst.name}")
        return True
    shutil.copy2(src, dst)
    print(f"  ✓ Copié ({dst.stat().st_size/1024:,.0f} KB) : {dst.name}")
    return True


def main():
    print("=" * 60)
    print("SETUP — Golden Test Dataset (v2)")
    print("=" * 60)
    print(f"Projet : {PROJECT_ROOT}")
    print(f"Dest.  : {DST_ROOT}\n")

    if not PROJECT_ROOT.exists():
        print(f"ERREUR : Projet introuvable : {PROJECT_ROOT}")
        sys.exit(1)

    for d in [DST_PROCESSED, DST_MODELS, DST_RAW]:
        d.mkdir(parents=True, exist_ok=True)
    print("✓ Structure créée\n")

    missing = []

    print("── Fichiers processed ──")
    for fname in PROCESSED_FILES:
        if not copy_file(SRC_PROCESSED / fname, DST_PROCESSED / fname):
            missing.append(fname)
    print()

    print("── Modèles & scalers ──")
    for src, (dst, req) in MODEL_FILES.items():
        if not copy_file(src, dst, required=req) and req:
            missing.append(src.name)
    print()

    print("── Golden dataset CSV ──")
    if not copy_file(SRC_GOLDEN, DST_RAW / "oiken-golden-dataset.csv"):
        missing.append("oiken-golden-dataset.csv")
    print()

    print("── Couverture temporelle ──")
    try:
        import polars as pl
        for fname in ["oiken_clean_v2.parquet", "meteo_real_clean.parquet", "meteo_pred_clean.parquet"]:
            p = DST_PROCESSED / fname
            if p.exists():
                df = pl.read_parquet(p, columns=["timestamp"])
                print(f"  {fname:<35} {str(df['timestamp'].min())[:10]} → {str(df['timestamp'].max())[:10]}  ({df.height:,} lignes)")
        p = DST_PROCESSED / "pv_scaler_v4.parquet"
        if p.exists():
            df_s = pl.read_parquet(p)
            r0, r1 = df_s.row(0, named=True), df_s.row(-1, named=True)
            print(f"  {'pv_scaler_v4.parquet':<35} {r0['year']}-{r0['month']:02d} → {r1['year']}-{r1['month']:02d}  ({df_s.height} mois)")
        p = DST_RAW / "oiken-golden-dataset.csv"
        if p.exists():
            import pandas as pd
            df_g = pd.read_csv(p)
            print(f"  {'oiken-golden-dataset.csv':<35} {df_g.shape[0]:,} lignes, {df_g.shape[1]} colonnes")
    except ImportError as e:
        print(f"  (import manquant : {e})")
    print()

    if missing:
        print("=" * 60)
        print(f"⚠  {len(missing)} fichier(s) manquant(s) :")
        for f in missing:
            print(f"   - {f}")
    else:
        print("=" * 60)
        print("✓ Setup complet — tous les fichiers sont en place.")
        print("\nProchaine étape : lancer build_golden_parquet.py")
    print("=" * 60)


if __name__ == "__main__":
    main()