"""
J6.1 — Préparation du split temporel pour le Machine Learning.

Le dataset est séparé chronologiquement afin d'éviter
toute fuite de données entre le passé et le futur.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/sales_ml_ready.csv")

OUTPUT_DIR = Path("data/processed/ml_split")

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

TRAIN_END = "2026-06-30"
VALIDATION_START = "2026-07-01"
VALIDATION_END = "2026-07-31"
TEST_START = "2026-08-01"

TARGET = "quantity"


# ============================================================
# FONCTIONS
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Charge et valide le dataset ML."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "date",
        "product_id",
        "quantity",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Des dates invalides ont été détectées.")

    df = df.sort_values(
        ["date", "product_id"]
    ).reset_index(drop=True)

    return df


def validate_temporal_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Valide l'intégrité du split temporel."""

    if train.empty:
        raise ValueError("Le dataset Train est vide.")

    if validation.empty:
        raise ValueError("Le dataset Validation est vide.")

    if test.empty:
        raise ValueError("Le dataset Test est vide.")

    # Vérification des périodes
    if train["date"].max() > pd.Timestamp(TRAIN_END):
        raise ValueError("Le Train contient des données après le 2026-06-30.")

    if validation["date"].min() < pd.Timestamp(VALIDATION_START):
        raise ValueError("La Validation contient des données avant le 2026-07-01.")

    if validation["date"].max() > pd.Timestamp(VALIDATION_END):
        raise ValueError("La Validation contient des données après le 2026-07-31.")

    if test["date"].min() < pd.Timestamp(TEST_START):
        raise ValueError("Le Test contient des données avant le 2026-08-01.")

    # Vérification de l'ordre temporel
    if train["date"].max() >= validation["date"].min():
        raise ValueError("Chevauchement temporel Train / Validation.")

    if validation["date"].max() >= test["date"].min():
        raise ValueError("Chevauchement temporel Validation / Test.")

    # Vérification du nombre de produits
    products_train = set(train["product_id"].unique())
    products_validation = set(validation["product_id"].unique())
    products_test = set(test["product_id"].unique())

    if products_train != products_validation:
        raise ValueError(
            "Les produits Train et Validation ne correspondent pas."
        )

    if products_train != products_test:
        raise ValueError(
            "Les produits Train et Test ne correspondent pas."
        )

    # Vérification de la target
    for name, dataset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        if dataset[TARGET].isna().any():
            raise ValueError(
                f"Target NULL détectée dans {name}."
            )

        if (dataset[TARGET] < 0).any():
            raise ValueError(
                f"Target négative détectée dans {name}."
            )

    # Vérification des clés
    for name, dataset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        duplicates = dataset.duplicated(
            subset=["date", "product_id"]
        ).sum()

        if duplicates > 0:
            raise ValueError(
                f"{duplicates} doublon(s) date + product_id dans {name}."
            )


def save_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Sauvegarde les trois datasets."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Exécute la préparation du split temporel."""

    print("=" * 60)
    print("J6.1 — SPLIT TEMPOREL")
    print("=" * 60)

    print("\n=== 1. CHARGEMENT ===")

    df = load_dataset()

    print(f"[OK] Dataset chargé : {INPUT_FILE}")
    print(f"[OK] Lignes : {len(df):,}")
    print(f"[OK] Colonnes : {len(df.columns)}")
    print(f"[OK] Période : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"[OK] Produits : {df['product_id'].nunique()}")

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train = df[
        df["date"] <= pd.Timestamp(TRAIN_END)
    ].copy()

    validation = df[
        (df["date"] >= pd.Timestamp(VALIDATION_START))
        & (df["date"] <= pd.Timestamp(VALIDATION_END))
    ].copy()

    test = df[
        df["date"] >= pd.Timestamp(TEST_START)
    ].copy()

    print("\n=== 2. SPLIT ===")

    print(
        f"[OK] Train      : {len(train):,} lignes"
    )

    print(
        f"[OK] Validation : {len(validation):,} lignes"
    )

    print(
        f"[OK] Test       : {len(test):,} lignes"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n=== 3. VALIDATION ===")

    validate_temporal_split(
        train,
        validation,
        test
    )

    print("[PASS] Aucun chevauchement temporel")
    print("[PASS] Train avant Validation")
    print("[PASS] Validation avant Test")
    print("[PASS] 14 produits présents dans chaque split")
    print("[PASS] Target quantity valide")
    print("[PASS] Aucune clé date + product_id dupliquée")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("\n=== 4. SAUVEGARDE ===")

    save_split(
        train,
        validation,
        test
    )

    print(f"[OK] {TRAIN_FILE}")
    print(f"[OK] {VALIDATION_FILE}")
    print(f"[OK] {TEST_FILE}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("J6.1 — RÉSUMÉ")
    print("=" * 60)

    print(
        f"Train      : {train['date'].min().date()} → "
        f"{train['date'].max().date()} "
        f"({len(train):,} lignes)"
    )

    print(
        f"Validation : {validation['date'].min().date()} → "
        f"{validation['date'].max().date()} "
        f"({len(validation):,} lignes)"
    )

    print(
        f"Test       : {test['date'].min().date()} → "
        f"{test['date'].max().date()} "
        f"({len(test):,} lignes)"
    )

    print("\nTarget :", TARGET)

    print("\n" + "=" * 60)
    print("J6.1 — SPLIT TEMPOREL : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()