"""
AI Sales Forecasting
J3.2 - Data Cleaning

Nettoie le dataset RAW sans jamais modifier le fichier original.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration des chemins
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "kshop_sales_synthetic.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "sales_clean.csv"


# Colonnes attendues dans le dataset
REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "revenue",
]


def load_data() -> pd.DataFrame:
    """Charge le dataset RAW."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset RAW introuvable : {RAW_FILE}"
        )

    return pd.read_csv(RAW_FILE)


def validate_columns(df: pd.DataFrame) -> None:
    """Vérifie la présence des colonnes nécessaires."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les règles de nettoyage définies en J3.1.
    """

    cleaned = df.copy()

    # -----------------------------------------------------------------------
    # 1. Nettoyage des espaces dans les colonnes texte
    # -----------------------------------------------------------------------

    text_columns = [
        "product_name",
        "category",
    ]

    for column in text_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    # -----------------------------------------------------------------------
    # 2. Conversion des types
    # -----------------------------------------------------------------------

    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        errors="coerce",
    )

    cleaned["product_id"] = pd.to_numeric(
        cleaned["product_id"],
        errors="coerce",
    )

    cleaned["quantity"] = pd.to_numeric(
        cleaned["quantity"],
        errors="coerce",
    )

    cleaned["unit_price"] = pd.to_numeric(
        cleaned["unit_price"],
        errors="coerce",
    )

    cleaned["revenue"] = pd.to_numeric(
        cleaned["revenue"],
        errors="coerce",
    )

    # -----------------------------------------------------------------------
    # 3. Suppression des lignes critiques invalides
    # -----------------------------------------------------------------------

    critical_columns = [
        "date",
        "product_id",
        "quantity",
        "unit_price",
    ]

    before_null_removal = len(cleaned)

    cleaned = cleaned.dropna(
        subset=critical_columns
    )

    removed_null_rows = before_null_removal - len(cleaned)

    # -----------------------------------------------------------------------
    # 4. Suppression des quantités invalides
    # -----------------------------------------------------------------------

    before_quantity_filter = len(cleaned)

    cleaned = cleaned[
        cleaned["quantity"] >= 0
    ]

    removed_invalid_quantity = (
        before_quantity_filter - len(cleaned)
    )

    # -----------------------------------------------------------------------
    # 5. Suppression des prix invalides
    # -----------------------------------------------------------------------

    before_price_filter = len(cleaned)

    cleaned = cleaned[
        cleaned["unit_price"] > 0
    ]

    removed_invalid_price = (
        before_price_filter - len(cleaned)
    )

    # -----------------------------------------------------------------------
    # 6. Suppression des doublons date + produit
    # -----------------------------------------------------------------------

    before_duplicates = len(cleaned)

    cleaned = cleaned.drop_duplicates(
        subset=["date", "product_id"],
        keep="first",
    )

    removed_duplicates = (
        before_duplicates - len(cleaned)
    )

    # -----------------------------------------------------------------------
    # 7. Recalcul du chiffre d'affaires
    # -----------------------------------------------------------------------

    cleaned["revenue"] = (
        cleaned["quantity"] * cleaned["unit_price"]
    )

    # -----------------------------------------------------------------------
    # 8. Conversion finale des types
    # -----------------------------------------------------------------------

    cleaned["product_id"] = cleaned["product_id"].astype("int64")

    cleaned["quantity"] = cleaned["quantity"].astype("int64")

    cleaned["unit_price"] = cleaned["unit_price"].astype("int64")

    cleaned["revenue"] = cleaned["revenue"].astype("int64")

    # -----------------------------------------------------------------------
    # 9. Tri chronologique
    # -----------------------------------------------------------------------

    cleaned = cleaned.sort_values(
        by=["date", "product_id"]
    ).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Rapport interne
    # -----------------------------------------------------------------------

    print()
    print("=== RAPPORT DE NETTOYAGE ===")
    print(f"Lignes initiales          : {len(df):,}")
    print(f"Lignes finales            : {len(cleaned):,}")
    print(f"NULL critiques supprimés : {removed_null_rows:,}")
    print(
        f"Quantités invalides      : "
        f"{removed_invalid_quantity:,}"
    )
    print(
        f"Prix invalides           : "
        f"{removed_invalid_price:,}"
    )
    print(
        f"Doublons supprimés       : "
        f"{removed_duplicates:,}"
    )

    return cleaned


def validate_clean_data(df: pd.DataFrame) -> None:
    """Effectue les contrôles essentiels après nettoyage."""

    assert list(df.columns) == REQUIRED_COLUMNS

    assert df["date"].notna().all()
    assert df["product_id"].notna().all()
    assert df["quantity"].notna().all()
    assert df["unit_price"].notna().all()
    assert df["revenue"].notna().all()

    assert (df["quantity"] >= 0).all()
    assert (df["unit_price"] > 0).all()
    assert (df["revenue"] >= 0).all()

    assert not df.duplicated(
        subset=["date", "product_id"]
    ).any()

    revenue_check = (
        df["revenue"]
        == df["quantity"] * df["unit_price"]
    )

    assert revenue_check.all()


def save_data(df: pd.DataFrame) -> None:
    """Sauvegarde le dataset nettoyé."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def main() -> None:
    """Point d'entrée principal."""

    print("AI Sales Forecasting")
    print("J3.2 - Data Cleaning")
    print("-" * 50)

    # Chargement
    df = load_data()

    # Vérification de la structure
    validate_columns(df)

    # Nettoyage
    cleaned_df = clean_data(df)

    # Validation
    validate_clean_data(cleaned_df)

    # Sauvegarde
    save_data(cleaned_df)

    print()
    print("=== VALIDATION ===")
    print("[PASS] Dataset nettoyé et validé")

    print()
    print("=== SORTIE ===")
    print(f"Lignes : {len(cleaned_df):,}")
    print(f"Fichier : {OUTPUT_FILE}")

    print()
    print("J3.2 — SCRIPT DE CLEANING : OK")


if __name__ == "__main__":
    main()