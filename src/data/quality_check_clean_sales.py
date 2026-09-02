"""
AI Sales Forecasting
J3.4 - Quality Control

Contrôle qualité indépendant du dataset nettoyé.
Le fichier RAW n'est jamais modifié.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_clean.csv"
)

EXPECTED_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "revenue",
]

EXPECTED_ROWS = 5110
EXPECTED_PRODUCTS = 14
EXPECTED_DAYS = 365

EXPECTED_START_DATE = pd.Timestamp("2025-09-01")
EXPECTED_END_DATE = pd.Timestamp("2026-08-31")


def check(condition: bool, message: str) -> None:
    """Affiche PASS ou FAIL pour un contrôle."""

    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        raise AssertionError(message)


def main() -> None:
    """Exécute tous les contrôles qualité."""

    print("AI Sales Forecasting")
    print("J3.4 - Quality Control")
    print("-" * 50)

    # -----------------------------------------------------------------------
    # 1. Vérification du fichier
    # -----------------------------------------------------------------------

    check(
        CLEAN_FILE.exists(),
        f"Fichier présent : {CLEAN_FILE}",
    )

    # -----------------------------------------------------------------------
    # 2. Chargement
    # -----------------------------------------------------------------------

    df = pd.read_csv(CLEAN_FILE)

    print()
    print("=== STRUCTURE ===")

    check(
        len(df) == EXPECTED_ROWS,
        f"Nombre de lignes = {len(df):,}",
    )

    check(
        list(df.columns) == EXPECTED_COLUMNS,
        "Colonnes conformes",
    )

    # -----------------------------------------------------------------------
    # 3. Types
    # -----------------------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    print()
    print("=== TYPES ===")

    check(
        pd.api.types.is_integer_dtype(df["product_id"]),
        "product_id est entier",
    )

    check(
        pd.api.types.is_integer_dtype(df["quantity"]),
        "quantity est entier",
    )

    check(
        pd.api.types.is_numeric_dtype(df["unit_price"]),
        "unit_price est numérique",
    )

    check(
        pd.api.types.is_numeric_dtype(df["revenue"]),
        "revenue est numérique",
    )

    # -----------------------------------------------------------------------
    # 4. Valeurs NULL
    # -----------------------------------------------------------------------

    print()
    print("=== VALEURS NULL ===")

    null_count = int(df.isna().sum().sum())

    check(
        null_count == 0,
        f"Valeurs NULL = {null_count}",
    )

    # -----------------------------------------------------------------------
    # 5. Dates
    # -----------------------------------------------------------------------

    print()
    print("=== DATES ===")

    min_date = df["date"].min()
    max_date = df["date"].max()
    unique_days = df["date"].nunique()

    check(
        min_date == EXPECTED_START_DATE,
        f"Date minimale = {min_date.date()}",
    )

    check(
        max_date == EXPECTED_END_DATE,
        f"Date maximale = {max_date.date()}",
    )

    check(
        unique_days == EXPECTED_DAYS,
        f"Nombre de jours uniques = {unique_days}",
    )

    # -----------------------------------------------------------------------
    # 6. Doublons
    # -----------------------------------------------------------------------

    print()
    print("=== DOUBLONS ===")

    duplicate_count = int(
        df.duplicated(
            subset=["date", "product_id"]
        ).sum()
    )

    check(
        duplicate_count == 0,
        f"Doublons date + produit = {duplicate_count}",
    )

    # -----------------------------------------------------------------------
    # 7. Produits
    # -----------------------------------------------------------------------

    print()
    print("=== PRODUITS ===")

    product_count = df["product_id"].nunique()

    check(
        product_count == EXPECTED_PRODUCTS,
        f"Nombre de produits = {product_count}",
    )

    # -----------------------------------------------------------------------
    # 8. Quantité
    # -----------------------------------------------------------------------

    print()
    print("=== QUANTITÉ ===")

    negative_quantity = int(
        (df["quantity"] < 0).sum()
    )

    check(
        negative_quantity == 0,
        f"Quantités négatives = {negative_quantity}",
    )

    # -----------------------------------------------------------------------
    # 9. Prix
    # -----------------------------------------------------------------------

    print()
    print("=== PRIX ===")

    invalid_price = int(
        (df["unit_price"] <= 0).sum()
    )

    check(
        invalid_price == 0,
        f"Prix invalides = {invalid_price}",
    )

    # -----------------------------------------------------------------------
    # 10. Revenue
    # -----------------------------------------------------------------------

    print()
    print("=== REVENUE ===")

    invalid_revenue = int(
        (df["revenue"] < 0).sum()
    )

    check(
        invalid_revenue == 0,
        f"Revenus négatifs = {invalid_revenue}",
    )

    expected_revenue = (
        df["quantity"] * df["unit_price"]
    )

    revenue_mismatch = int(
        (df["revenue"] != expected_revenue).sum()
    )

    check(
        revenue_mismatch == 0,
        f"Erreurs revenue = {revenue_mismatch}",
    )

    # -----------------------------------------------------------------------
    # 11. Cohérence produit
    # -----------------------------------------------------------------------

    print()
    print("=== COHÉRENCE PRODUIT ===")

    product_price_counts = (
        df.groupby("product_id")["unit_price"]
        .nunique()
    )

    check(
        product_price_counts.max() == 1,
        "Chaque produit possède un prix unique",
    )

    product_category_counts = (
        df.groupby("product_id")["category"]
        .nunique()
    )

    check(
        product_category_counts.max() == 1,
        "Chaque produit possède une catégorie unique",
    )

    # -----------------------------------------------------------------------
    # 12. Cohérence temporelle
    # -----------------------------------------------------------------------

    print()
    print("=== COHÉRENCE TEMPORELLE ===")

    expected_rows = EXPECTED_PRODUCTS * EXPECTED_DAYS

    check(
        len(df) == expected_rows,
        f"Produits × jours = {EXPECTED_PRODUCTS} × "
        f"{EXPECTED_DAYS} = {expected_rows:,}",
    )

    # -----------------------------------------------------------------------
    # 13. Résumé statistique
    # -----------------------------------------------------------------------

    print()
    print("=== RÉSUMÉ ===")

    print(f"Lignes              : {len(df):,}")
    print(f"Produits            : {product_count}")
    print(f"Jours               : {unique_days}")
    print(f"Quantité totale     : {df['quantity'].sum():,}")
    print(f"CA total            : {df['revenue'].sum():,} AR")
    print(f"Quantité maximale   : {df['quantity'].max():,}")
    print(f"Revenu maximal      : {df['revenue'].max():,} AR")

    # -----------------------------------------------------------------------
    # Validation finale
    # -----------------------------------------------------------------------

    print()
    print("=" * 50)
    print("J3.4 — CONTRÔLE QUALITÉ : PASS")
    print("=" * 50)


if __name__ == "__main__":
    main()