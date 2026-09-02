"""
AI Sales Forecasting
J2.6 - Validation finale du dataset synthétique.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data/raw/kshop_sales_synthetic.csv")

EXPECTED_ROWS = 5110
EXPECTED_PRODUCTS = 14
EXPECTED_DAYS = 365

EXPECTED_START_DATE = pd.Timestamp("2025-09-01")
EXPECTED_END_DATE = pd.Timestamp("2026-08-31")

REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "revenue",
]


# ============================================================
# UTILITAIRES
# ============================================================

def check(condition: bool, message: str) -> None:
    """Affiche le résultat d'un contrôle."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")

    if not condition:
        raise AssertionError(message)


# ============================================================
# VALIDATION
# ============================================================

def main():
    print("=" * 70)
    print("AI SALES FORECASTING — J2.6 VALIDATION FINALE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Vérification du fichier
    # --------------------------------------------------------

    print("\n[1] FICHIER")
    print("-" * 70)

    check(
        DATA_FILE.exists(),
        f"Fichier présent : {DATA_FILE}"
    )

    df = pd.read_csv(DATA_FILE)

    print(f"Fichier : {DATA_FILE}")
    print(f"Taille  : {DATA_FILE.stat().st_size:,} octets")

    # --------------------------------------------------------
    # 2. Structure
    # --------------------------------------------------------

    print("\n[2] STRUCTURE")
    print("-" * 70)

    check(
        len(df) == EXPECTED_ROWS,
        f"Nombre de lignes = {len(df)} (attendu : {EXPECTED_ROWS})"
    )

    check(
        list(df.columns) == REQUIRED_COLUMNS,
        "Colonnes conformes"
    )

    print(f"Colonnes : {list(df.columns)}")

    # --------------------------------------------------------
    # 3. Types
    # --------------------------------------------------------

    print("\n[3] TYPES")
    print("-" * 70)

    df["date"] = pd.to_datetime(df["date"], errors="raise")

    check(
        pd.api.types.is_integer_dtype(df["product_id"]),
        "product_id est entier"
    )

    check(
        pd.api.types.is_integer_dtype(df["quantity"]),
        "quantity est entier"
    )

    check(
        pd.api.types.is_numeric_dtype(df["unit_price"]),
        "unit_price est numérique"
    )

    check(
        pd.api.types.is_numeric_dtype(df["revenue"]),
        "revenue est numérique"
    )

    print(df.dtypes)

    # --------------------------------------------------------
    # 4. Dates
    # --------------------------------------------------------

    print("\n[4] DATES")
    print("-" * 70)

    min_date = df["date"].min()
    max_date = df["date"].max()

    unique_days = df["date"].nunique()

    check(
        min_date == EXPECTED_START_DATE,
        f"Date minimale = {min_date.date()}"
    )

    check(
        max_date == EXPECTED_END_DATE,
        f"Date maximale = {max_date.date()}"
    )

    check(
        unique_days == EXPECTED_DAYS,
        f"Nombre de jours uniques = {unique_days}"
    )

    # --------------------------------------------------------
    # 5. Unicité date + produit
    # --------------------------------------------------------

    print("\n[5] UNICITE")
    print("-" * 70)

    duplicate_count = df.duplicated(
        subset=["date", "product_id"]
    ).sum()

    check(
        duplicate_count == 0,
        f"Doublons date + produit = {duplicate_count}"
    )

    # --------------------------------------------------------
    # 6. Produits
    # --------------------------------------------------------

    print("\n[6] PRODUITS")
    print("-" * 70)

    product_count = df["product_id"].nunique()

    check(
        product_count == EXPECTED_PRODUCTS,
        f"Nombre de produits = {product_count}"
    )

    price_per_product = df.groupby("product_id")["unit_price"].nunique()

    check(
        (price_per_product == 1).all(),
        "Chaque produit possède un prix unique"
    )

    category_per_product = df.groupby("product_id")["category"].nunique()

    check(
        (category_per_product == 1).all(),
        "Chaque produit possède une catégorie unique"
    )

    # --------------------------------------------------------
    # 7. Valeurs NULL
    # --------------------------------------------------------

    print("\n[7] VALEURS MANQUANTES")
    print("-" * 70)

    null_count = int(df.isnull().sum().sum())

    check(
        null_count == 0,
        f"Nombre total de valeurs NULL = {null_count}"
    )

    # --------------------------------------------------------
    # 8. Valeurs négatives
    # --------------------------------------------------------

    print("\n[8] VALEURS METIER")
    print("-" * 70)

    check(
        (df["quantity"] >= 0).all(),
        "Toutes les quantités sont >= 0"
    )

    check(
        (df["unit_price"] > 0).all(),
        "Tous les prix sont > 0"
    )

    check(
        (df["revenue"] >= 0).all(),
        "Tous les revenus sont >= 0"
    )

    # --------------------------------------------------------
    # 9. Cohérence revenue
    # --------------------------------------------------------

    print("\n[9] COHERENCE FINANCIERE")
    print("-" * 70)

    expected_revenue = df["quantity"] * df["unit_price"]

    revenue_ok = (
        df["revenue"].round(2)
        == expected_revenue.round(2)
    ).all()

    check(
        revenue_ok,
        "revenue = quantity × unit_price"
    )

    # --------------------------------------------------------
    # 10. Valeurs extrêmes
    # --------------------------------------------------------

    print("\n[10] VALEURS EXTREMES")
    print("-" * 70)

    quantity_q99 = df["quantity"].quantile(0.99)

    extreme_rows = df[
        df["quantity"] > quantity_q99
    ]

    print(
        f"99e percentile quantité : "
        f"{quantity_q99:.2f}"
    )

    print(
        f"Observations au-dessus du P99 : "
        f"{len(extreme_rows)}"
    )

    if not extreme_rows.empty:
        print("\nExemples :")
        print(
            extreme_rows[
                [
                    "date",
                    "product_id",
                    "product_name",
                    "quantity",
                    "unit_price",
                    "revenue",
                ]
            ]
            .sort_values("quantity", ascending=False)
            .head(10)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 11. Résumé final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESUME FINAL J2.6")
    print("=" * 70)

    print(f"Lignes             : {len(df):,}")
    print(f"Colonnes           : {len(df.columns)}")
    print(f"Produits           : {df['product_id'].nunique()}")
    print(f"Jours              : {df['date'].nunique()}")
    print(f"Date début         : {min_date.date()}")
    print(f"Date fin           : {max_date.date()}")
    print(f"Valeurs NULL       : {null_count}")
    print(f"Doublons           : {duplicate_count}")
    print(f"Quantité totale    : {df['quantity'].sum():,}")
    print(f"CA total           : {df['revenue'].sum():,.0f} AR")

    print("\n" + "=" * 70)
    print("J2.6 — VALIDATION FINALE : PASS")
    print("J2 — DATASET : VALIDÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()