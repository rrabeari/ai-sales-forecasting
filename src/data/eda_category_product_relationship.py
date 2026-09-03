"""
AI Sales Forecasting
J4.4.3 - Relations catégories <-> produits
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_clean.csv"


def load_data():
    """Charger le dataset nettoyé."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    return pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )


def analyze_category_product(df):
    """Analyser les relations entre catégories et produits."""

    product = (
        df.groupby(
            [
                "product_id",
                "product_name",
                "category",
                "unit_price",
            ],
            as_index=False,
        )
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
        )
    )

    # Totaux par catégorie.
    category_totals = (
        product.groupby(
            "category",
            as_index=False,
        )
        .agg(
            category_quantity=("total_quantity", "sum"),
            category_revenue=("total_revenue", "sum"),
            product_count=("product_id", "nunique"),
        )
    )

    product = product.merge(
        category_totals,
        on="category",
        how="left",
    )

    # Contribution du produit à sa catégorie.
    product["category_quantity_share"] = (
        product["total_quantity"]
        / product["category_quantity"]
        * 100
    )

    product["category_revenue_share"] = (
        product["total_revenue"]
        / product["category_revenue"]
        * 100
    )

    # Rang à l'intérieur de la catégorie.
    product["category_volume_rank"] = (
        product.groupby("category")[
            "total_quantity"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    product["category_revenue_rank"] = (
        product.groupby("category")[
            "total_revenue"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # Indicateur de domination :
    # part du CA du produit dans sa catégorie.
    product["revenue_dominance"] = (
        product["category_revenue_share"]
        >= 50
    )

    # Indicateur de domination du volume.
    product["volume_dominance"] = (
        product["category_quantity_share"]
        >= 50
    )

    return product, category_totals


def calculate_category_concentration(product):
    """
    Calculer la concentration du CA par catégorie.

    HHI = somme des parts de marché au carré.
    Les parts sont exprimées en proportion (0 à 1).

    Plus le HHI est élevé, plus la catégorie dépend
    d'un petit nombre de produits.
    """

    concentration = (
        product.assign(
            revenue_share_decimal=
            product["category_revenue_share"] / 100
        )
        .groupby("category")
        .agg(
            hhi=(
                "revenue_share_decimal",
                lambda x: (x ** 2).sum(),
            )
        )
        .reset_index()
    )

    concentration["hhi_percent"] = (
        concentration["hhi"] * 100
    )

    def classify_concentration(hhi):
        if hhi >= 0.50:
            return "Très concentrée"
        if hhi >= 0.33:
            return "Concentrée"
        if hhi >= 0.20:
            return "Modérément concentrée"
        return "Diversifiée"

    concentration["concentration_profile"] = (
        concentration["hhi"].apply(
            classify_concentration
        )
    )

    return concentration


def validate_results(product, category_totals, df):
    """Valider les résultats."""

    expected_products = df["product_id"].nunique()
    expected_categories = df["category"].nunique()

    assert len(product) == expected_products
    assert len(category_totals) == expected_categories

    assert (
        product["total_quantity"].sum()
        == df["quantity"].sum()
    )

    assert (
        product["total_revenue"].sum()
        == df["revenue"].sum()
    )

    # Chaque catégorie doit totaliser 100 %.
    quantity_share_check = (
        product.groupby("category")[
            "category_quantity_share"
        ]
        .sum()
    )

    revenue_share_check = (
        product.groupby("category")[
            "category_revenue_share"
        ]
        .sum()
    )

    assert (
        quantity_share_check.sub(100).abs() < 0.01
    ).all()

    assert (
        revenue_share_check.sub(100).abs() < 0.01
    ).all()

    assert (
        product["category_quantity_share"]
        .between(0, 100)
    ).all()

    assert (
        product["category_revenue_share"]
        .between(0, 100)
    ).all()

    print("[PASS] Nombre de produits cohérent")
    print("[PASS] Nombre de catégories cohérent")
    print("[PASS] Quantité totale cohérente")
    print("[PASS] CA total cohérent")
    print("[PASS] Parts de volume par catégorie = 100 %")
    print("[PASS] Parts de CA par catégorie = 100 %")
    print("[PASS] Parts de contribution valides")


def print_report(
    product,
    category_totals,
    concentration,
):
    """Afficher le rapport."""

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        240,
    )

    pd.set_option(
        "display.float_format",
        "{:.2f}".format,
    )

    print("\n=== CONTRIBUTION DES PRODUITS À LEUR CATÉGORIE ===")

    columns = [
        "category",
        "product_name",
        "unit_price",
        "total_quantity",
        "category_quantity_share",
        "category_volume_rank",
        "total_revenue",
        "category_revenue_share",
        "category_revenue_rank",
        "volume_dominance",
        "revenue_dominance",
    ]

    report = product.sort_values(
        [
            "category",
            "total_revenue",
        ],
        ascending=[
            True,
            False,
        ],
    )

    print(
        report[columns].to_string(
            index=False
        )
    )

    print("\n=== PRODUIT DOMINANT EN CA PAR CATÉGORIE ===")

    top_revenue = (
        product.sort_values(
            [
                "category",
                "total_revenue",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "category",
            as_index=False,
        )
        .first()
    )

    for _, row in top_revenue.iterrows():
        print(
            f"- {row['category']}: "
            f"{row['product_name']} → "
            f"{row['total_revenue']:,.0f} AR "
            f"({row['category_revenue_share']:.2f} %)"
        )

    print("\n=== PRODUIT DOMINANT EN VOLUME PAR CATÉGORIE ===")

    top_volume = (
        product.sort_values(
            [
                "category",
                "total_quantity",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "category",
            as_index=False,
        )
        .first()
    )

    for _, row in top_volume.iterrows():
        print(
            f"- {row['category']}: "
            f"{row['product_name']} → "
            f"{int(row['total_quantity'])} unités "
            f"({row['category_quantity_share']:.2f} %)"
        )

    print("\n=== CONCENTRATION DES CATÉGORIES ===")

    concentration_report = concentration[
        [
            "category",
            "hhi",
            "hhi_percent",
            "concentration_profile",
        ]
    ]

    print(
        concentration_report.to_string(
            index=False
        )
    )

    print("\n=== RÉSUMÉ DES CATÉGORIES ===")

    summary = category_totals.sort_values(
        "category_revenue",
        ascending=False,
    )

    for _, row in summary.iterrows():
        print(
            f"- {row['category']}: "
            f"{int(row['product_count'])} produit(s), "
            f"{int(row['category_quantity']):,} unités, "
            f"{row['category_revenue']:,.0f} AR"
        )


def main():
    print("AI Sales Forecasting")
    print("J4.4.3 - Relations catégories <-> produits")
    print("-" * 60)

    df = load_data()

    print(
        f"\nLignes analysées : "
        f"{len(df):,}"
    )

    print(
        f"Produits analysés : "
        f"{df['product_id'].nunique()}"
    )

    print(
        f"Catégories analysées : "
        f"{df['category'].nunique()}"
    )

    product, category_totals = (
        analyze_category_product(df)
    )

    concentration = (
        calculate_category_concentration(
            product
        )
    )

    validate_results(
        product,
        category_totals,
        df,
    )

    print_report(
        product,
        category_totals,
        concentration,
    )

    print("\n" + "=" * 60)
    print(
        "J4.4.3 — RELATIONS CATÉGORIES/PRODUITS : OK"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()