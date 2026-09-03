"""
AI Sales Forecasting
J4.2 - Analyse descriptive
J4.2.1 - Profil général
J4.2.2 - Statistiques de quantity
"""

from pathlib import Path

import pandas as pd


# Chemin du dataset nettoyé
DATA_PATH = Path("data/processed/sales_clean.csv")


def main():
    print("AI Sales Forecasting")
    print("J4.2.2 - Statistiques de quantity")
    print("-" * 50)

    # Vérification de l'existence du dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}"
        )

    # Chargement
    df = pd.read_csv(DATA_PATH)

    # Conversion de la date
    df["date"] = pd.to_datetime(df["date"])

    # Variable cible
    quantity = df["quantity"]

    print("\n=== STATISTIQUES GÉNÉRALES ===")
    print(f"Nombre d'observations : {len(quantity):,}")
    print(f"Quantité totale       : {quantity.sum():,}")
    print(f"Moyenne               : {quantity.mean():.2f}")
    print(f"Médiane               : {quantity.median():.2f}")
    print(f"Écart-type            : {quantity.std():.2f}")
    print(f"Minimum               : {quantity.min():,}")
    print(f"Maximum               : {quantity.max():,}")

    print("\n=== QUARTILES ===")
    print(f"Q1 (25%)              : {quantity.quantile(0.25):.2f}")
    print(f"Q2 (50%)              : {quantity.quantile(0.50):.2f}")
    print(f"Q3 (75%)              : {quantity.quantile(0.75):.2f}")

    print("\n=== PERCENTILES ===")
    print(f"P90                   : {quantity.quantile(0.90):.2f}")
    print(f"P95                   : {quantity.quantile(0.95):.2f}")
    print(f"P99                   : {quantity.quantile(0.99):.2f}")

    print("\n=== ZÉROS ===")
    zero_quantity = (quantity == 0).sum()
    zero_percentage = (zero_quantity / len(quantity)) * 100

    print(f"Observations à 0     : {zero_quantity:,}")
    print(f"Pourcentage à 0      : {zero_percentage:.2f}%")

    print("\n=== VALEURS ÉLEVÉES ===")
    p99 = quantity.quantile(0.99)
    above_p99 = (quantity > p99).sum()

    print(f"Seuil P99             : {p99:.2f}")
    print(f"Observations > P99    : {above_p99:,}")

    print("\n=== RÉSUMÉ ===")
    print(f"Quantité totale       : {quantity.sum():,}")
    print(f"Moyenne               : {quantity.mean():.2f}")
    print(f"Médiane               : {quantity.median():.2f}")
    print(f"Écart-type            : {quantity.std():.2f}")
    print(f"Minimum               : {quantity.min():,}")
    print(f"Maximum               : {quantity.max():,}")
    print(f"Observations à 0      : {zero_quantity:,}")
    print(f"Observations > P99    : {above_p99:,}")

    print("\n" + "=" * 50)
    print("J4.2.2 — STATISTIQUES QUANTITY : TERMINÉ")
    print("=" * 50)

        # ==========================================================
    # J4.2.3 - STATISTIQUES DE REVENUE
    # ==========================================================

    revenue = df["revenue"]

    print("\n=== STATISTIQUES REVENUE ===")
    print(f"Nombre d'observations : {len(revenue):,}")
    print(f"CA total              : {revenue.sum():,} AR")
    print(f"CA moyen              : {revenue.mean():,.2f} AR")
    print(f"CA médian             : {revenue.median():,.2f} AR")
    print(f"Écart-type             : {revenue.std():,.2f} AR")
    print(f"Minimum               : {revenue.min():,} AR")
    print(f"Maximum               : {revenue.max():,} AR")

    print("\n=== QUARTILES REVENUE ===")
    print(f"Q1 (25%)              : {revenue.quantile(0.25):,.2f} AR")
    print(f"Q2 (50%)              : {revenue.quantile(0.50):,.2f} AR")
    print(f"Q3 (75%)              : {revenue.quantile(0.75):,.2f} AR")

    print("\n=== PERCENTILES REVENUE ===")
    print(f"P90                   : {revenue.quantile(0.90):,.2f} AR")
    print(f"P95                   : {revenue.quantile(0.95):,.2f} AR")
    print(f"P99                   : {revenue.quantile(0.99):,.2f} AR")

    print("\n=== REVENUS NULS ===")
    zero_revenue = (revenue == 0).sum()
    zero_revenue_percentage = (zero_revenue / len(revenue)) * 100

    print(f"Observations à 0      : {zero_revenue:,}")
    print(f"Pourcentage à 0       : {zero_revenue_percentage:.2f}%")

    print("\n=== VALEURS ÉLEVÉES ===")
    p99_revenue = revenue.quantile(0.99)
    above_p99_revenue = (revenue > p99_revenue).sum()

    print(f"Seuil P99             : {p99_revenue:,.2f} AR")
    print(f"Observations > P99    : {above_p99_revenue:,}")

    print("\n" + "=" * 50)
    print("J4.2.3 — STATISTIQUES REVENUE : TERMINÉ")
    print("=" * 50)

        # ==========================================================
    # J4.2.4 - ANALYSE PAR PRODUIT
    # ==========================================================

    product_analysis = (
        df.groupby(
            ["product_id", "product_name"],
            as_index=False
        )
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            average_quantity=("quantity", "mean"),
            average_revenue=("revenue", "mean"),
            selling_days=("quantity", lambda x: (x > 0).sum()),
            zero_sales_days=("quantity", lambda x: (x == 0).sum()),
        )
    )

    # Arrondir les moyennes pour une lecture plus propre
    product_analysis["average_quantity"] = (
        product_analysis["average_quantity"].round(2)
    )

    product_analysis["average_revenue"] = (
        product_analysis["average_revenue"].round(2)
    )

    print("\n=== ANALYSE PAR PRODUIT ===")

    print("\nNombre de produits :", len(product_analysis))

    print("\n--- TOP 10 PAR QUANTITÉ ---")

    top_quantity = product_analysis.sort_values(
        by="total_quantity",
        ascending=False
    ).head(10)

    print(
        top_quantity[
            [
                "product_id",
                "product_name",
                "total_quantity",
                "average_quantity",
                "selling_days",
                "zero_sales_days",
            ]
        ].to_string(index=False)
    )

    print("\n--- TOP 10 PAR CHIFFRE D'AFFAIRES ---")

    top_revenue = product_analysis.sort_values(
        by="total_revenue",
        ascending=False
    ).head(10)

    print(
        top_revenue[
            [
                "product_id",
                "product_name",
                "total_revenue",
                "average_revenue",
                "selling_days",
                "zero_sales_days",
            ]
        ].to_string(index=False)
    )

    print("\n--- PRODUITS LES MOINS VENDUS ---")

    bottom_quantity = product_analysis.sort_values(
        by="total_quantity",
        ascending=True
    ).head(5)

    print(
        bottom_quantity[
            [
                "product_id",
                "product_name",
                "total_quantity",
                "total_revenue",
                "selling_days",
                "zero_sales_days",
            ]
        ].to_string(index=False)
    )

    print("\n--- RÉSUMÉ PRODUIT ---")

    best_quantity_product = product_analysis.loc[
        product_analysis["total_quantity"].idxmax()
    ]

    best_revenue_product = product_analysis.loc[
        product_analysis["total_revenue"].idxmax()
    ]

    print(
        f"Produit n°1 en quantité : "
        f"{best_quantity_product['product_name']} "
        f"({int(best_quantity_product['total_quantity']):,} unités)"
    )

    print(
        f"Produit n°1 en CA : "
        f"{best_revenue_product['product_name']} "
        f"({int(best_revenue_product['total_revenue']):,} AR)"
    )

    print("\n" + "=" * 50)
    print("J4.2.4 — ANALYSE PAR PRODUIT : TERMINÉE")
    print("=" * 50)

        # ==========================================================
    # J4.2.5 - ANALYSE PAR CATÉGORIE
    # ==========================================================

    category_analysis = (
        df.groupby("category", as_index=False)
        .agg(
            product_count=("product_id", "nunique"),
            total_quantity=("quantity", "sum"),
            average_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            average_revenue=("revenue", "mean"),
            selling_days=("quantity", lambda x: (x > 0).sum()),
            zero_sales_days=("quantity", lambda x: (x == 0).sum()),
        )
    )

    # Calcul des contributions au total
    total_quantity = df["quantity"].sum()
    total_revenue = df["revenue"].sum()

    category_analysis["quantity_share"] = (
        category_analysis["total_quantity"]
        / total_quantity
        * 100
    ).round(2)

    category_analysis["revenue_share"] = (
        category_analysis["total_revenue"]
        / total_revenue
        * 100
    ).round(2)

    category_analysis["average_quantity"] = (
        category_analysis["average_quantity"].round(2)
    )

    category_analysis["average_revenue"] = (
        category_analysis["average_revenue"].round(2)
    )

    print("\n=== ANALYSE PAR CATÉGORIE ===")

    print("\nNombre de catégories :", len(category_analysis))

    print("\n--- PERFORMANCE DES CATÉGORIES ---")

    category_display = category_analysis.sort_values(
        by="total_quantity",
        ascending=False
    )

    print(
        category_display[
            [
                "category",
                "product_count",
                "total_quantity",
                "quantity_share",
                "total_revenue",
                "revenue_share",
                "average_quantity",
                "average_revenue",
            ]
        ].to_string(index=False)
    )

    print("\n--- CLASSEMENT PAR QUANTITÉ ---")

    quantity_ranking = category_analysis.sort_values(
        by="total_quantity",
        ascending=False
    )

    for rank, (_, row) in enumerate(
        quantity_ranking.iterrows(),
        start=1
    ):
        print(
            f"{rank}. {row['category']} : "
            f"{int(row['total_quantity']):,} unités "
            f"({row['quantity_share']:.2f}%)"
        )

    print("\n--- CLASSEMENT PAR CHIFFRE D'AFFAIRES ---")

    revenue_ranking = category_analysis.sort_values(
        by="total_revenue",
        ascending=False
    )

    for rank, (_, row) in enumerate(
        revenue_ranking.iterrows(),
        start=1
    ):
        print(
            f"{rank}. {row['category']} : "
            f"{int(row['total_revenue']):,} AR "
            f"({row['revenue_share']:.2f}%)"
        )

    print("\n--- RÉSUMÉ CATÉGORIE ---")

    best_quantity_category = category_analysis.loc[
        category_analysis["total_quantity"].idxmax()
    ]

    best_revenue_category = category_analysis.loc[
        category_analysis["total_revenue"].idxmax()
    ]

    print(
        f"Catégorie n°1 en quantité : "
        f"{best_quantity_category['category']} "
        f"({int(best_quantity_category['total_quantity']):,} unités)"
    )

    print(
        f"Catégorie n°1 en CA : "
        f"{best_revenue_category['category']} "
        f"({int(best_revenue_category['total_revenue']):,} AR)"
    )

    print("\n" + "=" * 50)
    print("J4.2.5 — ANALYSE PAR CATÉGORIE : TERMINÉE")
    print("=" * 50)


if __name__ == "__main__":
    main()