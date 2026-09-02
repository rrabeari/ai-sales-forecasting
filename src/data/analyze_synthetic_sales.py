"""
Statistical analysis of the synthetic KShop sales dataset.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/raw/kshop_sales_synthetic.csv")


# ============================================================
# Main analysis
# ============================================================

def main() -> None:
    """Load and analyze the synthetic sales dataset."""

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    print("=" * 70)
    print("AI SALES FORECASTING — J2.5 ANALYSE STATISTIQUE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. General information
    # --------------------------------------------------------

    print("\n[1] INFORMATIONS GENERALES")
    print("-" * 70)

    print(f"Nombre de lignes       : {len(df):,}")
    print(f"Nombre de produits     : {df['product_id'].nunique()}")
    print(f"Nombre de catégories   : {df['category'].nunique()}")
    print(f"Date début             : {df['date'].min().date()}")
    print(f"Date fin               : {df['date'].max().date()}")

    # --------------------------------------------------------
    # 2. Total sales
    # --------------------------------------------------------

    print("\n[2] VENTES GLOBALES")
    print("-" * 70)

    total_quantity = df["quantity"].sum()
    total_revenue = df["revenue"].sum()

    print(f"Quantité totale vendue : {total_quantity:,.0f}")
    print(f"Chiffre d'affaires     : {total_revenue:,.0f} AR")
    print(f"Vente moyenne/jour     : {df.groupby('date')['quantity'].sum().mean():,.2f}")
    print(f"CA moyen/jour          : {df.groupby('date')['revenue'].sum().mean():,.2f} AR")

    # --------------------------------------------------------
    # 3. Sales by product
    # --------------------------------------------------------

    print("\n[3] VENTES PAR PRODUIT")
    print("-" * 70)

    product_sales = (
        df.groupby(["product_id", "product_name"], as_index=False)
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_values("quantity", ascending=False)
    )

    print(product_sales.to_string(index=False))

    # --------------------------------------------------------
    # 4. Revenue by category
    # --------------------------------------------------------

    print("\n[4] CA PAR CATEGORIE")
    print("-" * 70)

    category_sales = (
        df.groupby("category", as_index=False)
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )

    print(category_sales.to_string(index=False))

    # --------------------------------------------------------
    # 5. Monthly analysis
    # --------------------------------------------------------

    print("\n[5] VENTES PAR MOIS")
    print("-" * 70)

    monthly_sales = (
        df.assign(month=df["date"].dt.to_period("M"))
        .groupby("month", as_index=False)
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
    )

    print(monthly_sales.to_string(index=False))

    # --------------------------------------------------------
    # 6. Weekday analysis
    # --------------------------------------------------------

    print("\n[6] VENTES PAR JOUR DE LA SEMAINE")
    print("-" * 70)

    weekday_names = {
        0: "Lundi",
        1: "Mardi",
        2: "Mercredi",
        3: "Jeudi",
        4: "Vendredi",
        5: "Samedi",
        6: "Dimanche",
    }

    weekday_sales = (
        df.assign(
            weekday=df["date"].dt.dayofweek
        )
        .groupby("weekday", as_index=False)
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
    )

    weekday_sales["day_name"] = weekday_sales["weekday"].map(
        weekday_names
    )

    print(
        weekday_sales[
            ["weekday", "day_name", "quantity", "revenue"]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 7. Zero-sales observations
    # --------------------------------------------------------

    print("\n[7] JOURS SANS VENTE PAR PRODUIT")
    print("-" * 70)

    zero_sales = (
        df.groupby(["product_id", "product_name"])["quantity"]
        .apply(lambda x: (x == 0).sum())
        .reset_index(name="zero_sales_days")
        .sort_values("zero_sales_days", ascending=False)
    )

    print(zero_sales.to_string(index=False))

    # --------------------------------------------------------
    # 8. Highest demand
    # --------------------------------------------------------

    print("\n[8] PLUS FORTE DEMANDE")
    print("-" * 70)

    max_quantity = df["quantity"].max()

    print(f"Quantité maximale sur une observation : {max_quantity}")

    print(
        df[df["quantity"] == max_quantity][
            [
                "date",
                "product_id",
                "product_name",
                "category",
                "quantity",
                "unit_price",
                "revenue",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 9. Revenue consistency
    # --------------------------------------------------------

    print("\n[9] COHERENCE DU CHIFFRE D'AFFAIRES")
    print("-" * 70)

    revenue_ok = (
        df["revenue"] == df["quantity"] * df["unit_price"]
    ).all()

    print(f"Revenue = quantity × unit_price : {revenue_ok}")

    # --------------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIN DE L'ANALYSE J2.5")
    print("=" * 70)


if __name__ == "__main__":
    main()