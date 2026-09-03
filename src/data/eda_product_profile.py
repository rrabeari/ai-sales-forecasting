"""
AI Sales Forecasting
J4.4.1 - Analyse approfondie des produits
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_clean.csv"


def load_data():
    """Load the cleaned sales dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    return df


def analyze_products(df):
    """Build a detailed demand profile for each product."""

    profile = (
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
            average_quantity=("quantity", "mean"),
            median_quantity=("quantity", "median"),
            std_quantity=("quantity", "std"),
            min_quantity=("quantity", "min"),
            max_quantity=("quantity", "max"),
            selling_days=("quantity", lambda x: (x > 0).sum()),
            zero_days=("quantity", lambda x: (x == 0).sum()),
            total_revenue=("revenue", "sum"),
            average_revenue=("revenue", "mean"),
        )
    )

    total_days = df["date"].nunique()

    profile["active_rate"] = (
        profile["selling_days"] / total_days * 100
    )

    profile["zero_rate"] = (
        profile["zero_days"] / total_days * 100
    )

    # Coefficient de variation :
    # mesure la variabilité de la demande relativement
    # à sa moyenne.
    profile["coefficient_variation"] = np.where(
        profile["average_quantity"] > 0,
        profile["std_quantity"] / profile["average_quantity"],
        np.nan,
    )

    # Classification de régularité
    def classify_regularity(cv):
        if pd.isna(cv):
            return "Non déterminé"
        if cv < 0.50:
            return "Très stable"
        if cv < 0.75:
            return "Stable"
        if cv < 1.00:
            return "Variable"
        return "Très variable"

    profile["demand_regularity"] = profile[
        "coefficient_variation"
    ].apply(classify_regularity)

    # Classification du volume
    quantity_median = profile["total_quantity"].median()

    profile["volume_profile"] = np.where(
        profile["total_quantity"] >= quantity_median,
        "Volume élevé",
        "Volume faible",
    )

    # Classification du CA
    revenue_median = profile["total_revenue"].median()

    profile["revenue_profile"] = np.where(
        profile["total_revenue"] >= revenue_median,
        "CA élevé",
        "CA faible",
    )

    return profile.sort_values(
        "total_quantity",
        ascending=False,
    )


def validate_profile(profile, df):
    """Validate the product analysis."""

    expected_products = df["product_id"].nunique()

    assert len(profile) == expected_products
    assert profile["total_quantity"].sum() == df["quantity"].sum()
    assert profile["total_revenue"].sum() == df["revenue"].sum()

    assert (profile["selling_days"] >= 0).all()
    assert (profile["zero_days"] >= 0).all()

    assert (
        profile["selling_days"] + profile["zero_days"]
        == df["date"].nunique()
    ).all()

    assert profile["active_rate"].between(0, 100).all()
    assert profile["zero_rate"].between(0, 100).all()

    print("[PASS] Nombre de produits cohérent")
    print("[PASS] Quantité totale cohérente")
    print("[PASS] Chiffre d'affaires total cohérent")
    print("[PASS] Jours actifs / jours zéro cohérents")
    print("[PASS] Taux d'activité valides")


def print_report(profile):
    """Display the product profile report."""

    print("\n=== PROFIL DES PRODUITS ===")

    columns = [
        "product_name",
        "category",
        "unit_price",
        "total_quantity",
        "average_quantity",
        "std_quantity",
        "selling_days",
        "zero_days",
        "active_rate",
        "total_revenue",
        "coefficient_variation",
        "demand_regularity",
        "volume_profile",
        "revenue_profile",
    ]

    display_df = profile[columns].copy()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.2f}".format)

    print(display_df.to_string(index=False))

    print("\n=== TOP 5 VOLUME ===")

    top_volume = profile.nlargest(
        5,
        "total_quantity",
    )

    for _, row in top_volume.iterrows():
        print(
            f"- {row['product_name']}: "
            f"{int(row['total_quantity'])} unités"
        )

    print("\n=== TOP 5 CA ===")

    top_revenue = profile.nlargest(
        5,
        "total_revenue",
    )

    for _, row in top_revenue.iterrows():
        print(
            f"- {row['product_name']}: "
            f"{row['total_revenue']:,.0f} AR"
        )

    print("\n=== PRODUITS LES PLUS STABLES ===")

    stable = profile.nsmallest(
        5,
        "coefficient_variation",
    )

    for _, row in stable.iterrows():
        print(
            f"- {row['product_name']}: "
            f"CV={row['coefficient_variation']:.2f} "
            f"({row['demand_regularity']})"
        )

    print("\n=== PRODUITS LES PLUS VARIABLES ===")

    variable = profile.nlargest(
        5,
        "coefficient_variation",
    )

    for _, row in variable.iterrows():
        print(
            f"- {row['product_name']}: "
            f"CV={row['coefficient_variation']:.2f} "
            f"({row['demand_regularity']})"
        )


def main():
    print("AI Sales Forecasting")
    print("J4.4.1 - Analyse approfondie des produits")
    print("-" * 60)

    df = load_data()

    print(f"\nLignes analysées : {len(df):,}")
    print(f"Produits analysés : {df['product_id'].nunique()}")

    profile = analyze_products(df)

    validate_profile(profile, df)
    print_report(profile)

    print("\n" + "=" * 60)
    print("J4.4.1 — ANALYSE PRODUITS : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()