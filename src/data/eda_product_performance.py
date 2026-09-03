"""
AI Sales Forecasting
J4.4.2 - Comparaison de performance produits
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


def analyze_performance(df):
    """Analyser la performance volume vs chiffre d'affaires."""

    total_quantity = df["quantity"].sum()
    total_revenue = df["revenue"].sum()

    performance = (
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

    # Parts dans le volume et le CA global.
    performance["quantity_share"] = (
        performance["total_quantity"]
        / total_quantity
        * 100
    )

    performance["revenue_share"] = (
        performance["total_revenue"]
        / total_revenue
        * 100
    )

    # CA moyen généré par unité vendue.
    performance["revenue_per_unit"] = (
        performance["total_revenue"]
        / performance["total_quantity"]
    )

    # Classements.
    performance["volume_rank"] = (
        performance["total_quantity"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    performance["revenue_rank"] = (
        performance["total_revenue"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # Différence entre le classement CA et volume.
    #
    # Valeur positive :
    # le produit est mieux classé en CA qu'en volume.
    #
    # Valeur négative :
    # le produit est mieux classé en volume qu'en CA.
    performance["rank_difference"] = (
        performance["volume_rank"]
        - performance["revenue_rank"]
    )

    def classify_performance(row):
        high_volume = row["quantity_share"] >= (
            100 / len(performance)
        )
        high_revenue = row["revenue_share"] >= (
            100 / len(performance)
        )

        if high_volume and high_revenue:
            return "Volume élevé + CA élevé"

        if high_volume and not high_revenue:
            return "Volume élevé + CA faible"

        if not high_volume and high_revenue:
            return "Volume faible + CA élevé"

        return "Volume faible + CA faible"

    performance["performance_profile"] = performance.apply(
        classify_performance,
        axis=1,
    )

    return performance.sort_values(
        "total_revenue",
        ascending=False,
    )


def validate_performance(performance, df):
    """Valider les résultats."""

    assert len(performance) == df["product_id"].nunique()

    assert (
        performance["total_quantity"].sum()
        == df["quantity"].sum()
    )

    assert (
        performance["total_revenue"].sum()
        == df["revenue"].sum()
    )

    assert abs(
        performance["quantity_share"].sum() - 100
    ) < 0.01

    assert abs(
        performance["revenue_share"].sum() - 100
    ) < 0.01

    assert (
        performance["revenue_per_unit"] > 0
    ).all()

    assert (
        performance["volume_rank"].nunique()
        == len(performance)
    )

    assert (
        performance["revenue_rank"].nunique()
        == len(performance)
    )

    print("[PASS] Nombre de produits cohérent")
    print("[PASS] Quantité totale cohérente")
    print("[PASS] CA total cohérent")
    print("[PASS] Parts de volume = 100 %")
    print("[PASS] Parts de CA = 100 %")
    print("[PASS] CA par unité valide")
    print("[PASS] Classements cohérents")


def print_report(performance):
    """Afficher le rapport d'analyse."""

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option(
        "display.float_format",
        "{:.2f}".format,
    )

    print("\n=== PERFORMANCE PRODUITS ===")

    columns = [
        "product_name",
        "category",
        "unit_price",
        "total_quantity",
        "quantity_share",
        "total_revenue",
        "revenue_share",
        "revenue_per_unit",
        "volume_rank",
        "revenue_rank",
        "rank_difference",
        "performance_profile",
    ]

    print(
        performance[columns].to_string(
            index=False
        )
    )

    print("\n=== PRODUITS LES PLUS PERFORMANTS EN VOLUME ===")

    for _, row in performance.nlargest(
        5,
        "total_quantity",
    ).iterrows():
        print(
            f"- {row['product_name']}: "
            f"{int(row['total_quantity'])} unités "
            f"({row['quantity_share']:.2f} % du volume)"
        )

    print("\n=== PRODUITS LES PLUS PERFORMANTS EN CA ===")

    for _, row in performance.nlargest(
        5,
        "total_revenue",
    ).iterrows():
        print(
            f"- {row['product_name']}: "
            f"{row['total_revenue']:,.0f} AR "
            f"({row['revenue_share']:.2f} % du CA)"
        )

    print("\n=== PRODUITS AVEC LE PLUS GRAND ÉCART DE CLASSEMENT ===")

    ranking_gap = performance.reindex(
        performance["rank_difference"]
        .abs()
        .sort_values(ascending=False)
        .index
    )

    for _, row in ranking_gap.head(5).iterrows():
        print(
            f"- {row['product_name']}: "
            f"rang volume={row['volume_rank']}, "
            f"rang CA={row['revenue_rank']}, "
            f"écart={row['rank_difference']:+d}"
        )

    print("\n=== PROFILS DE PERFORMANCE ===")

    profile_counts = (
        performance["performance_profile"]
        .value_counts()
    )

    for profile, count in profile_counts.items():
        print(
            f"- {profile}: {count} produit(s)"
        )


def main():
    print("AI Sales Forecasting")
    print("J4.4.2 - Comparaison de performance produits")
    print("-" * 60)

    df = load_data()

    print(f"\nLignes analysées : {len(df):,}")
    print(
        f"Produits analysés : "
        f"{df['product_id'].nunique()}"
    )

    performance = analyze_performance(df)

    validate_performance(
        performance,
        df,
    )

    print_report(performance)

    print("\n" + "=" * 60)
    print("J4.4.2 — PERFORMANCE PRODUITS : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()