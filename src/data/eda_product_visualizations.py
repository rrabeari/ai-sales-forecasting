"""
AI Sales Forecasting
J4.4.4 - Visualisation de l'analyse produits
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sales_clean.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "eda"
)


def load_data():
    """Charger les données nettoyées."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    return pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )


def prepare_product_data(df):
    """Préparer les données au niveau produit."""

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
            average_quantity=("quantity", "mean"),
            std_quantity=("quantity", "std"),
        )
    )

    product["coefficient_variation"] = (
        product["std_quantity"]
        / product["average_quantity"]
    )

    return product


def prepare_category_product_data(product):
    """Calculer la contribution de chaque produit à sa catégorie."""

    category_totals = (
        product.groupby(
            "category",
            as_index=False,
        )
        .agg(
            category_revenue=("total_revenue", "sum")
        )
    )

    product_category = product.merge(
        category_totals,
        on="category",
        how="left",
    )

    product_category["category_revenue_share"] = (
        product_category["total_revenue"]
        / product_category["category_revenue"]
        * 100
    )

    return product_category


def create_output_directory():
    """Créer le dossier de sortie."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def plot_top_quantity(product):
    """Graphique Top 10 produits par quantité."""

    data = product.nlargest(
        10,
        "total_quantity",
    ).sort_values(
        "total_quantity"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        data["product_name"],
        data["total_quantity"],
    )

    plt.title(
        "Top 10 produits par quantité vendue"
    )

    plt.xlabel(
        "Quantité vendue"
    )

    plt.ylabel(
        "Produit"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "top_products_quantity.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[OK] {output.name}"
    )


def plot_top_revenue(product):
    """Graphique Top 10 produits par CA."""

    data = product.nlargest(
        10,
        "total_revenue",
    ).sort_values(
        "total_revenue"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        data["product_name"],
        data["total_revenue"],
    )

    plt.title(
        "Top 10 produits par chiffre d'affaires"
    )

    plt.xlabel(
        "Chiffre d'affaires (AR)"
    )

    plt.ylabel(
        "Produit"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "top_products_revenue.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[OK] {output.name}"
    )


def plot_volume_vs_revenue(product):
    """Nuage de points quantité vs CA."""

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        product["total_quantity"],
        product["total_revenue"],
        s=80,
    )

    for _, row in product.iterrows():
        plt.annotate(
            row["product_name"],
            (
                row["total_quantity"],
                row["total_revenue"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.title(
        "Relation entre volume vendu et chiffre d'affaires"
    )

    plt.xlabel(
        "Quantité totale vendue"
    )

    plt.ylabel(
        "Chiffre d'affaires (AR)"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "product_volume_vs_revenue.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[OK] {output.name}"
    )


def plot_category_contribution(product_category):
    """Contribution des produits au CA de leur catégorie."""

    data = product_category.sort_values(
        [
            "category",
            "category_revenue_share",
        ],
        ascending=[
            True,
            False,
        ],
    )

    plt.figure(
        figsize=(12, 7)
    )

    labels = (
        data["product_name"]
        + " (" 
        + data["category"]
        + ")"
    )

    plt.barh(
        labels,
        data["category_revenue_share"],
    )

    plt.title(
        "Contribution des produits au CA de leur catégorie"
    )

    plt.xlabel(
        "Part du CA de la catégorie (%)"
    )

    plt.ylabel(
        "Produit"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "category_product_contribution.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[OK] {output.name}"
    )


def plot_demand_variability(product):
    """Visualiser la variabilité de la demande."""

    data = product.sort_values(
        "coefficient_variation"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        data["product_name"],
        data["coefficient_variation"],
    )

    plt.axvline(
        0.50,
        linestyle="--",
        label="Seuil CV = 0.50",
    )

    plt.axvline(
        0.75,
        linestyle="--",
        label="Seuil CV = 0.75",
    )

    plt.title(
        "Variabilité de la demande par produit"
    )

    plt.xlabel(
        "Coefficient de variation"
    )

    plt.ylabel(
        "Produit"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "product_demand_variability.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[OK] {output.name}"
    )


def validate_outputs():
    """Vérifier que tous les graphiques ont été créés."""

    expected_files = [
        "top_products_quantity.png",
        "top_products_revenue.png",
        "product_volume_vs_revenue.png",
        "category_product_contribution.png",
        "product_demand_variability.png",
    ]

    for filename in expected_files:
        filepath = OUTPUT_DIR / filename

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    print(
        "\n[PASS] Les 5 visualisations ont été générées"
    )


def main():
    print(
        "AI Sales Forecasting"
    )

    print(
        "J4.4.4 - Visualisation de l'analyse produits"
    )

    print(
        "-" * 60
    )

    df = load_data()

    print(
        f"\nLignes analysées : {len(df):,}"
    )

    print(
        f"Produits analysés : "
        f"{df['product_id'].nunique()}"
    )

    create_output_directory()

    product = prepare_product_data(
        df
    )

    product_category = (
        prepare_category_product_data(
            product
        )
    )

    print(
        "\n=== GÉNÉRATION DES GRAPHIQUES ==="
    )

    plot_top_quantity(
        product
    )

    plot_top_revenue(
        product
    )

    plot_volume_vs_revenue(
        product
    )

    plot_category_contribution(
        product_category
    )

    plot_demand_variability(
        product
    )

    validate_outputs()

    print(
        "\n=== SORTIE ==="
    )

    print(
        f"Dossier : {OUTPUT_DIR}"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "J4.4.4 — VISUALISATIONS PRODUITS : OK"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()