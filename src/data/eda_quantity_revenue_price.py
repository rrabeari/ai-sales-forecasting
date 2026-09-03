"""
AI Sales Forecasting
J4.5.3 - Analyse des relations Quantité / CA / Prix

Objectif :
    Étudier les relations entre la quantité vendue,
    le chiffre d'affaires et le prix unitaire.

Aucune donnée n'est supprimée ou modifiée.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "eda"

PRODUCT_OUTPUT = OUTPUT_DIR / "quantity_revenue_price_products.csv"


# ============================================================
# CHARGEMENT
# ============================================================

def load_data() -> pd.DataFrame:
    """Charge le dataset nettoyé."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    return df


# ============================================================
# VALIDATION MATHÉMATIQUE
# ============================================================

def validate_revenue_formula(df: pd.DataFrame):
    """Vérifie que CA = quantité × prix unitaire."""

    print("\n=== VALIDATION CA = QUANTITÉ × PRIX ===")

    expected_revenue = (
        df["quantity"] * df["unit_price"]
    )

    differences = (
        df["revenue"] - expected_revenue
    ).abs()

    max_difference = differences.max()

    valid = (differences < 0.01).all()

    print(
        f"Écart maximum : {max_difference:,.2f} AR"
    )

    if valid:
        print(
            "[PASS] Toutes les lignes respectent "
            "CA = quantité × prix"
        )
    else:
        raise ValueError(
            "Erreur dans la relation CA = quantité × prix."
        )


# ============================================================
# CORRÉLATIONS
# ============================================================

def analyze_correlations(df: pd.DataFrame):
    """Calcule les corrélations entre quantité, prix et CA."""

    print("\n=== CORRÉLATIONS ===")

    correlation = df[
        ["quantity", "unit_price", "revenue"]
    ].corr()

    print(
        correlation.round(3).to_string()
    )

    print("\nInterprétation des corrélations :")

    quantity_revenue = correlation.loc[
        "quantity", "revenue"
    ]

    quantity_price = correlation.loc[
        "quantity", "unit_price"
    ]

    price_revenue = correlation.loc[
        "unit_price", "revenue"
    ]

    print(
        f"Quantité ↔ CA       : "
        f"{quantity_revenue:.3f}"
    )

    print(
        f"Quantité ↔ Prix     : "
        f"{quantity_price:.3f}"
    )

    print(
        f"Prix ↔ CA           : "
        f"{price_revenue:.3f}"
    )

    return correlation


# ============================================================
# ANALYSE PAR PRODUIT
# ============================================================

def analyze_products(df: pd.DataFrame):
    """Analyse quantité, prix et CA par produit."""

    print("\n=== ANALYSE PAR PRODUIT ===")

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
            avg_daily_quantity=("quantity", "mean"),
            avg_daily_revenue=("revenue", "mean"),
            active_days=("quantity", lambda x: (x > 0).sum()),
            zero_days=("quantity", lambda x: (x == 0).sum()),
        )
    )

    product["revenue_per_unit"] = (
        product["total_revenue"]
        / product["total_quantity"]
    )

    product["quantity_share_pct"] = (
        product["total_quantity"]
        / product["total_quantity"].sum()
        * 100
    )

    product["revenue_share_pct"] = (
        product["total_revenue"]
        / product["total_revenue"].sum()
        * 100
    )

    product["quantity_rank"] = (
        product["total_quantity"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    product["revenue_rank"] = (
        product["total_revenue"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    product["rank_difference"] = (
        product["quantity_rank"]
        - product["revenue_rank"]
    )

    # Médianes utilisées uniquement pour classer les profils.
    median_quantity = product["total_quantity"].median()
    median_revenue = product["total_revenue"].median()

    def classify(row):

        high_quantity = (
            row["total_quantity"] >= median_quantity
        )

        high_revenue = (
            row["total_revenue"] >= median_revenue
        )

        if high_quantity and high_revenue:
            return "Volume élevé + CA élevé"

        if high_quantity and not high_revenue:
            return "Volume élevé + CA faible"

        if not high_quantity and high_revenue:
            return "Volume faible + CA élevé"

        return "Volume faible + CA faible"

    product["performance_profile"] = product.apply(
        classify,
        axis=1,
    )

    return product


# ============================================================
# AFFICHAGE TOP PRODUITS
# ============================================================

def display_products(product: pd.DataFrame):
    """Affiche les principaux produits."""

    print("\n=== PRODUITS PAR VOLUME ===")

    top_quantity = product.sort_values(
        "total_quantity",
        ascending=False,
    ).head(10)

    print(
        top_quantity[
            [
                "product_name",
                "unit_price",
                "total_quantity",
                "total_revenue",
                "revenue_per_unit",
                "quantity_share_pct",
            ]
        ].to_string(
            index=False,
            formatters={
                "unit_price": "{:,.0f}".format,
                "total_quantity": "{:,.0f}".format,
                "total_revenue": "{:,.0f}".format,
                "revenue_per_unit": "{:,.0f}".format,
                "quantity_share_pct": "{:.2f}%".format,
            },
        )
    )

    print("\n=== PRODUITS PAR CA ===")

    top_revenue = product.sort_values(
        "total_revenue",
        ascending=False,
    ).head(10)

    print(
        top_revenue[
            [
                "product_name",
                "unit_price",
                "total_quantity",
                "total_revenue",
                "revenue_per_unit",
                "revenue_share_pct",
            ]
        ].to_string(
            index=False,
            formatters={
                "unit_price": "{:,.0f}".format,
                "total_quantity": "{:,.0f}".format,
                "total_revenue": "{:,.0f}".format,
                "revenue_per_unit": "{:,.0f}".format,
                "revenue_share_pct": "{:.2f}%".format,
            },
        )
    )


# ============================================================
# PROFILS VOLUME / CA
# ============================================================

def analyze_profiles(product: pd.DataFrame):
    """Analyse les profils volume / CA."""

    print("\n=== PROFILS VOLUME / CA ===")

    profile_counts = (
        product["performance_profile"]
        .value_counts()
    )

    print(
        profile_counts.to_string()
    )

    print("\nDétail des produits :")

    details = product.sort_values(
        [
            "performance_profile",
            "total_revenue",
        ],
        ascending=[True, False],
    )

    print(
        details[
            [
                "product_name",
                "total_quantity",
                "total_revenue",
                "performance_profile",
            ]
        ].to_string(
            index=False,
            formatters={
                "total_quantity": "{:,.0f}".format,
                "total_revenue": "{:,.0f}".format,
            },
        )
    )


# ============================================================
# ANALYSE DU PRIX
# ============================================================

def analyze_price_effect(product: pd.DataFrame):
    """Analyse le rapport entre prix et contribution au CA."""

    print("\n=== ANALYSE DU PRIX UNITAIRE ===")

    price_analysis = product[
        [
            "product_name",
            "unit_price",
            "total_quantity",
            "total_revenue",
            "revenue_per_unit",
        ]
    ].sort_values(
        "unit_price",
        ascending=False,
    )

    print(
        price_analysis.to_string(
            index=False,
            formatters={
                "unit_price": "{:,.0f}".format,
                "total_quantity": "{:,.0f}".format,
                "total_revenue": "{:,.0f}".format,
                "revenue_per_unit": "{:,.0f}".format,
            },
        )
    )

    price_correlation = product[
        [
            "unit_price",
            "total_quantity",
            "total_revenue",
        ]
    ].corr()

    print("\nCorrélations au niveau produit :")

    print(
        price_correlation.round(3).to_string()
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(df: pd.DataFrame, product: pd.DataFrame):
    """Effectue les contrôles finaux."""

    print("\n=== VALIDATION ===")

    checks = {
        "Dataset non vide": len(df) > 0,
        "14 produits analysés": (
            product["product_id"].nunique() == 14
        ),
        "Prix positifs": (
            product["unit_price"] > 0
        ).all(),
        "Quantités non négatives": (
            product["total_quantity"] >= 0
        ).all(),
        "CA non négatif": (
            product["total_revenue"] >= 0
        ).all(),
        "CA produit cohérent": (
            (
                product["total_revenue"]
                - (
                    product["total_quantity"]
                    * product["revenue_per_unit"]
                )
            ).abs()
            < 0.01
        ).all(),
    }

    all_pass = True

    for name, status in checks.items():

        if status:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")
            all_pass = False

    if not all_pass:
        raise ValueError(
            "La validation J4.5.3 a échoué."
        )

    print(
        "\n[PASS] Analyse quantité / CA / prix validée"
    )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_results(product: pd.DataFrame):
    """Sauvegarde les résultats produits."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    product.to_csv(
        PRODUCT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== SORTIE ===")
    print(
        f"Fichier : {PRODUCT_OUTPUT}"
    )
    print(
        f"Lignes : {len(product):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("AI Sales Forecasting")
    print("J4.5.3 - Analyse Quantité / CA / Prix")
    print("-" * 60)

    df = load_data()

    print(
        f"\nLignes analysées : {len(df):,}"
    )

    validate_revenue_formula(df)

    correlation = analyze_correlations(df)

    product = analyze_products(df)

    display_products(product)

    analyze_profiles(product)

    analyze_price_effect(product)

    validate(
        df,
        product,
    )

    save_results(product)

    print("\n" + "=" * 60)
    print(
        "J4.5.3 — ANALYSE QUANTITÉ / CA / PRIX : OK"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()