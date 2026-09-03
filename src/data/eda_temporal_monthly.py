"""
AI Sales Forecasting
J4.3.3 - Analyse temporelle mensuelle

Objectif :
Analyser l'évolution des ventes par mois afin
d'identifier les tendances et éventuels effets saisonniers.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_clean.csv"


# ============================================================
# CHARGEMENT
# ============================================================

def load_data():
    """Charge le dataset nettoyé."""

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    return df


# ============================================================
# PRÉPARATION
# ============================================================

def prepare_data(df):
    """Ajoute les informations mensuelles."""

    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month_number"] = df["date"].dt.month

    # Période mensuelle YYYY-MM
    df["month"] = df["date"].dt.to_period("M")

    return df


# ============================================================
# ANALYSE MENSUELLE
# ============================================================

def monthly_analysis(df):
    """Agrège les ventes par mois."""

    monthly = (
        df.groupby(
            ["year", "month_number", "month"],
            as_index=False
        )
        .agg(
            total_quantity=("quantity", "sum"),
            average_daily_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            average_daily_revenue=("revenue", "mean"),
            active_products=("quantity", lambda x: (x > 0).sum()),
            observations=("quantity", "count"),
        )
    )

    return monthly


# ============================================================
# CALCUL DES VARIATIONS
# ============================================================

def calculate_variations(monthly):
    """Calcule les variations mensuelles."""

    monthly = monthly.copy()

    monthly["quantity_change_pct"] = (
        monthly["total_quantity"]
        .pct_change()
        * 100
    )

    monthly["revenue_change_pct"] = (
        monthly["total_revenue"]
        .pct_change()
        * 100
    )

    return monthly


# ============================================================
# AFFICHAGE
# ============================================================

def print_monthly_analysis(monthly):
    """Affiche l'analyse mensuelle."""

    print("\n=== J4.3.3 — ANALYSE MENSUELLE ===")
    print("-" * 100)

    print(
        f"{'Mois':<10}"
        f"{'Qté totale':>14}"
        f"{'Qté moy./jour':>16}"
        f"{'CA total':>20}"
        f"{'CA moy./jour':>20}"
        f"{'Produits actifs':>16}"
    )

    print("-" * 100)

    for _, row in monthly.iterrows():

        print(
            f"{str(row['month']):<10}"
            f"{row['total_quantity']:>14,.0f}"
            f"{row['average_daily_quantity']:>16,.2f}"
            f"{row['total_revenue']:>20,.0f}"
            f"{row['average_daily_revenue']:>20,.2f}"
            f"{row['active_products']:>16,.0f}"
        )


# ============================================================
# VARIATIONS MENSUELLES
# ============================================================

def print_variations(monthly):
    """Affiche les variations mensuelles."""

    print("\n=== VARIATIONS MENSUELLES ===")
    print("-" * 75)

    for _, row in monthly.iloc[1:].iterrows():

        quantity_change = row["quantity_change_pct"]
        revenue_change = row["revenue_change_pct"]

        print(
            f"{row['month']} | "
            f"Quantity : {quantity_change:+.2f}% | "
            f"CA : {revenue_change:+.2f}%"
        )


# ============================================================
# MOIS FORTS / FAIBLES
# ============================================================

def print_extreme_months(monthly):
    """Identifie les mois les plus forts et les plus faibles."""

    max_quantity = monthly.loc[
        monthly["total_quantity"].idxmax()
    ]

    min_quantity = monthly.loc[
        monthly["total_quantity"].idxmin()
    ]

    max_revenue = monthly.loc[
        monthly["total_revenue"].idxmax()
    ]

    min_revenue = monthly.loc[
        monthly["total_revenue"].idxmin()
    ]

    max_average_quantity = monthly.loc[
        monthly["average_daily_quantity"].idxmax()
    ]

    min_average_quantity = monthly.loc[
        monthly["average_daily_quantity"].idxmin()
    ]

    print("\n=== MOIS FORTS / FAIBLES ===")
    print("-" * 75)

    print(
        f"Quantity totale maximale : "
        f"{max_quantity['month']} "
        f"({max_quantity['total_quantity']:,.0f})"
    )

    print(
        f"Quantity totale minimale : "
        f"{min_quantity['month']} "
        f"({min_quantity['total_quantity']:,.0f})"
    )

    print(
        f"CA total maximal         : "
        f"{max_revenue['month']} "
        f"({max_revenue['total_revenue']:,.0f} AR)"
    )

    print(
        f"CA total minimal         : "
        f"{min_revenue['month']} "
        f"({min_revenue['total_revenue']:,.0f} AR)"
    )

    print(
        f"Meilleure moyenne/jour   : "
        f"{max_average_quantity['month']} "
        f"({max_average_quantity['average_daily_quantity']:.2f})"
    )

    print(
        f"Plus faible moyenne/jour : "
        f"{min_average_quantity['month']} "
        f"({min_average_quantity['average_daily_quantity']:.2f})"
    )


# ============================================================
# MOYENNES PAR MOIS CALENDAIRE
# ============================================================

def print_calendar_month_analysis(df):
    """
    Compare les mois calendaires indépendamment de l'année.

    Exemple :
    toutes les observations de septembre 2025 et septembre 2026.
    """

    calendar_month = (
        df.groupby("month_number", as_index=False)
        .agg(
            total_quantity=("quantity", "sum"),
            average_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            average_revenue=("revenue", "mean"),
        )
    )

    print("\n=== ANALYSE PAR MOIS CALENDAIRE ===")
    print("-" * 75)

    for _, row in calendar_month.iterrows():

        print(
            f"Mois {int(row['month_number']):02d} | "
            f"Qté moyenne : {row['average_quantity']:.2f} | "
            f"CA moyen : {row['average_revenue']:,.2f} AR"
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_results(df, monthly):
    """Vérifie la cohérence de l'agrégation mensuelle."""

    original_quantity = df["quantity"].sum()
    monthly_quantity = monthly["total_quantity"].sum()

    original_revenue = df["revenue"].sum()
    monthly_revenue = monthly["total_revenue"].sum()

    assert original_quantity == monthly_quantity
    assert original_revenue == monthly_revenue

    assert monthly["observations"].sum() == len(df)

    assert monthly["month"].nunique() == len(monthly)

    print("\n=== VALIDATION ===")
    print("[PASS] Agrégation mensuelle cohérente")
    print(f"[PASS] Quantity : {monthly_quantity:,.0f}")
    print(f"[PASS] Revenue  : {monthly_revenue:,.0f} AR")
    print(f"[PASS] Mois     : {len(monthly)}")
    print("[PASS] Nombre d'observations conservé")


# ============================================================
# MAIN
# ============================================================

def main():
    """Point d'entrée principal."""

    print("AI Sales Forecasting")
    print("J4.3.3 - Analyse temporelle mensuelle")
    print("-" * 100)

    df = load_data()

    df = prepare_data(df)

    monthly = monthly_analysis(df)

    monthly = calculate_variations(monthly)

    print_monthly_analysis(monthly)

    print_variations(monthly)

    print_extreme_months(monthly)

    print_calendar_month_analysis(df)

    validate_results(df, monthly)

    print("\nJ4.3.3 — ANALYSE MENSUELLE : OK")


if __name__ == "__main__":
    main()