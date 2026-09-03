"""
AI Sales Forecasting
J4.3.2 - Analyse temporelle par jour de la semaine

Objectif :
Analyser le comportement des ventes selon le jour
de la semaine et comparer semaine / week-end.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_clean.csv"


# Ordre professionnel des jours
WEEKDAY_ORDER = [
    "Lundi",
    "Mardi",
    "Mercredi",
    "Jeudi",
    "Vendredi",
    "Samedi",
    "Dimanche",
]


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
    """Ajoute les informations liées au jour de la semaine."""

    day_mapping = {
        0: "Lundi",
        1: "Mardi",
        2: "Mercredi",
        3: "Jeudi",
        4: "Vendredi",
        5: "Samedi",
        6: "Dimanche",
    }

    df = df.copy()

    df["weekday_number"] = df["date"].dt.dayofweek
    df["weekday"] = df["weekday_number"].map(day_mapping)

    df["period"] = df["weekday"].apply(
        lambda x: "Week-end"
        if x in ["Samedi", "Dimanche"]
        else "Semaine"
    )

    return df


# ============================================================
# ANALYSE PAR JOUR
# ============================================================

def weekday_analysis(df):
    """Agrège les ventes par jour de la semaine."""

    result = (
        df.groupby(
            ["weekday_number", "weekday"],
            as_index=False
        )
        .agg(
            total_quantity=("quantity", "sum"),
            average_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            average_revenue=("revenue", "mean"),
            observations=("quantity", "count"),
        )
        .sort_values("weekday_number")
    )

    return result


# ============================================================
# ANALYSE SEMAINE / WEEK-END
# ============================================================

def period_analysis(df):
    """Compare les ventes en semaine et le week-end."""

    result = (
        df.groupby("period", as_index=False)
        .agg(
            total_quantity=("quantity", "sum"),
            average_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            average_revenue=("revenue", "mean"),
            observations=("quantity", "count"),
        )
    )

    return result


# ============================================================
# AFFICHAGE JOUR DE LA SEMAINE
# ============================================================

def print_weekday_analysis(result):
    """Affiche les résultats par jour."""

    print("\n=== J4.3.2 — ANALYSE PAR JOUR DE LA SEMAINE ===")
    print("-" * 70)

    print(
        f"{'Jour':<12}"
        f"{'Qté totale':>12}"
        f"{'Qté moy.':>12}"
        f"{'CA total':>18}"
        f"{'CA moy.':>18}"
    )

    print("-" * 70)

    for _, row in result.iterrows():
        print(
            f"{row['weekday']:<12}"
            f"{row['total_quantity']:>12,.0f}"
            f"{row['average_quantity']:>12,.2f}"
            f"{row['total_revenue']:>18,.0f}"
            f"{row['average_revenue']:>18,.2f}"
        )


# ============================================================
# AFFICHAGE SEMAINE / WEEK-END
# ============================================================

def print_period_analysis(result):
    """Affiche la comparaison semaine / week-end."""

    print("\n=== SEMAINE vs WEEK-END ===")
    print("-" * 70)

    for _, row in result.iterrows():
        print(f"\n{row['period']} :")
        print(f"  Observations       : {row['observations']:,.0f}")
        print(f"  Quantity totale    : {row['total_quantity']:,.0f}")
        print(f"  Quantity moyenne   : {row['average_quantity']:,.2f}")
        print(f"  CA total           : {row['total_revenue']:,.0f} AR")
        print(f"  CA moyen           : {row['average_revenue']:,.2f} AR")


# ============================================================
# JOURS FORTS / FAIBLES
# ============================================================

def print_extreme_weekdays(result):
    """Identifie les jours les plus forts et les plus faibles."""

    strongest_quantity = result.loc[
        result["average_quantity"].idxmax()
    ]

    weakest_quantity = result.loc[
        result["average_quantity"].idxmin()
    ]

    strongest_revenue = result.loc[
        result["average_revenue"].idxmax()
    ]

    weakest_revenue = result.loc[
        result["average_revenue"].idxmin()
    ]

    print("\n=== JOURS FORTS / FAIBLES ===")
    print("-" * 70)

    print(
        f"Quantity moyenne maximale : "
        f"{strongest_quantity['weekday']} "
        f"({strongest_quantity['average_quantity']:.2f})"
    )

    print(
        f"Quantity moyenne minimale : "
        f"{weakest_quantity['weekday']} "
        f"({weakest_quantity['average_quantity']:.2f})"
    )

    print(
        f"CA moyen maximal          : "
        f"{strongest_revenue['weekday']} "
        f"({strongest_revenue['average_revenue']:,.2f} AR)"
    )

    print(
        f"CA moyen minimal          : "
        f"{weakest_revenue['weekday']} "
        f"({weakest_revenue['average_revenue']:,.2f} AR)"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_results(df, weekday_result, period_result):
    """Vérifie la cohérence des agrégations."""

    original_quantity = df["quantity"].sum()
    weekday_quantity = weekday_result["total_quantity"].sum()
    period_quantity = period_result["total_quantity"].sum()

    original_revenue = df["revenue"].sum()
    weekday_revenue = weekday_result["total_revenue"].sum()
    period_revenue = period_result["total_revenue"].sum()

    assert original_quantity == weekday_quantity
    assert original_quantity == period_quantity

    assert original_revenue == weekday_revenue
    assert original_revenue == period_revenue

    assert weekday_result["observations"].sum() == len(df)
    assert period_result["observations"].sum() == len(df)

    assert set(weekday_result["weekday"]) == set(WEEKDAY_ORDER)

    print("\n=== VALIDATION ===")
    print("[PASS] Agrégation par jour cohérente")
    print(f"[PASS] Quantity : {weekday_quantity:,.0f}")
    print(f"[PASS] Revenue  : {weekday_revenue:,.0f} AR")
    print("[PASS] 7 jours de la semaine présents")
    print("[PASS] Semaine + week-end = totalité des observations")


# ============================================================
# MAIN
# ============================================================

def main():
    """Point d'entrée principal."""

    print("AI Sales Forecasting")
    print("J4.3.2 - Analyse temporelle par jour de la semaine")
    print("-" * 70)

    df = load_data()

    df = prepare_data(df)

    weekday_result = weekday_analysis(df)

    period_result = period_analysis(df)

    print_weekday_analysis(weekday_result)

    print_period_analysis(period_result)

    print_extreme_weekdays(weekday_result)

    validate_results(
        df,
        weekday_result,
        period_result
    )

    print("\nJ4.3.2 — ANALYSE PAR JOUR DE LA SEMAINE : OK")


if __name__ == "__main__":
    main()