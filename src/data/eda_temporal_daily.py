"""
AI Sales Forecasting
J4.3.1 - Analyse temporelle quotidienne

Objectif :
Analyser l'évolution quotidienne des ventes
(quantity et revenue) sur l'ensemble de la période.
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
# ANALYSE QUOTIDIENNE
# ============================================================

def daily_analysis(df):
    """Agrège les ventes par jour."""

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            active_products=("quantity", lambda x: (x > 0).sum()),
        )
    )

    return daily


# ============================================================
# STATISTIQUES
# ============================================================

def print_statistics(daily):
    """Affiche les statistiques quotidiennes."""

    print("\n=== J4.3.1 — PROFIL QUOTIDIEN DES VENTES ===")
    print("-" * 55)

    print(f"Nombre de jours analysés : {len(daily)}")

    print("\n--- QUANTITY ---")

    print(f"Total quantité          : {daily['total_quantity'].sum():,.0f}")
    print(f"Moyenne quotidienne     : {daily['total_quantity'].mean():,.2f}")
    print(f"Médiane quotidienne     : {daily['total_quantity'].median():,.2f}")
    print(f"Minimum quotidien       : {daily['total_quantity'].min():,.0f}")
    print(f"Maximum quotidien       : {daily['total_quantity'].max():,.0f}")
    print(f"Écart-type              : {daily['total_quantity'].std():,.2f}")

    print("\n--- REVENUE ---")

    print(f"CA total                : {daily['total_revenue'].sum():,.0f} AR")
    print(f"CA moyen quotidien      : {daily['total_revenue'].mean():,.2f} AR")
    print(f"CA médian quotidien     : {daily['total_revenue'].median():,.2f} AR")
    print(f"CA minimum quotidien    : {daily['total_revenue'].min():,.0f} AR")
    print(f"CA maximum quotidien    : {daily['total_revenue'].max():,.0f} AR")
    print(f"Écart-type              : {daily['total_revenue'].std():,.2f} AR")

    print("\n--- PRODUITS ACTIFS ---")

    print(f"Moyenne produits actifs : "
          f"{daily['active_products'].mean():,.2f}")

    print(f"Minimum produits actifs : "
          f"{daily['active_products'].min():,.0f}")

    print(f"Maximum produits actifs : "
          f"{daily['active_products'].max():,.0f}")


# ============================================================
# JOURS EXTRÊMES
# ============================================================

def print_extreme_days(daily):
    """Affiche les jours avec les ventes les plus fortes/faibles."""

    max_quantity = daily.loc[
        daily["total_quantity"].idxmax()
    ]

    min_quantity = daily.loc[
        daily["total_quantity"].idxmin()
    ]

    max_revenue = daily.loc[
        daily["total_revenue"].idxmax()
    ]

    min_revenue = daily.loc[
        daily["total_revenue"].idxmin()
    ]

    print("\n=== JOURS EXTRÊMES ===")
    print("-" * 55)

    print("\nJour avec quantité maximale :")
    print(
        f"  Date     : {max_quantity['date'].date()}\n"
        f"  Quantity : {max_quantity['total_quantity']:,.0f}\n"
        f"  CA       : {max_quantity['total_revenue']:,.0f} AR"
    )

    print("\nJour avec quantité minimale :")
    print(
        f"  Date     : {min_quantity['date'].date()}\n"
        f"  Quantity : {min_quantity['total_quantity']:,.0f}\n"
        f"  CA       : {min_quantity['total_revenue']:,.0f} AR"
    )

    print("\nJour avec CA maximal :")
    print(
        f"  Date     : {max_revenue['date'].date()}\n"
        f"  Quantity : {max_revenue['total_quantity']:,.0f}\n"
        f"  CA       : {max_revenue['total_revenue']:,.0f} AR"
    )

    print("\nJour avec CA minimal :")
    print(
        f"  Date     : {min_revenue['date'].date()}\n"
        f"  Quantity : {min_revenue['total_quantity']:,.0f}\n"
        f"  CA       : {min_revenue['total_revenue']:,.0f} AR"
    )


# ============================================================
# TOP 10 JOURS
# ============================================================

def print_top_days(daily):
    """Affiche les 10 meilleurs jours."""

    top_days = (
        daily.sort_values(
            "total_quantity",
            ascending=False
        )
        .head(10)
    )

    print("\n=== TOP 10 JOURS PAR QUANTITÉ ===")
    print("-" * 55)

    for _, row in top_days.iterrows():
        print(
            f"{row['date'].date()} | "
            f"{row['total_quantity']:>3.0f} unités | "
            f"{row['total_revenue']:>10,.0f} AR"
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_daily_data(df, daily):
    """Vérifie la cohérence de l'agrégation quotidienne."""

    original_quantity = df["quantity"].sum()
    daily_quantity = daily["total_quantity"].sum()

    original_revenue = df["revenue"].sum()
    daily_revenue = daily["total_revenue"].sum()

    assert original_quantity == daily_quantity
    assert original_revenue == daily_revenue

    assert daily["date"].nunique() == len(daily)

    print("\n=== VALIDATION ===")
    print("[PASS] Agrégation quotidienne cohérente")
    print(f"[PASS] Quantity : {daily_quantity:,.0f}")
    print(f"[PASS] Revenue  : {daily_revenue:,.0f} AR")
    print(f"[PASS] Jours    : {len(daily)}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Point d'entrée principal."""

    print("AI Sales Forecasting")
    print("J4.3.1 - Analyse temporelle quotidienne")
    print("-" * 55)

    df = load_data()

    daily = daily_analysis(df)

    print_statistics(daily)

    print_extreme_days(daily)

    print_top_days(daily)

    validate_daily_data(df, daily)

    print("\nJ4.3.1 — ANALYSE QUOTIDIENNE : OK")


if __name__ == "__main__":
    main()