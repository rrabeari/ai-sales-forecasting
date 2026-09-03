"""
AI Sales Forecasting
J4.3.4 - Analyse de la tendance et de la saisonnalité

Objectif :
Identifier les tendances générales et les effets saisonniers
dans les ventes quotidiennes.

Important :
Les moyennes mobiles sont utilisées uniquement pour l'EDA.
Les features ML seront créées en J5.
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
# AGRÉGATION QUOTIDIENNE
# ============================================================

def build_daily_data(df):
    """Construit les ventes totales par jour."""

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily


# ============================================================
# MOYENNES MOBILES
# ============================================================

def calculate_moving_averages(daily):
    """
    Calcule des moyennes mobiles descriptives.

    Elles servent uniquement à visualiser la tendance
    et ne constituent pas encore des features ML.
    """

    daily = daily.copy()

    daily["ma_7"] = (
        daily["total_quantity"]
        .rolling(window=7)
        .mean()
    )

    daily["ma_30"] = (
        daily["total_quantity"]
        .rolling(window=30)
        .mean()
    )

    daily["ma_90"] = (
        daily["total_quantity"]
        .rolling(window=90)
        .mean()
    )

    return daily


# ============================================================
# TENDANCE DÉBUT / FIN DE PÉRIODE
# ============================================================

def analyze_period_trend(daily):
    """
    Compare plusieurs périodes afin d'identifier
    une tendance générale.
    """

    first_30 = daily.head(30)
    last_30 = daily.tail(30)

    first_90 = daily.head(90)
    last_90 = daily.tail(90)

    results = {
        "first_30_quantity": first_30["total_quantity"].mean(),
        "last_30_quantity": last_30["total_quantity"].mean(),
        "first_90_quantity": first_90["total_quantity"].mean(),
        "last_90_quantity": last_90["total_quantity"].mean(),
        "first_30_revenue": first_30["total_revenue"].mean(),
        "last_30_revenue": last_30["total_revenue"].mean(),
        "first_90_revenue": first_90["total_revenue"].mean(),
        "last_90_revenue": last_90["total_revenue"].mean(),
    }

    return results


# ============================================================
# SAISONNALITÉ MENSUELLE
# ============================================================

def analyze_month_seasonality(df):
    """
    Calcule la demande moyenne par mois calendaire.

    Le but est d'identifier les mois structurellement
    plus forts ou plus faibles.
    """

    data = df.copy()

    data["month_number"] = data["date"].dt.month

    monthly = (
        data.groupby("month_number", as_index=False)
        .agg(
            average_quantity=("quantity", "mean"),
            average_revenue=("revenue", "mean"),
        )
        .sort_values("month_number")
    )

    return monthly


# ============================================================
# SAISONNALITÉ HEBDOMADAIRE
# ============================================================

def analyze_weekday_seasonality(df):
    """Analyse la demande moyenne par jour de la semaine."""

    data = df.copy()

    data["weekday_number"] = data["date"].dt.dayofweek

    weekday = (
        data.groupby("weekday_number", as_index=False)
        .agg(
            average_quantity=("quantity", "mean"),
            average_revenue=("revenue", "mean"),
        )
        .sort_values("weekday_number")
    )

    return weekday


# ============================================================
# AFFICHAGE TENDANCE
# ============================================================

def print_trend_analysis(daily, trend):
    """Affiche les résultats de tendance."""

    print("\n=== J4.3.4 — ANALYSE DE LA TENDANCE ===")
    print("-" * 75)

    print("\n--- MOYENNES MOBILES DES VENTES QUOTIDIENNES ---")

    print(
        f"MA 7 jours  - dernière valeur : "
        f"{daily['ma_7'].iloc[-1]:.2f}"
    )

    print(
        f"MA 30 jours - dernière valeur : "
        f"{daily['ma_30'].iloc[-1]:.2f}"
    )

    print(
        f"MA 90 jours - dernière valeur : "
        f"{daily['ma_90'].iloc[-1]:.2f}"
    )

    print("\n--- COMPARAISON DÉBUT / FIN DE PÉRIODE ---")

    print(
        f"Quantité moyenne 30 premiers jours : "
        f"{trend['first_30_quantity']:.2f}"
    )

    print(
        f"Quantité moyenne 30 derniers jours : "
        f"{trend['last_30_quantity']:.2f}"
    )

    print(
        f"Évolution quantité sur 30 jours : "
        f"{(
            (trend['last_30_quantity']
             / trend['first_30_quantity']) - 1
        ) * 100:+.2f}%"
    )

    print()

    print(
        f"Quantité moyenne 90 premiers jours : "
        f"{trend['first_90_quantity']:.2f}"
    )

    print(
        f"Quantité moyenne 90 derniers jours : "
        f"{trend['last_90_quantity']:.2f}"
    )

    print(
        f"Évolution quantité sur 90 jours : "
        f"{(
            (trend['last_90_quantity']
             / trend['first_90_quantity']) - 1
        ) * 100:+.2f}%"
    )

    print("\n--- CA ---")

    print(
        f"CA moyen 30 premiers jours : "
        f"{trend['first_30_revenue']:,.2f} AR"
    )

    print(
        f"CA moyen 30 derniers jours : "
        f"{trend['last_30_revenue']:,.2f} AR"
    )

    print(
        f"Évolution CA sur 30 jours : "
        f"{(
            (trend['last_30_revenue']
             / trend['first_30_revenue']) - 1
        ) * 100:+.2f}%"
    )

    print()

    print(
        f"CA moyen 90 premiers jours : "
        f"{trend['first_90_revenue']:,.2f} AR"
    )

    print(
        f"CA moyen 90 derniers jours : "
        f"{trend['last_90_revenue']:,.2f} AR"
    )

    print(
        f"Évolution CA sur 90 jours : "
        f"{(
            (trend['last_90_revenue']
             / trend['first_90_revenue']) - 1
        ) * 100:+.2f}%"
    )


# ============================================================
# AFFICHAGE SAISONNALITÉ MENSUELLE
# ============================================================

def print_month_seasonality(monthly):
    """Affiche la saisonnalité mensuelle."""

    month_names = {
        1: "Janvier",
        2: "Février",
        3: "Mars",
        4: "Avril",
        5: "Mai",
        6: "Juin",
        7: "Juillet",
        8: "Août",
        9: "Septembre",
        10: "Octobre",
        11: "Novembre",
        12: "Décembre",
    }

    print("\n=== SAISONNALITÉ MENSUELLE ===")
    print("-" * 65)

    for _, row in monthly.iterrows():

        month_name = month_names[int(row["month_number"])]

        print(
            f"{month_name:<12} | "
            f"Qté moyenne : {row['average_quantity']:.2f} | "
            f"CA moyen : {row['average_revenue']:,.2f} AR"
        )

    strongest = monthly.loc[
        monthly["average_quantity"].idxmax()
    ]

    weakest = monthly.loc[
        monthly["average_quantity"].idxmin()
    ]

    print("\nMois structurellement le plus fort : "
          f"{month_names[int(strongest['month_number'])]} "
          f"({strongest['average_quantity']:.2f})")

    print("Mois structurellement le plus faible : "
          f"{month_names[int(weakest['month_number'])]} "
          f"({weakest['average_quantity']:.2f})")


# ============================================================
# AFFICHAGE SAISONNALITÉ HEBDOMADAIRE
# ============================================================

def print_weekday_seasonality(weekday):
    """Affiche la saisonnalité hebdomadaire."""

    weekday_names = {
        0: "Lundi",
        1: "Mardi",
        2: "Mercredi",
        3: "Jeudi",
        4: "Vendredi",
        5: "Samedi",
        6: "Dimanche",
    }

    print("\n=== SAISONNALITÉ HEBDOMADAIRE ===")
    print("-" * 65)

    for _, row in weekday.iterrows():

        name = weekday_names[int(row["weekday_number"])]

        print(
            f"{name:<12} | "
            f"Qté moyenne : {row['average_quantity']:.2f} | "
            f"CA moyen : {row['average_revenue']:,.2f} AR"
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_results(df, daily, monthly, weekday):
    """Vérifie la cohérence des résultats."""

    assert len(daily) == 365

    assert daily["total_quantity"].sum() == df["quantity"].sum()

    assert daily["total_revenue"].sum() == df["revenue"].sum()

    assert len(monthly) == 12

    assert len(weekday) == 7

    assert daily["ma_7"].notna().sum() == 359
    assert daily["ma_30"].notna().sum() == 336
    assert daily["ma_90"].notna().sum() == 276

    print("\n=== VALIDATION ===")
    print("[PASS] 365 jours présents")
    print("[PASS] Quantity quotidienne cohérente")
    print("[PASS] Revenue quotidien cohérent")
    print("[PASS] 12 mois calendaires présents")
    print("[PASS] 7 jours de la semaine présents")
    print("[PASS] Moyennes mobiles calculées correctement")


# ============================================================
# MAIN
# ============================================================

def main():
    """Point d'entrée principal."""

    print("AI Sales Forecasting")
    print("J4.3.4 - Analyse de la tendance et de la saisonnalité")
    print("-" * 75)

    df = load_data()

    daily = build_daily_data(df)

    daily = calculate_moving_averages(daily)

    trend = analyze_period_trend(daily)

    monthly = analyze_month_seasonality(df)

    weekday = analyze_weekday_seasonality(df)

    print_trend_analysis(daily, trend)

    print_month_seasonality(monthly)

    print_weekday_seasonality(weekday)

    validate_results(
        df,
        daily,
        monthly,
        weekday
    )

    print(
        "\nJ4.3.4 — TENDANCE / SAISONNALITÉ : OK"
    )


if __name__ == "__main__":
    main()