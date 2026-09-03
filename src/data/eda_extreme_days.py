"""
AI Sales Forecasting
J4.5.2 - Analyse des journées extrêmes

Objectif :
    Analyser les journées exceptionnellement fortes ou faibles
    afin de déterminer si elles correspondent à des anomalies
    ou à des variations cohérentes de la demande.

Aucune donnée n'est supprimée.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "eda"

OUTPUT_FILE = OUTPUT_DIR / "extreme_days_analysis.csv"


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
# AGRÉGATION JOURNALIÈRE
# ============================================================

def build_daily_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les indicateurs journaliers."""

    daily = (
        df.groupby("date")
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            active_products=("quantity", lambda x: (x > 0).sum()),
            total_products=("product_id", "nunique"),
        )
        .reset_index()
    )

    daily["avg_quantity_per_product"] = (
        daily["total_quantity"]
        / daily["active_products"]
    )

    daily["avg_revenue_per_product"] = (
        daily["total_revenue"]
        / daily["active_products"]
    )

    daily["day_of_week"] = daily["date"].dt.day_name(
        locale="English"
    )

    daily["day_of_week_num"] = daily["date"].dt.dayofweek

    daily["month"] = daily["date"].dt.month

    daily["month_name"] = daily["date"].dt.month_name(
        locale="English"
    )

    return daily


# ============================================================
# STATISTIQUES DE RÉFÉRENCE
# ============================================================

def calculate_reference_values(daily: pd.DataFrame):
    """Calcule les seuils de référence."""

    mean_quantity = daily["total_quantity"].mean()
    median_quantity = daily["total_quantity"].median()

    mean_revenue = daily["total_revenue"].mean()
    median_revenue = daily["total_revenue"].median()

    q1_quantity = daily["total_quantity"].quantile(0.25)
    q3_quantity = daily["total_quantity"].quantile(0.75)

    iqr_quantity = q3_quantity - q1_quantity

    upper_iqr = q3_quantity + 1.5 * iqr_quantity
    lower_iqr = q1_quantity - 1.5 * iqr_quantity

    p95_quantity = daily["total_quantity"].quantile(0.95)
    p99_quantity = daily["total_quantity"].quantile(0.99)

    print("\n=== RÉFÉRENCES JOURNALIÈRES ===")

    print(f"Quantité moyenne     : {mean_quantity:,.2f}")
    print(f"Quantité médiane     : {median_quantity:,.2f}")
    print(f"CA moyen             : {mean_revenue:,.2f} AR")
    print(f"CA médian            : {median_revenue:,.2f} AR")
    print(f"Q1 quantité          : {q1_quantity:,.2f}")
    print(f"Q3 quantité          : {q3_quantity:,.2f}")
    print(f"IQR quantité         : {iqr_quantity:,.2f}")
    print(f"Borne IQR supérieure : {upper_iqr:,.2f}")
    print(f"Borne IQR inférieure : {lower_iqr:,.2f}")
    print(f"P95 quantité         : {p95_quantity:,.2f}")
    print(f"P99 quantité         : {p99_quantity:,.2f}")

    return {
        "mean_quantity": mean_quantity,
        "median_quantity": median_quantity,
        "mean_revenue": mean_revenue,
        "median_revenue": median_revenue,
        "upper_iqr": upper_iqr,
        "lower_iqr": lower_iqr,
        "p95_quantity": p95_quantity,
        "p99_quantity": p99_quantity,
    }


# ============================================================
# CLASSIFICATION DES JOURNÉES
# ============================================================

def classify_days(daily: pd.DataFrame, reference: dict):
    """Classe les journées selon leur niveau de demande."""

    daily["quantity_vs_mean_pct"] = (
        (
            daily["total_quantity"]
            - reference["mean_quantity"]
        )
        / reference["mean_quantity"]
        * 100
    )

    daily["revenue_vs_mean_pct"] = (
        (
            daily["total_revenue"]
            - reference["mean_revenue"]
        )
        / reference["mean_revenue"]
        * 100
    )

    daily["above_iqr"] = (
        daily["total_quantity"]
        > reference["upper_iqr"]
    )

    daily["above_p95"] = (
        daily["total_quantity"]
        > reference["p95_quantity"]
    )

    daily["above_p99"] = (
        daily["total_quantity"]
        > reference["p99_quantity"]
    )

    daily["below_iqr"] = (
        daily["total_quantity"]
        < reference["lower_iqr"]
    )

    return daily


# ============================================================
# TOP / BOTTOM JOURNÉES
# ============================================================

def display_extreme_days(daily: pd.DataFrame):
    """Affiche les journées les plus fortes et les plus faibles."""

    columns = [
        "date",
        "total_quantity",
        "total_revenue",
        "active_products",
        "day_of_week",
        "month_name",
        "quantity_vs_mean_pct",
    ]

    print("\n=== TOP 10 JOURNÉES PAR QUANTITÉ ===")

    top_10 = daily.nlargest(
        10,
        "total_quantity"
    )[columns]

    print(
        top_10.to_string(
            index=False,
            formatters={
                "total_quantity": "{:,.0f}".format,
                "total_revenue": "{:,.0f}".format,
                "quantity_vs_mean_pct": "{:+.2f}%".format,
            },
        )
    )

    print("\n=== BOTTOM 10 JOURNÉES PAR QUANTITÉ ===")

    bottom_10 = daily.nsmallest(
        10,
        "total_quantity"
    )[columns]

    print(
        bottom_10.to_string(
            index=False,
            formatters={
                "total_quantity": "{:,.0f}".format,
                "total_revenue": "{:,.0f}".format,
                "quantity_vs_mean_pct": "{:+.2f}%".format,
            },
        )
    )


# ============================================================
# ANALYSE DES SEUILS
# ============================================================

def analyze_thresholds(daily: pd.DataFrame):
    """Analyse le nombre de journées dépassant les seuils."""

    print("\n=== ANALYSE DES SEUILS ===")

    iqr_count = daily["above_iqr"].sum()
    p95_count = daily["above_p95"].sum()
    p99_count = daily["above_p99"].sum()
    below_iqr_count = daily["below_iqr"].sum()

    print(f"Journées > borne IQR : {iqr_count:,}")
    print(f"Journées > P95       : {p95_count:,}")
    print(f"Journées > P99       : {p99_count:,}")
    print(f"Journées < borne IQR : {below_iqr_count:,}")

    print(
        f"\nPart des journées > P95 : "
        f"{p95_count / len(daily) * 100:.2f}%"
    )

    print(
        f"Part des journées > P99 : "
        f"{p99_count / len(daily) * 100:.2f}%"
    )


# ============================================================
# ANALYSE PAR JOUR DE LA SEMAINE
# ============================================================

def analyze_weekdays(daily: pd.DataFrame):
    """Analyse les journées extrêmes selon le jour de semaine."""

    print("\n=== JOURNÉES > P95 PAR JOUR DE LA SEMAINE ===")

    p95_days = daily[daily["above_p95"]]

    weekday_counts = (
        p95_days["day_of_week"]
        .value_counts()
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            fill_value=0,
        )
    )

    print(weekday_counts.to_string())


# ============================================================
# ANALYSE MENSUELLE
# ============================================================

def analyze_months(daily: pd.DataFrame):
    """Analyse les journées extrêmes par mois."""

    print("\n=== JOURNÉES > P95 PAR MOIS ===")

    p95_days = daily[daily["above_p95"]]

    monthly_counts = (
        p95_days.groupby("month")
        .size()
        .sort_index()
    )

    print(monthly_counts.to_string())


# ============================================================
# VALIDATION
# ============================================================

def validate(daily: pd.DataFrame):
    """Valide les résultats."""

    print("\n=== VALIDATION ===")

    checks = {
        "365 journées analysées": len(daily) == 365,
        "Aucune quantité négative": (
            daily["total_quantity"] >= 0
        ).all(),
        "Aucun CA négatif": (
            daily["total_revenue"] >= 0
        ).all(),
        "Toutes les dates sont uniques": (
            daily["date"].nunique() == len(daily)
        ),
        "Colonnes extrêmes présentes": all(
            column in daily.columns
            for column in [
                "above_iqr",
                "above_p95",
                "above_p99",
            ]
        ),
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
            "La validation J4.5.2 a échoué."
        )

    print("\n[PASS] Analyse des journées extrêmes validée")


# ============================================================
# SAUVEGARDE
# ============================================================

def save_results(daily: pd.DataFrame):
    """Sauvegarde l'analyse."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    daily.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== SORTIE ===")
    print(f"Fichier : {OUTPUT_FILE}")
    print(f"Lignes : {len(daily):,}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("AI Sales Forecasting")
    print("J4.5.2 - Analyse des journées extrêmes")
    print("-" * 60)

    df = load_data()

    print(f"\nLignes sources : {len(df):,}")

    daily = build_daily_dataset(df)

    print(f"Journées analysées : {len(daily):,}")

    reference = calculate_reference_values(daily)

    daily = classify_days(
        daily,
        reference
    )

    display_extreme_days(daily)

    analyze_thresholds(daily)

    analyze_weekdays(daily)

    analyze_months(daily)

    validate(daily)

    save_results(daily)

    print("\n" + "=" * 60)
    print("J4.5.2 — ANALYSE DES JOURNÉES EXTRÊMES : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()