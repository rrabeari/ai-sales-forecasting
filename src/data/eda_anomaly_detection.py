"""
AI Sales Forecasting
J4.5.1 - Détection des valeurs atypiques

Objectif :
    Identifier les observations atypiques du dataset nettoyé
    sans supprimer aucune donnée.

Méthodes :
    - IQR
    - P95
    - P99
    - Analyse globale
    - Analyse par produit
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "eda"

OUTPUT_FILE = OUTPUT_DIR / "anomaly_detection_report.csv"


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
# CALCUL IQR
# ============================================================

def calculate_iqr_bounds(series: pd.Series):
    """
    Calcule les bornes IQR.

    Lower bound = Q1 - 1.5 × IQR
    Upper bound = Q3 + 1.5 × IQR
    """

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return q1, q3, iqr, lower_bound, upper_bound


# ============================================================
# ANALYSE GLOBALE
# ============================================================

def analyze_global(df: pd.DataFrame):
    """Analyse les valeurs atypiques sur l'ensemble du dataset."""

    print("\n=== ANALYSE GLOBALE ===")

    results = []

    for column in ["quantity", "revenue"]:

        series = df[column]

        q1, q3, iqr, lower, upper = calculate_iqr_bounds(series)

        p95 = series.quantile(0.95)
        p99 = series.quantile(0.99)

        iqr_outliers = ((series < lower) | (series > upper)).sum()
        above_p95 = (series > p95).sum()
        above_p99 = (series > p99).sum()

        print(f"\n--- {column.upper()} ---")
        print(f"Q1                 : {q1:,.2f}")
        print(f"Q3                 : {q3:,.2f}")
        print(f"IQR                : {iqr:,.2f}")
        print(f"Borne inférieure   : {lower:,.2f}")
        print(f"Borne supérieure   : {upper:,.2f}")
        print(f"P95                : {p95:,.2f}")
        print(f"P99                : {p99:,.2f}")
        print(f"Valeurs IQR        : {iqr_outliers:,}")
        print(f"Valeurs > P95      : {above_p95:,}")
        print(f"Valeurs > P99      : {above_p99:,}")

        results.append(
            {
                "scope": "global",
                "product_id": None,
                "product_name": None,
                "metric": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "p95": p95,
                "p99": p99,
                "iqr_outliers": iqr_outliers,
                "above_p95": above_p95,
                "above_p99": above_p99,
            }
        )

    return results


# ============================================================
# ANALYSE PAR PRODUIT
# ============================================================

def analyze_by_product(df: pd.DataFrame):
    """Analyse les valeurs atypiques pour chaque produit."""

    print("\n=== ANALYSE PAR PRODUIT ===")

    results = []

    for (product_id, product_name), group in df.groupby(
        ["product_id", "product_name"]
    ):

        series = group["quantity"]

        q1, q3, iqr, lower, upper = calculate_iqr_bounds(series)

        p95 = series.quantile(0.95)
        p99 = series.quantile(0.99)

        iqr_outliers = ((series < lower) | (series > upper)).sum()
        above_p95 = (series > p95).sum()
        above_p99 = (series > p99).sum()

        results.append(
            {
                "scope": "product",
                "product_id": product_id,
                "product_name": product_name,
                "metric": "quantity",
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "p95": p95,
                "p99": p99,
                "iqr_outliers": iqr_outliers,
                "above_p95": above_p95,
                "above_p99": above_p99,
            }
        )

    return results


# ============================================================
# JOURNÉES EXTRÊMES
# ============================================================

def analyze_extreme_days(df: pd.DataFrame):
    """Identifie les journées avec les quantités les plus élevées."""

    print("\n=== JOURNÉES EXTRÊMES ===")

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
        )
        .sort_values("total_quantity", ascending=False)
    )

    print("\nTop 10 journées par quantité :")

    print(
        daily.head(10).to_string(
            index=False,
            formatters={
                "total_quantity": "{:,.0f}".format,
                "total_revenue": "{:,.0f}".format,
            },
        )
    )

    return daily


# ============================================================
# VALIDATION
# ============================================================

def validate_results(df: pd.DataFrame, results: list):
    """Effectue les contrôles finaux."""

    print("\n=== VALIDATION ===")

    checks = {
        "Dataset non vide": len(df) > 0,
        "Quantité sans valeur négative": (df["quantity"] >= 0).all(),
        "CA sans valeur négative": (df["revenue"] >= 0).all(),
        "Résultats générés": len(results) > 0,
    }

    all_pass = True

    for check_name, status in checks.items():

        if status:
            print(f"[PASS] {check_name}")
        else:
            print(f"[FAIL] {check_name}")
            all_pass = False

    if not all_pass:
        raise ValueError(
            "La validation de J4.5.1 a échoué."
        )

    print("\n[PASS] Détection des anomalies validée")


# ============================================================
# SAUVEGARDE
# ============================================================

def save_results(results: list):
    """Sauvegarde le rapport d'anomalies."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(results)

    report.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== SORTIE ===")
    print(f"Fichier : {OUTPUT_FILE}")
    print(f"Lignes du rapport : {len(report):,}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("AI Sales Forecasting")
    print("J4.5.1 - Détection des valeurs atypiques")
    print("-" * 60)

    df = load_data()

    print(f"\nLignes analysées   : {len(df):,}")
    print(f"Produits analysés  : {df['product_id'].nunique():,}")
    print(
        f"Période            : "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )

    global_results = analyze_global(df)

    product_results = analyze_by_product(df)

    all_results = global_results + product_results

    analyze_extreme_days(df)

    validate_results(df, all_results)

    save_results(all_results)

    print("\n" + "=" * 60)
    print("J4.5.1 — DÉTECTION DES ANOMALIES : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()