"""
AI Sales Forecasting
J4.5.4 - Analyse des anomalies par produit

Objectif :
Analyser les valeurs atypiques de quantité pour chaque produit
par rapport à sa propre distribution de demande.

Aucune observation n'est supprimée.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/sales_clean.csv")
OUTPUT_FILE = Path(
    "data/processed/eda/product_anomalies_analysis.csv"
)


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
# ANALYSE PAR PRODUIT
# ============================================================

def analyze_product_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse les anomalies de quantité pour chaque produit.
    """

    results = []

    for product_id, group in df.groupby("product_id"):

        group = group.sort_values("date").copy()

        product_name = group["product_name"].iloc[0]
        category = group["category"].iloc[0]

        quantity = group["quantity"]

        # ----------------------------------------------------
        # Statistiques centrales
        # ----------------------------------------------------

        mean_qty = quantity.mean()
        median_qty = quantity.median()

        min_qty = quantity.min()
        max_qty = quantity.max()

        q1 = quantity.quantile(0.25)
        q3 = quantity.quantile(0.75)

        iqr = q3 - q1

        # ----------------------------------------------------
        # Bornes IQR
        # ----------------------------------------------------

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # ----------------------------------------------------
        # Percentiles
        # ----------------------------------------------------

        p95 = quantity.quantile(0.95)
        p99 = quantity.quantile(0.99)

        # ----------------------------------------------------
        # Détection IQR
        # ----------------------------------------------------

        iqr_anomalies = quantity[
            (quantity < lower_bound)
            | (quantity > upper_bound)
        ]

        iqr_anomaly_count = len(iqr_anomalies)

        iqr_anomaly_rate = (
            iqr_anomaly_count / len(quantity) * 100
        )

        # ----------------------------------------------------
        # Détection P95 / P99
        # ----------------------------------------------------

        p95_count = (quantity > p95).sum()
        p99_count = (quantity > p99).sum()

        p95_rate = p95_count / len(quantity) * 100
        p99_rate = p99_count / len(quantity) * 100

        # ----------------------------------------------------
        # Zéros
        # ----------------------------------------------------

        zero_days = (quantity == 0).sum()

        zero_rate = zero_days / len(quantity) * 100

        # ----------------------------------------------------
        # Variabilité
        # ----------------------------------------------------

        std_qty = quantity.std()

        if mean_qty != 0:
            coefficient_variation = std_qty / mean_qty
        else:
            coefficient_variation = 0

        # ----------------------------------------------------
        # Date de la demande maximale
        # ----------------------------------------------------

        max_row = group.loc[
            group["quantity"].idxmax()
        ]

        max_quantity_date = max_row["date"].date()

        # ----------------------------------------------------
        # Niveau de variabilité
        # ----------------------------------------------------

        if coefficient_variation < 0.50:
            variability_level = "Faible"
        elif coefficient_variation < 0.70:
            variability_level = "Modérée"
        else:
            variability_level = "Élevée"

        # ----------------------------------------------------
        # Niveau d'anomalies
        # ----------------------------------------------------

        if iqr_anomaly_rate == 0:
            anomaly_level = "Aucune"
        elif iqr_anomaly_rate < 2:
            anomaly_level = "Faible"
        elif iqr_anomaly_rate < 5:
            anomaly_level = "Modérée"
        else:
            anomaly_level = "Élevée"

        # ----------------------------------------------------
        # Résultat
        # ----------------------------------------------------

        results.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "observations": len(quantity),
                "mean_quantity": round(mean_qty, 2),
                "median_quantity": round(median_qty, 2),
                "min_quantity": int(min_qty),
                "max_quantity": int(max_qty),
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_iqr_bound": round(lower_bound, 2),
                "upper_iqr_bound": round(upper_bound, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2),
                "iqr_anomaly_count": int(iqr_anomaly_count),
                "iqr_anomaly_rate_pct": round(
                    iqr_anomaly_rate, 2
                ),
                "p95_count": int(p95_count),
                "p95_rate_pct": round(p95_rate, 2),
                "p99_count": int(p99_count),
                "p99_rate_pct": round(p99_rate, 2),
                "zero_days": int(zero_days),
                "zero_rate_pct": round(zero_rate, 2),
                "std_quantity": round(std_qty, 2),
                "coefficient_variation": round(
                    coefficient_variation, 2
                ),
                "variability_level": variability_level,
                "anomaly_level": anomaly_level,
                "max_quantity_date": max_quantity_date,
            }
        )

    return pd.DataFrame(results)


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    df: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    """Valide le rapport d'anomalies."""

    print("\n=== VALIDATION ===")

    # Dataset source
    assert not df.empty
    print("[PASS] Dataset source non vide")

    # Nombre de produits
    assert df["product_id"].nunique() == 14
    print("[PASS] 14 produits analysés")

    # Nombre de résultats
    assert len(results) == 14
    print("[PASS] 14 rapports produits générés")

    # Observations par produit
    assert (results["observations"] == 365).all()
    print("[PASS] 365 observations par produit")

    # Quantités
    assert (df["quantity"] >= 0).all()
    print("[PASS] Quantités non négatives")

    # Pas de NULL dans les résultats critiques
    critical_columns = [
        "product_id",
        "product_name",
        "mean_quantity",
        "median_quantity",
        "min_quantity",
        "max_quantity",
        "iqr_anomaly_count",
        "p95_count",
        "p99_count",
    ]

    assert not results[critical_columns].isnull().any().any()
    print("[PASS] Résultats critiques sans NULL")

    # Cohérence min / max
    assert (
        results["min_quantity"]
        <= results["median_quantity"]
    ).all()

    assert (
        results["median_quantity"]
        <= results["max_quantity"]
    ).all()

    print("[PASS] Statistiques min / médiane / max cohérentes")

    # Comptages positifs
    assert (results["iqr_anomaly_count"] >= 0).all()
    assert (results["p95_count"] >= 0).all()
    assert (results["p99_count"] >= 0).all()

    print("[PASS] Comptages d'anomalies cohérents")

    print("[PASS] Analyse des anomalies par produit validée")


# ============================================================
# AFFICHAGE
# ============================================================

def display_summary(results: pd.DataFrame) -> None:
    """Affiche les principaux résultats."""

    print("\n=== RÉSUMÉ DES ANOMALIES PAR PRODUIT ===")

    display_columns = [
        "product_name",
        "mean_quantity",
        "max_quantity",
        "iqr_anomaly_count",
        "iqr_anomaly_rate_pct",
        "p95_count",
        "p99_count",
        "zero_days",
        "coefficient_variation",
        "variability_level",
        "anomaly_level",
    ]

    print(
        results[
            display_columns
        ].sort_values(
            "iqr_anomaly_rate_pct",
            ascending=False,
        ).to_string(index=False)
    )

    print("\n=== PRODUITS AVEC LE PLUS D'ANOMALIES IQR ===")

    top_anomalies = results.sort_values(
        "iqr_anomaly_rate_pct",
        ascending=False,
    ).head(5)

    for _, row in top_anomalies.iterrows():

        print(
            f"{row['product_name']} | "
            f"{row['iqr_anomaly_count']} anomalies | "
            f"{row['iqr_anomaly_rate_pct']:.2f}% | "
            f"max={row['max_quantity']} | "
            f"CV={row['coefficient_variation']:.2f}"
        )

    print("\n=== PRODUITS LES PLUS VARIABLES ===")

    top_variable = results.sort_values(
        "coefficient_variation",
        ascending=False,
    ).head(5)

    for _, row in top_variable.iterrows():

        print(
            f"{row['product_name']} | "
            f"CV={row['coefficient_variation']:.2f} | "
            f"anomalies IQR="
            f"{row['iqr_anomaly_count']}"
        )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_results(results: pd.DataFrame) -> None:
    """Sauvegarde le rapport."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Point d'entrée principal."""

    print("AI Sales Forecasting")
    print("J4.5.4 - Analyse des anomalies par produit")
    print("-" * 60)

    df = load_data()

    print(f"\nLignes analysées : {len(df):,}")
    print(
        f"Produits analysés : "
        f"{df['product_id'].nunique()}"
    )
    print(
        f"Période : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    results = analyze_product_anomalies(df)

    display_summary(results)

    validate_results(df, results)

    save_results(results)

    print("\n=== SORTIE ===")
    print(f"Fichier : {OUTPUT_FILE}")
    print(f"Lignes du rapport : {len(results)}")

    print("\n" + "=" * 60)
    print("J4.5.4 — ANOMALIES PAR PRODUIT : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()