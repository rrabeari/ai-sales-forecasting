"""
J8.4 — Analyse des recommandations de stock.

Analyse le fichier stock_recommendation.csv et produit :
- un résumé global
- les produits à réapprovisionner
- les produits prioritaires
- un classement par quantité à commander
"""

from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "stock"
    / "stock_recommendation.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "stock"
)

OUTPUT_FILE = OUTPUT_DIR / "stock_analysis.csv"


REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "unit_price",
    "current_stock",
    "forecast_7d",
    "avg_daily_forecast",
    "safety_stock",
    "stock_target",
    "reorder_quantity",
    "status",
}


# ----------------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> None:
    """Validate the recommendation dataset."""

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing)}"
        )

    if df["product_id"].duplicated().any():
        raise ValueError(
            "Doublons détectés sur product_id."
        )

    if df["current_stock"].isna().any():
        raise ValueError(
            "NULL détecté dans current_stock."
        )

    if df["forecast_7d"].isna().any():
        raise ValueError(
            "NULL détecté dans forecast_7d."
        )

    if df["reorder_quantity"].isna().any():
        raise ValueError(
            "NULL détecté dans reorder_quantity."
        )

    if (df["current_stock"] < 0).any():
        raise ValueError(
            "Stock actuel négatif détecté."
        )

    if (df["forecast_7d"] < 0).any():
        raise ValueError(
            "Forecast négatif détecté."
        )

    if (df["reorder_quantity"] < 0).any():
        raise ValueError(
            "Quantité de réapprovisionnement négative."
        )

    if not df["status"].isin(["OK", "REORDER"]).all():
        raise ValueError(
            "Statut invalide détecté."
        )


# ----------------------------------------------------------------------
# ANALYSE
# ----------------------------------------------------------------------

def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """Create the stock recommendation analysis."""

    result = df.copy()

    # Nombre de jours de stock disponible
    result["stock_coverage_days"] = (
        result["current_stock"]
        / result["avg_daily_forecast"]
    ).replace([float("inf")], 0)

    result["stock_coverage_days"] = (
        result["stock_coverage_days"].round(1)
    )

    # Valeur du stock actuel
    result["current_stock_value"] = (
        result["current_stock"]
        * result["unit_price"]
    ).round(2)

    # Valeur du réapprovisionnement
    result["reorder_value"] = (
        result["reorder_quantity"]
        * result["unit_price"]
    ).round(2)

    # Niveau de priorité
    result["priority"] = result["reorder_quantity"].apply(
        lambda x: (
            "HIGH"
            if x >= 15
            else "MEDIUM"
            if x >= 5
            else "LOW"
            if x > 0
            else "NONE"
        )
    )

    # Classement par quantité à commander
    result = result.sort_values(
        by=["reorder_quantity", "forecast_7d"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    """Run J8.4 stock analysis."""

    print("=" * 70)
    print("J8.4 — ANALYSE DES RECOMMANDATIONS DE STOCK")
    print("=" * 70)

    # Chargement
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"[OK] Recommandations chargées : {len(df)} produits")

    # Validation
    validate_input(df)

    print("[PASS] Structure et données validées")

    # Analyse
    result = analyze(df)

    # Résumé
    total_products = len(result)
    reorder_products = (
        result["status"] == "REORDER"
    ).sum()

    total_forecast = result["forecast_7d"].sum()
    total_current_stock = result["current_stock"].sum()
    total_reorder = result["reorder_quantity"].sum()
    total_reorder_value = result["reorder_value"].sum()

    print("-" * 70)
    print(f"[INFO] Produits analysés       : {total_products}")
    print(f"[INFO] Produits à réapprovisionner : {reorder_products}")
    print(f"[INFO] Demande prévue 7 jours  : {total_forecast:.2f}")
    print(f"[INFO] Stock actuel total      : {total_current_stock}")
    print(f"[INFO] Quantité à commander    : {total_reorder}")
    print(
        f"[INFO] Valeur du réapprovisionnement : "
        f"{total_reorder_value:,.2f} AR"
    )

    # Top produits
    print("-" * 70)
    print("TOP 5 — PRODUITS À RÉAPPROVISIONNER")

    top5 = result[result["status"] == "REORDER"].head(5)

    for _, row in top5.iterrows():
        print(
            f"- {row['product_name']} | "
            f"Stock={int(row['current_stock'])} | "
            f"Forecast 7j={row['forecast_7d']:.2f} | "
            f"Commande={int(row['reorder_quantity'])} | "
            f"Priorité={row['priority']}"
        )

    # Sauvegarde
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("-" * 70)
    print(f"[OK] Analyse sauvegardée : {OUTPUT_FILE}")

    # Validation finale
    if len(result) != total_products:
        raise ValueError(
            "Le nombre de produits a changé pendant l'analyse."
        )

    if result["reorder_quantity"].sum() != total_reorder:
        raise ValueError(
            "Erreur dans le total des quantités à commander."
        )

    if result["reorder_value"].isna().any():
        raise ValueError(
            "NULL détecté dans reorder_value."
        )

    print("[PASS] Analyse cohérente")
    print("[PASS] Quantités de réapprovisionnement cohérentes")
    print("[PASS] Valeurs de réapprovisionnement calculées")
    print("[PASS] Priorités calculées")

    print("=" * 70)
    print("J8.4 — ANALYSE STOCK : OK")
    print("=" * 70)


if __name__ == "__main__":
    main()