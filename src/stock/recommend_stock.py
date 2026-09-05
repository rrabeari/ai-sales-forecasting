"""
J8.3 — Moteur de recommandation de stock.

Transforme les prévisions J+1 → J+7 et le stock actuel
en recommandations de réapprovisionnement.

IMPORTANT :
Les données de stock sont synthétiques.
Le facteur de sécurité est une hypothèse métier du projet.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast"
    / "sales_forecast_j1_j7.csv"
)

STOCK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "stock_snapshot.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stock"

OUTPUT_FILE = OUTPUT_DIR / "stock_recommendation.csv"

SAFETY_STOCK_RATE = 0.20


# ---------------------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------------------

REQUIRED_FORECAST_COLUMNS = {
    "date",
    "product_id",
    "product_name",
    "category",
    "forecast_quantity",
    "unit_price",
}

REQUIRED_STOCK_COLUMNS = {
    "stock_snapshot_date",
    "product_id",
    "product_name",
    "category",
    "unit_price",
    "current_stock",
}


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

def validate_inputs(
    forecast: pd.DataFrame,
    stock: pd.DataFrame,
) -> None:
    """Validate forecast and stock inputs."""

    missing_forecast = (
        REQUIRED_FORECAST_COLUMNS - set(forecast.columns)
    )

    if missing_forecast:
        raise ValueError(
            "Colonnes manquantes dans le forecast : "
            f"{sorted(missing_forecast)}"
        )

    missing_stock = REQUIRED_STOCK_COLUMNS - set(stock.columns)

    if missing_stock:
        raise ValueError(
            "Colonnes manquantes dans le snapshot stock : "
            f"{sorted(missing_stock)}"
        )

    if forecast.empty:
        raise ValueError("Le forecast est vide.")

    if stock.empty:
        raise ValueError("Le snapshot stock est vide.")

    if forecast["product_id"].isna().any():
        raise ValueError(
            "product_id contient des NULL dans le forecast."
        )

    if stock["product_id"].isna().any():
        raise ValueError(
            "product_id contient des NULL dans le stock."
        )

    if forecast.duplicated(
        ["date", "product_id"]
    ).any():
        raise ValueError(
            "Doublons date + product_id dans le forecast."
        )

    if stock["product_id"].duplicated().any():
        raise ValueError(
            "Doublons product_id dans le snapshot stock."
        )

    if (forecast["forecast_quantity"] < 0).any():
        raise ValueError(
            "Quantités forecast négatives détectées."
        )

    if (stock["current_stock"] < 0).any():
        raise ValueError(
            "Stocks négatifs détectés."
        )

    if (stock["current_stock"] % 1 != 0).any():
        raise ValueError(
            "current_stock doit être entier."
        )


# ---------------------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------------------

def calculate_recommendations(
    forecast: pd.DataFrame,
    stock: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate stock recommendations per product."""

    forecast = forecast.copy()
    stock = stock.copy()

    # -------------------------------------------------------------
    # Forecast total J+1 → J+7 par produit
    # -------------------------------------------------------------

    forecast_summary = (
        forecast
        .groupby("product_id", as_index=False)
        .agg(
            forecast_7d=("forecast_quantity", "sum"),
            forecast_days=("date", "nunique"),
        )
    )

    # -------------------------------------------------------------
    # Vérification de l'horizon
    # -------------------------------------------------------------

    if not (forecast_summary["forecast_days"] == 7).all():
        raise ValueError(
            "Chaque produit doit avoir exactement 7 jours de forecast."
        )

    # -------------------------------------------------------------
    # Informations produit
    # -------------------------------------------------------------

    product_info = (
        forecast[
            [
                "product_id",
                "product_name",
                "category",
                "unit_price",
            ]
        ]
        .drop_duplicates("product_id")
    )

    # -------------------------------------------------------------
    # Fusion forecast + stock
    # -------------------------------------------------------------

    result = forecast_summary.merge(
        product_info,
        on="product_id",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        stock[
            [
                "product_id",
                "stock_snapshot_date",
                "current_stock",
            ]
        ],
        on="product_id",
        how="left",
        validate="one_to_one",
    )

    if result["current_stock"].isna().any():
        raise ValueError(
            "Certains produits forecast n'ont pas de stock correspondant."
        )

    # -------------------------------------------------------------
    # Calculs
    # -------------------------------------------------------------

    # Calcul des indicateurs de stock
    result["avg_daily_forecast"] = result["forecast_7d"] / 7
    result["safety_stock"] = result["forecast_7d"] * SAFETY_STOCK_RATE
    result["stock_target"] = (
        result["forecast_7d"] + result["safety_stock"]
    )

    # Arrondi des indicateurs avant le calcul final du réapprovisionnement
    result["forecast_7d"] = result["forecast_7d"].round(2)
    result["avg_daily_forecast"] = result["avg_daily_forecast"].round(2)
    result["safety_stock"] = result["safety_stock"].round(2)
    result["stock_target"] = result["stock_target"].round(2)

    # Quantité à commander basée sur le stock_target final
    result["reorder_quantity"] = (
        result["stock_target"] - result["current_stock"]
    ).clip(lower=0).round(0).astype(int)

    # Statut
    result["status"] = result["reorder_quantity"].apply(
        lambda x: "REORDER" if x > 0 else "OK"
    )

    result["status"] = result.apply(
        lambda row: (
            "REORDER"
            if row["reorder_quantity"] > 0
            else "OK"
        ),
        axis=1,
    )

    # -------------------------------------------------------------
    # Arrondis
    # -------------------------------------------------------------

    result["forecast_7d"] = result["forecast_7d"].round(2)
    result["avg_daily_forecast"] = (
        result["avg_daily_forecast"].round(2)
    )
    result["safety_stock"] = (
        result["safety_stock"].round(2)
    )
    result["stock_target"] = (
        result["stock_target"].round(2)
    )
    result["reorder_quantity"] = (
        result["reorder_quantity"].round(0).astype(int)
    )

    # -------------------------------------------------------------
    # Colonnes finales
    # -------------------------------------------------------------

    result = result[
        [
            "stock_snapshot_date",
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
        ]
    ].sort_values(
        by=["reorder_quantity", "forecast_7d"],
        ascending=[False, False],
    )

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------
# FINAL VALIDATION
# ---------------------------------------------------------------------

def validate_recommendations(
    result: pd.DataFrame,
) -> None:
    """Validate final recommendation dataset."""

    if len(result) != 14:
        raise ValueError(
            f"Nombre de produits incorrect : {len(result)} au lieu de 14."
        )

    if result["product_id"].duplicated().any():
        raise ValueError(
            "Doublons product_id dans les recommandations."
        )

    numeric_columns = [
        "current_stock",
        "forecast_7d",
        "avg_daily_forecast",
        "safety_stock",
        "stock_target",
        "reorder_quantity",
    ]

    if result[numeric_columns].isna().any().any():
        raise ValueError(
            "Des valeurs NULL sont présentes dans les calculs."
        )

    if (result["current_stock"] < 0).any():
        raise ValueError("Stock actuel négatif.")

    if (result["forecast_7d"] < 0).any():
        raise ValueError("Forecast négatif.")

    if (result["safety_stock"] < 0).any():
        raise ValueError("Stock de sécurité négatif.")

    if (result["stock_target"] < result["forecast_7d"]).any():
        raise ValueError(
            "Le stock cible doit être >= au forecast 7 jours."
        )

    if (result["reorder_quantity"] < 0).any():
        raise ValueError(
            "Quantité de réapprovisionnement négative."
        )

    expected_reorder = (
        result["stock_target"]
        - result["current_stock"]
    ).clip(lower=0)

    expected_reorder = (
        expected_reorder
        .round(0)
        .astype(int)
    )

    if not result["reorder_quantity"].equals(expected_reorder):
        raise ValueError(
            "Erreur dans le calcul de reorder_quantity."
        )

    expected_status = result["reorder_quantity"].apply(
        lambda x: "REORDER" if x > 0 else "OK"
    )

    if not (
        result["status"].values
        == expected_status.values
    ).all():
        raise ValueError(
            "Erreur dans le calcul du status."
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("J8.3 — RECOMMANDATION DE STOCK")
    print("=" * 70)

    print(f"[INFO] Forecast : {FORECAST_FILE}")
    print(f"[INFO] Stock    : {STOCK_FILE}")

    forecast = pd.read_csv(FORECAST_FILE)
    stock = pd.read_csv(STOCK_FILE)

    print(
        f"[OK] Forecast chargé : {len(forecast)} lignes"
    )
    print(
        f"[OK] Stock chargé    : {len(stock)} produits"
    )

    validate_inputs(
        forecast=forecast,
        stock=stock,
    )

    print("[PASS] Données sources validées")

    result = calculate_recommendations(
        forecast=forecast,
        stock=stock,
    )

    validate_recommendations(result)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    reorder_count = (
        result["status"] == "REORDER"
    ).sum()

    total_reorder = result[
        "reorder_quantity"
    ].sum()

    total_forecast = result[
        "forecast_7d"
    ].sum()

    print("-" * 70)
    print(
        f"[OK] Recommandations générées : {len(result)} produits"
    )
    print(
        f"[INFO] Demande prévue 7 jours : {total_forecast:.2f}"
    )
    print(
        f"[INFO] Produits à réapprovisionner : {reorder_count}"
    )
    print(
        f"[INFO] Quantité totale à commander : {total_reorder}"
    )
    print(
        f"[OK] Fichier sauvegardé : {OUTPUT_FILE}"
    )
    print("-" * 70)

    print("[PASS] 14 produits présents")
    print("[PASS] Forecast J+1 → J+7 validé")
    print("[PASS] Stock actuel aligné")
    print("[PASS] Stock de sécurité calculé")
    print("[PASS] Stock cible calculé")
    print("[PASS] Quantité de réapprovisionnement validée")
    print("[PASS] Statuts OK / REORDER validés")
    print("[PASS] Aucun NULL")
    print("[PASS] Aucune quantité négative")

    print("=" * 70)
    print("J8.3 — RECOMMANDATION STOCK : OK")
    print("=" * 70)


if __name__ == "__main__":
    main()