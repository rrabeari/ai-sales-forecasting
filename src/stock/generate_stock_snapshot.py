"""
J8.2 — Génération du snapshot de stock synthétique.

Ce fichier génère un état de stock fictif et reproductible
pour les 14 produits du projet AI Sales Forecasting.

IMPORTANT :
Les stocks générés sont SYNTHÉTIQUES et ne représentent pas
le stock réel de KShop.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SALES_FILE = PROJECT_ROOT / "data" / "processed" / "sales_clean.csv"
FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast"
    / "sales_forecast_j1_j7.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stock"
OUTPUT_FILE = OUTPUT_DIR / "stock_snapshot.csv"

RANDOM_SEED = 42


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

REQUIRED_SALES_COLUMNS = {
    "date",
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
}

REQUIRED_FORECAST_COLUMNS = {
    "date",
    "product_id",
    "product_name",
    "category",
    "unit_price",
}


def validate_input_data(sales: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Validate source datasets."""

    missing_sales = REQUIRED_SALES_COLUMNS - set(sales.columns)
    if missing_sales:
        raise ValueError(
            f"Colonnes manquantes dans sales_clean.csv : {sorted(missing_sales)}"
        )

    missing_forecast = REQUIRED_FORECAST_COLUMNS - set(forecast.columns)
    if missing_forecast:
        raise ValueError(
            "Colonnes manquantes dans le forecast : "
            f"{sorted(missing_forecast)}"
        )

    if sales.empty:
        raise ValueError("Le dataset historique est vide.")

    if forecast.empty:
        raise ValueError("Le forecast est vide.")

    if sales["product_id"].isna().any():
        raise ValueError("product_id contient des NULL dans sales_clean.")

    if forecast["product_id"].isna().any():
        raise ValueError("product_id contient des NULL dans le forecast.")

    if (sales["quantity"] < 0).any():
        raise ValueError("Des quantités historiques négatives existent.")

    if forecast.duplicated(["date", "product_id"]).any():
        raise ValueError(
            "Doublons détectés sur la combinaison date + product_id dans le forecast."
        )


# ---------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------

def generate_stock_snapshot(
    sales: pd.DataFrame,
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Génère un stock synthétique pour chaque produit.

    Le stock est basé sur la demande quotidienne moyenne récente.
    Une variation contrôlée est ajoutée afin d'obtenir des niveaux
    de stock réalistes et reproductibles.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    sales = sales.copy()
    forecast = forecast.copy()

    sales["date"] = pd.to_datetime(sales["date"])

    latest_date = sales["date"].max()

    # On utilise les 30 derniers jours historiques.
    recent_start = latest_date - pd.Timedelta(days=29)

    recent_sales = sales[
        sales["date"].between(recent_start, latest_date)
    ].copy()

    # Moyenne quotidienne par produit sur les 30 derniers jours.
    recent_demand = (
        recent_sales
        .groupby("product_id", as_index=False)
        .agg(
            avg_daily_historical_demand=("quantity", "mean")
        )
    )

    result = forecast[
        [
            "product_id",
            "product_name",
            "category",
            "unit_price",
        ]
    ].drop_duplicates("product_id").copy()

    result = result.merge(
        recent_demand,
        on="product_id",
        how="left",
    )

    if result["avg_daily_historical_demand"].isna().any():
        raise ValueError(
            "Certains produits du forecast ne possèdent pas "
            "de demande historique récente."
        )

    # Facteur de stock synthétique.
    #
    # Le stock initial représente environ 2 à 8 jours
    # de demande historique récente.
    stock_days = rng.uniform(
        low=2.0,
        high=8.0,
        size=len(result),
    )

    result["current_stock"] = np.rint(
        result["avg_daily_historical_demand"] * stock_days
    ).astype(int)

    # Garantit un stock >= 0.
    result["current_stock"] = result["current_stock"].clip(lower=0)

    result["stock_snapshot_date"] = latest_date.strftime("%Y-%m-%d")

    result = result[
        [
            "stock_snapshot_date",
            "product_id",
            "product_name",
            "category",
            "unit_price",
            "avg_daily_historical_demand",
            "current_stock",
        ]
    ]

    return result


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("J8.2 — GÉNÉRATION DU SNAPSHOT DE STOCK SYNTHÉTIQUE")
    print("=" * 70)

    print(f"[INFO] Source historique : {SALES_FILE}")
    print(f"[INFO] Source forecast    : {FORECAST_FILE}")

    sales = pd.read_csv(SALES_FILE)
    forecast = pd.read_csv(FORECAST_FILE)

    print(f"[OK] Historique chargé : {len(sales):,} lignes")
    print(f"[OK] Forecast chargé   : {len(forecast):,} lignes")

    validate_input_data(sales, forecast)

    print("[PASS] Données sources validées")

    snapshot = generate_stock_snapshot(
        sales=sales,
        forecast=forecast,
    )

    # -------------------------------------------------------------
    # VALIDATION FINALE
    # -------------------------------------------------------------

    if len(snapshot) != 14:
        raise ValueError(
            f"Nombre de produits incorrect : {len(snapshot)} au lieu de 14."
        )

    if snapshot["product_id"].duplicated().any():
        raise ValueError("Doublons product_id détectés.")

    if snapshot["current_stock"].isna().any():
        raise ValueError("NULL détecté dans current_stock.")

    if (snapshot["current_stock"] < 0).any():
        raise ValueError("Stock négatif détecté.")

    if not pd.api.types.is_integer_dtype(snapshot["current_stock"]):
        raise ValueError("current_stock doit être entier.")

    forecast_products = set(forecast["product_id"])
    snapshot_products = set(snapshot["product_id"])

    if forecast_products != snapshot_products:
        raise ValueError(
            "Les produits du snapshot ne correspondent pas "
            "aux produits du forecast."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("-" * 70)
    print(f"[OK] Snapshot généré : {len(snapshot)} produits")
    print(f"[OK] Stock total      : {snapshot['current_stock'].sum():,}")
    print(f"[OK] Fichier sauvegardé : {OUTPUT_FILE}")
    print("-" * 70)

    print("[PASS] 14 produits présents")
    print("[PASS] Aucun doublon product_id")
    print("[PASS] Aucun NULL")
    print("[PASS] Aucun stock négatif")
    print("[PASS] current_stock entier")
    print("[PASS] Produits alignés avec le forecast")
    print("[PASS] Snapshot clairement synthétique")
    print("=" * 70)
    print("J8.2 — SNAPSHOT STOCK : OK")
    print("=" * 70)


if __name__ == "__main__":
    main()