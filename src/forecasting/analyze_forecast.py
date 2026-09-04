"""
J7.5 — Analyse des prévisions J+1 → J+7.

Analyse :
- demande totale par jour
- demande totale par produit
- CA prévisionnel par jour
- CA prévisionnel par produit
- classement des produits
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_FILE = Path(
    "data/processed/forecast/sales_forecast_j1_j7.csv"
)

OUTPUT_DIR = Path("data/processed/forecast")

DAILY_FILE = OUTPUT_DIR / "forecast_daily_summary.csv"
PRODUCT_FILE = OUTPUT_DIR / "forecast_product_summary.csv"


# ============================================================
# CHARGEMENT
# ============================================================

def load_forecast() -> pd.DataFrame:
    """Charge et valide le fichier de prévisions."""

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FORECAST_FILE}"
        )

    df = pd.read_csv(FORECAST_FILE)

    required_columns = [
        "date",
        "product_id",
        "product_name",
        "category",
        "forecast_quantity",
        "unit_price",
        "forecast_revenue",
        "forecast_day",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    return df


# ============================================================
# ANALYSE PAR JOUR
# ============================================================

def analyze_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les prévisions par jour."""

    daily = (
        df.groupby(
            ["date", "forecast_day"],
            as_index=False
        )
        .agg(
            forecast_quantity=(
                "forecast_quantity",
                "sum"
            ),
            forecast_revenue=(
                "forecast_revenue",
                "sum"
            ),
            products=("product_id", "nunique"),
        )
        .sort_values("date")
    )

    daily["average_quantity_per_product"] = (
        daily["forecast_quantity"]
        / daily["products"]
    )

    return daily


# ============================================================
# ANALYSE PAR PRODUIT
# ============================================================

def analyze_products(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les prévisions par produit."""

    products = (
        df.groupby(
            [
                "product_id",
                "product_name",
                "category",
                "unit_price",
            ],
            as_index=False,
        )
        .agg(
            forecast_quantity=(
                "forecast_quantity",
                "sum"
            ),
            forecast_revenue=(
                "forecast_revenue",
                "sum"
            ),
            forecast_days=(
                "date",
                "nunique"
            ),
        )
        .sort_values(
            "forecast_quantity",
            ascending=False,
        )
    )

    products["average_daily_quantity"] = (
        products["forecast_quantity"]
        / products["forecast_days"]
    )

    products["revenue_share_pct"] = (
        products["forecast_revenue"]
        / products["forecast_revenue"].sum()
        * 100
    )

    return products


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    products: pd.DataFrame,
) -> None:
    """Valide les résultats d'analyse."""

    expected_rows = 98

    if len(df) != expected_rows:
        raise ValueError(
            f"Nombre de lignes inattendu : {len(df)}"
        )

    if len(daily) != 7:
        raise ValueError(
            f"Nombre de jours inattendu : {len(daily)}"
        )

    if len(products) != 14:
        raise ValueError(
            f"Nombre de produits inattendu : {len(products)}"
        )

    if daily["forecast_quantity"].isna().any():
        raise ValueError(
            "Quantité NULL dans analyse quotidienne."
        )

    if products["forecast_quantity"].isna().any():
        raise ValueError(
            "Quantité NULL dans analyse produits."
        )

    # Vérification de conservation des totaux.
    original_quantity = df["forecast_quantity"].sum()
    daily_quantity = daily["forecast_quantity"].sum()
    product_quantity = products["forecast_quantity"].sum()

    if abs(original_quantity - daily_quantity) > 0.01:
        raise ValueError(
            "Erreur de conservation des quantités (jour)."
        )

    if abs(original_quantity - product_quantity) > 0.01:
        raise ValueError(
            "Erreur de conservation des quantités (produit)."
        )

    original_revenue = df["forecast_revenue"].sum()
    daily_revenue = daily["forecast_revenue"].sum()
    product_revenue = products["forecast_revenue"].sum()

    if abs(original_revenue - daily_revenue) > 0.01:
        raise ValueError(
            "Erreur de conservation du CA (jour)."
        )

    if abs(original_revenue - product_revenue) > 0.01:
        raise ValueError(
            "Erreur de conservation du CA (produit)."
        )


# ============================================================
# AFFICHAGE
# ============================================================

def print_analysis(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    products: pd.DataFrame,
) -> None:
    """Affiche les principaux résultats."""

    total_quantity = df["forecast_quantity"].sum()
    total_revenue = df["forecast_revenue"].sum()

    best_day = daily.loc[
        daily["forecast_quantity"].idxmax()
    ]

    best_product = products.iloc[0]

    print("=" * 70)
    print("J7.5 — ANALYSE DES PRÉVISIONS J+1 → J+7")
    print("=" * 70)

    print(f"[INFO] Prévisions analysées : {len(df)}")
    print(f"[INFO] Produits : {df['product_id'].nunique()}")
    print(f"[INFO] Jours : {df['date'].nunique()}")

    print()
    print("[GLOBAL]")
    print(
        f"Quantité totale prévue : "
        f"{total_quantity:.2f}"
    )
    print(
        f"CA total prévu : "
        f"{total_revenue:,.2f} AR"
    )

    print()
    print("[MEILLEUR JOUR]")
    print(
        f"Date : "
        f"{best_day['date'].strftime('%Y-%m-%d')}"
    )
    print(
        f"Forecast day : "
        f"{best_day['forecast_day']}"
    )
    print(
        f"Quantité : "
        f"{best_day['forecast_quantity']:.2f}"
    )
    print(
        f"CA : "
        f"{best_day['forecast_revenue']:,.2f} AR"
    )

    print()
    print("[TOP 5 PRODUITS — QUANTITÉ]")
    print("-" * 70)

    top5 = products.head(5)

    for rank, (_, row) in enumerate(
        top5.iterrows(),
        start=1,
    ):
        print(
            f"{rank}. "
            f"{row['product_name']} | "
            f"Qté : {row['forecast_quantity']:.2f} | "
            f"CA : {row['forecast_revenue']:,.2f} AR"
        )

    print()
    print("[TOP 5 PRODUITS — CA]")
    print("-" * 70)

    top5_revenue = products.sort_values(
        "forecast_revenue",
        ascending=False,
    ).head(5)

    for rank, (_, row) in enumerate(
        top5_revenue.iterrows(),
        start=1,
    ):
        print(
            f"{rank}. "
            f"{row['product_name']} | "
            f"CA : {row['forecast_revenue']:,.2f} AR | "
            f"Qté : {row['forecast_quantity']:.2f}"
        )

    print()
    print("[ANALYSE PAR JOUR]")
    print("-" * 70)

    print(
        daily[
            [
                "date",
                "forecast_day",
                "forecast_quantity",
                "forecast_revenue",
            ]
        ].to_string(index=False)
    )

    print("=" * 70)
    print("J7.5 — ANALYSE : TERMINÉE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Point d'entrée principal."""

    df = load_forecast()

    print(
        f"[OK] Forecast chargé : "
        f"{len(df)} lignes"
    )

    daily = analyze_daily(df)
    products = analyze_products(df)

    validate_results(
        df,
        daily,
        products,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        DAILY_FILE,
        index=False,
    )

    products.to_csv(
        PRODUCT_FILE,
        index=False,
    )

    print(
        f"[OK] Analyse quotidienne sauvegardée : "
        f"{DAILY_FILE}"
    )

    print(
        f"[OK] Analyse produits sauvegardée : "
        f"{PRODUCT_FILE}"
    )

    print_analysis(
        df,
        daily,
        products,
    )


if __name__ == "__main__":
    main()