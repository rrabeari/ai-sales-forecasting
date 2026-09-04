"""
J7.6 — Visualisation des prévisions J+1 → J+7.

Génère :
- demande prévisionnelle par jour
- CA prévisionnel par jour
- Top 10 produits par quantité
- Top 10 produits par CA
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_FILE = Path(
    "data/processed/forecast/sales_forecast_j1_j7.csv"
)

OUTPUT_DIR = Path(
    "data/processed/forecast/visualizations"
)


# ============================================================
# CHARGEMENT
# ============================================================

def load_forecast() -> pd.DataFrame:
    """Charge le fichier de prévisions."""

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FORECAST_FILE}"
        )

    df = pd.read_csv(FORECAST_FILE)

    required_columns = [
        "date",
        "product_id",
        "product_name",
        "forecast_quantity",
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
# GRAPHIQUE 1 — QUANTITÉ PAR JOUR
# ============================================================

def plot_quantity_by_day(df: pd.DataFrame) -> None:
    """Crée le graphique de demande prévisionnelle quotidienne."""

    daily = (
        df.groupby(
            ["date", "forecast_day"],
            as_index=False
        )["forecast_quantity"]
        .sum()
        .sort_values("date")
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        daily["forecast_day"],
        daily["forecast_quantity"],
        marker="o",
        linewidth=2,
    )

    plt.title(
        "Prévision de la demande — J+1 à J+7"
    )

    plt.xlabel("Horizon de prévision")
    plt.ylabel("Quantité prévisionnelle")

    plt.grid(True, alpha=0.3)

    for _, row in daily.iterrows():
        plt.annotate(
            f"{row['forecast_quantity']:.1f}",
            (
                row["forecast_day"],
                row["forecast_quantity"],
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    plt.tight_layout()

    output = OUTPUT_DIR / "forecast_quantity_by_day.png"

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[OK] Graphique quantité : {output}"
    )


# ============================================================
# GRAPHIQUE 2 — CA PAR JOUR
# ============================================================

def plot_revenue_by_day(df: pd.DataFrame) -> None:
    """Crée le graphique du CA prévisionnel quotidien."""

    daily = (
        df.groupby(
            ["date", "forecast_day"],
            as_index=False
        )["forecast_revenue"]
        .sum()
        .sort_values("date")
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        daily["forecast_day"],
        daily["forecast_revenue"],
        marker="o",
        linewidth=2,
    )

    plt.title(
        "CA prévisionnel — J+1 à J+7"
    )

    plt.xlabel("Horizon de prévision")
    plt.ylabel("CA prévisionnel (AR)")

    plt.grid(True, alpha=0.3)

    for _, row in daily.iterrows():
        plt.annotate(
            f"{row['forecast_revenue']:,.0f}",
            (
                row["forecast_day"],
                row["forecast_revenue"],
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    plt.tight_layout()

    output = OUTPUT_DIR / "forecast_revenue_by_day.png"

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[OK] Graphique CA : {output}"
    )


# ============================================================
# GRAPHIQUE 3 — TOP PRODUITS QUANTITÉ
# ============================================================

def plot_top_products_quantity(
    df: pd.DataFrame,
) -> None:
    """Crée le Top 10 produits par quantité."""

    products = (
        df.groupby(
            "product_name",
            as_index=False
        )["forecast_quantity"]
        .sum()
        .sort_values(
            "forecast_quantity",
            ascending=True,
        )
        .tail(10)
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        products["product_name"],
        products["forecast_quantity"],
    )

    plt.title(
        "Top 10 produits — Quantité prévisionnelle"
    )

    plt.xlabel("Quantité prévisionnelle")
    plt.ylabel("Produit")

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "forecast_top_products_quantity.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[OK] Top produits quantité : {output}"
    )


# ============================================================
# GRAPHIQUE 4 — TOP PRODUITS CA
# ============================================================

def plot_top_products_revenue(
    df: pd.DataFrame,
) -> None:
    """Crée le Top 10 produits par CA."""

    products = (
        df.groupby(
            "product_name",
            as_index=False
        )["forecast_revenue"]
        .sum()
        .sort_values(
            "forecast_revenue",
            ascending=True,
        )
        .tail(10)
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        products["product_name"],
        products["forecast_revenue"],
    )

    plt.title(
        "Top 10 produits — CA prévisionnel"
    )

    plt.xlabel("CA prévisionnel (AR)")
    plt.ylabel("Produit")

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "forecast_top_products_revenue.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[OK] Top produits CA : {output}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_forecast(df: pd.DataFrame) -> None:
    """Valide les données avant visualisation."""

    if len(df) != 98:
        raise ValueError(
            f"Nombre de lignes incorrect : {len(df)}"
        )

    if df["product_id"].nunique() != 14:
        raise ValueError(
            "Le nombre de produits doit être 14."
        )

    if df["date"].nunique() != 7:
        raise ValueError(
            "Le nombre de jours doit être 7."
        )

    if df["forecast_quantity"].isna().any():
        raise ValueError(
            "Quantités prévisionnelles NULL."
        )

    if df["forecast_revenue"].isna().any():
        raise ValueError(
            "CA prévisionnel NULL."
        )

    if (df["forecast_quantity"] < 0).any():
        raise ValueError(
            "Quantité prévisionnelle négative."
        )

    if df.duplicated(
        ["date", "product_id"]
    ).any():
        raise ValueError(
            "Doublons date + product_id."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Point d'entrée."""

    print("=" * 70)
    print(
        "J7.6 — VISUALISATION DU FORECAST"
    )
    print("=" * 70)

    df = load_forecast()

    print(
        f"[OK] Forecast chargé : {len(df)} lignes"
    )

    validate_forecast(df)

    print(
        "[PASS] Données validées"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_quantity_by_day(df)

    plot_revenue_by_day(df)

    plot_top_products_quantity(df)

    plot_top_products_revenue(df)

    print("-" * 70)
    print(
        "[PASS] 4 visualisations générées"
    )
    print(
        f"[OK] Dossier : {OUTPUT_DIR}"
    )
    print("=" * 70)
    print(
        "J7.6 — VISUALISATION : TERMINÉE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()