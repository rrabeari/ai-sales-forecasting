"""
J7.7 — Validation finale du pipeline Forecast.

Contrôles :
- fichier forecast principal
- 98 prévisions
- 14 produits
- 7 jours
- dates J+1 → J+7
- absence de NULL
- absence de valeurs négatives
- absence de doublons
- cohérence quantité / CA
- présence des fichiers d'analyse
- présence des visualisations
- présence du modèle final
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_FILE = Path(
    "data/processed/forecast/sales_forecast_j1_j7.csv"
)

DAILY_FILE = Path(
    "data/processed/forecast/forecast_daily_summary.csv"
)

PRODUCT_FILE = Path(
    "data/processed/forecast/forecast_product_summary.csv"
)

MODEL_FILE = Path(
    "models/final_model.joblib"
)

METADATA_FILE = Path(
    "models/final_model_metadata.json"
)

VISUALIZATION_DIR = Path(
    "data/processed/forecast/visualizations"
)

EXPECTED_VISUALIZATIONS = [
    "forecast_quantity_by_day.png",
    "forecast_revenue_by_day.png",
    "forecast_top_products_quantity.png",
    "forecast_top_products_revenue.png",
]

EXPECTED_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "category",
    "forecast_quantity",
    "unit_price",
    "forecast_revenue",
    "forecast_day",
]


# ============================================================
# UTILITAIRES
# ============================================================

def check(condition: bool, success: str, error: str) -> None:
    """Affiche PASS ou ERROR selon le résultat."""

    if not condition:
        raise ValueError(f"[FAIL] {error}")

    print(f"[PASS] {success}")


# ============================================================
# VALIDATION FICHIER PRINCIPAL
# ============================================================

def validate_forecast_file() -> pd.DataFrame:
    """Valide le fichier principal des prévisions."""

    check(
        FORECAST_FILE.exists(),
        "Fichier Forecast présent",
        f"Fichier absent : {FORECAST_FILE}",
    )

    df = pd.read_csv(FORECAST_FILE)

    print(
        f"[INFO] Forecast chargé : {len(df)} lignes"
    )

    check(
        list(df.columns) == EXPECTED_COLUMNS,
        "Colonnes Forecast correctes",
        "Structure des colonnes incorrecte",
    )

    check(
        len(df) == 98,
        "98 prévisions présentes",
        f"Nombre de lignes incorrect : {len(df)}",
    )

    check(
        df["product_id"].nunique() == 14,
        "14 produits présents",
        (
            "Nombre de produits incorrect : "
            f"{df['product_id'].nunique()}"
        ),
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    check(
        df["date"].notna().all(),
        "Toutes les dates sont valides",
        "Date invalide détectée",
    )

    check(
        df["date"].nunique() == 7,
        "7 dates de prévision présentes",
        (
            "Nombre de dates incorrect : "
            f"{df['date'].nunique()}"
        ),
    )

    expected_dates = pd.date_range(
        "2026-09-01",
        "2026-09-07",
        freq="D",
    )

    actual_dates = sorted(
        df["date"].unique()
    )

    check(
        actual_dates
        == list(expected_dates),
        "Dates J+1 → J+7 correctes",
        f"Dates incorrectes : {actual_dates}",
    )

    check(
        df["forecast_day"].tolist()
        == df["forecast_day"].tolist(),
        "Colonne forecast_day présente",
        "forecast_day invalide",
    )

    expected_forecast_days = {
        "J+1",
        "J+2",
        "J+3",
        "J+4",
        "J+5",
        "J+6",
        "J+7",
    }

    actual_forecast_days = set(
        df["forecast_day"].unique()
    )

    check(
        actual_forecast_days
        == expected_forecast_days,
        "Horizons J+1 → J+7 corrects",
        (
            "Horizons incorrects : "
            f"{actual_forecast_days}"
        ),
    )

    check(
        df.isna().sum().sum() == 0,
        "Aucune valeur NULL",
        "Des valeurs NULL sont présentes",
    )

    check(
        (df["forecast_quantity"] >= 0).all(),
        "Toutes les quantités sont non négatives",
        "Quantité prévisionnelle négative détectée",
    )

    check(
        (df["unit_price"] > 0).all(),
        "Tous les prix unitaires sont positifs",
        "Prix unitaire invalide détecté",
    )

    check(
        df.duplicated(
            ["date", "product_id"]
        ).sum() == 0,
        "Aucun doublon date + product_id",
        "Doublon détecté",
    )

    revenue_difference = (
        df["forecast_quantity"]
        * df["unit_price"]
        - df["forecast_revenue"]
    ).abs()

    check(
        (revenue_difference <= 0.01).all(),
        "Quantité × prix = CA prévisionnel",
        "Incohérence quantité / CA détectée",
    )

    return df


# ============================================================
# VALIDATION DES ANALYSES
# ============================================================

def validate_analysis_files(
    df: pd.DataFrame,
) -> None:
    """Valide les fichiers d'analyse J7.5."""

    check(
        DAILY_FILE.exists(),
        "Analyse quotidienne présente",
        f"Fichier absent : {DAILY_FILE}",
    )

    check(
        PRODUCT_FILE.exists(),
        "Analyse produits présente",
        f"Fichier absent : {PRODUCT_FILE}",
    )

    daily = pd.read_csv(DAILY_FILE)
    products = pd.read_csv(PRODUCT_FILE)

    check(
        len(daily) == 7,
        "Analyse quotidienne contient 7 lignes",
        f"Nombre de lignes daily : {len(daily)}",
    )

    check(
        len(products) == 14,
        "Analyse produits contient 14 lignes",
        f"Nombre de lignes products : {len(products)}",
    )

    forecast_total = df[
        "forecast_quantity"
    ].sum()

    daily_total = daily[
        "forecast_quantity"
    ].sum()

    product_total = products[
        "forecast_quantity"
    ].sum()

    check(
        abs(forecast_total - daily_total) <= 0.01,
        "Total quantité cohérent avec analyse quotidienne",
        "Total quantité daily incohérent",
    )

    check(
        abs(forecast_total - product_total) <= 0.01,
        "Total quantité cohérent avec analyse produits",
        "Total quantité produits incohérent",
    )

    forecast_revenue = df[
        "forecast_revenue"
    ].sum()

    daily_revenue = daily[
        "forecast_revenue"
    ].sum()

    product_revenue = products[
        "forecast_revenue"
    ].sum()

    check(
        abs(forecast_revenue - daily_revenue) <= 0.01,
        "Total CA cohérent avec analyse quotidienne",
        "Total CA daily incohérent",
    )

    check(
        abs(forecast_revenue - product_revenue) <= 0.01,
        "Total CA cohérent avec analyse produits",
        "Total CA produits incohérent",
    )


# ============================================================
# VALIDATION DU MODÈLE
# ============================================================

def validate_model_artifacts() -> None:
    """Vérifie les artefacts du modèle final."""

    check(
        MODEL_FILE.exists(),
        "Modèle final présent",
        f"Modèle absent : {MODEL_FILE}",
    )

    check(
        MODEL_FILE.stat().st_size > 0,
        "Modèle final non vide",
        "Le fichier modèle est vide",
    )

    check(
        METADATA_FILE.exists(),
        "Métadonnées du modèle présentes",
        f"Métadonnées absentes : {METADATA_FILE}",
    )

    check(
        METADATA_FILE.stat().st_size > 0,
        "Métadonnées non vides",
        "Le fichier metadata est vide",
    )


# ============================================================
# VALIDATION DES VISUALISATIONS
# ============================================================

def validate_visualizations() -> None:
    """Vérifie les quatre visualisations J7.6."""

    check(
        VISUALIZATION_DIR.exists(),
        "Dossier visualisations présent",
        (
            "Dossier absent : "
            f"{VISUALIZATION_DIR}"
        ),
    )

    for filename in EXPECTED_VISUALIZATIONS:
        filepath = VISUALIZATION_DIR / filename

        check(
            filepath.exists(),
            f"Visualisation présente : {filename}",
            f"Visualisation absente : {filename}",
        )

        check(
            filepath.stat().st_size > 0,
            f"Visualisation non vide : {filename}",
            f"Fichier vide : {filename}",
        )


# ============================================================
# RÉSUMÉ FINAL
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    """Affiche le résumé final du Forecast."""

    total_quantity = df[
        "forecast_quantity"
    ].sum()

    total_revenue = df[
        "forecast_revenue"
    ].sum()

    print()
    print("[FINAL SUMMARY]")
    print(
        f"Prévisions        : {len(df)}"
    )
    print(
        f"Produits          : "
        f"{df['product_id'].nunique()}"
    )
    print(
        f"Jours             : "
        f"{df['date'].nunique()}"
    )
    print(
        f"Quantité totale   : "
        f"{total_quantity:.2f}"
    )
    print(
        f"CA prévisionnel   : "
        f"{total_revenue:,.2f} AR"
    )
    print(
        f"Période           : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Point d'entrée de la validation finale."""

    print("=" * 70)
    print(
        "J7.7 — VALIDATION FINALE DU FORECAST"
    )
    print("=" * 70)

    df = validate_forecast_file()

    validate_analysis_files(df)

    validate_model_artifacts()

    validate_visualizations()

    print_summary(df)

    print()
    print("=" * 70)
    print(
        "J7.7 — VALIDATION FINALE : OK"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()