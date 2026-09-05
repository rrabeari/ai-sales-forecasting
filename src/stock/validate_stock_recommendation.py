"""
J8.5 — Validation métier finale des recommandations de stock.

Vérifie :
- cohérence des données
- calcul du stock de sécurité
- calcul du stock cible
- calcul du réapprovisionnement
- cohérence des statuts
- cohérence des valeurs financières
- cohérence du nombre de produits
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

ANALYSIS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "stock"
    / "stock_analysis.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "stock"
    / "stock_validation.csv"
)

SAFETY_STOCK_RATE = 0.20


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

def validate_business_rules(df: pd.DataFrame) -> None:
    """Validate all business rules."""

    # Nombre de produits
    if len(df) != 14:
        raise ValueError(
            f"Nombre de produits incorrect : {len(df)} au lieu de 14."
        )

    # Un produit = une ligne
    if df["product_id"].duplicated().any():
        raise ValueError(
            "Doublons détectés sur product_id."
        )

    # NULL
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(
            "Des valeurs NULL sont présentes."
        )

    # Valeurs négatives
    numeric_columns = [
        "unit_price",
        "current_stock",
        "forecast_7d",
        "avg_daily_forecast",
        "safety_stock",
        "stock_target",
        "reorder_quantity",
    ]

    for column in numeric_columns:
        if (df[column] < 0).any():
            raise ValueError(
                f"Valeur négative détectée dans {column}."
            )

    # Prix strictement positif
    if (df["unit_price"] <= 0).any():
        raise ValueError(
            "Prix unitaire invalide."
        )

    # Stock actuel entier
    if not (
        (df["current_stock"] % 1 == 0)
        .all()
    ):
        raise ValueError(
            "current_stock doit être entier."
        )

    # Reorder entier
    if not (
        (df["reorder_quantity"] % 1 == 0)
        .all()
    ):
        raise ValueError(
            "reorder_quantity doit être entier."
        )

    # --------------------------------------------------------------
    # Règle 1 : moyenne quotidienne
    # --------------------------------------------------------------

    expected_avg_daily = (
        df["forecast_7d"] / 7
    ).round(2)

    if not (
        df["avg_daily_forecast"].round(2)
        .equals(expected_avg_daily)
    ):
        raise ValueError(
            "Erreur dans avg_daily_forecast."
        )

    # --------------------------------------------------------------
    # Règle 2 : stock de sécurité = 20 %
    # --------------------------------------------------------------

    expected_safety = (
        df["forecast_7d"]
        * SAFETY_STOCK_RATE
    ).round(2)

    if not (
        df["safety_stock"].round(2)
        .equals(expected_safety)
    ):
        raise ValueError(
            "Erreur dans le calcul du stock de sécurité."
        )

    # --------------------------------------------------------------
    # Règle 3 : stock cible
    # --------------------------------------------------------------

    expected_target = (
        df["forecast_7d"] + df["safety_stock"]
    )

    actual_target = df["stock_target"]

    if not (
        (actual_target - expected_target).abs() <= 0.011
    ).all():
        raise ValueError(
            "Erreur dans le calcul du stock cible."
        )

    # --------------------------------------------------------------
    # Règle 4 : quantité de réapprovisionnement
    # --------------------------------------------------------------

    expected_reorder = (
        df["stock_target"]
        - df["current_stock"]
    ).clip(lower=0).round(0).astype(int)

    actual_reorder = (
        df["reorder_quantity"]
        .round(0)
        .astype(int)
    )

    if not actual_reorder.equals(expected_reorder):
        raise ValueError(
            "Erreur dans le calcul de reorder_quantity."
        )

    # --------------------------------------------------------------
    # Règle 5 : statut
    # --------------------------------------------------------------

    expected_status = df["reorder_quantity"].apply(
        lambda x: "REORDER" if x > 0 else "OK"
    )

    if not df["status"].equals(expected_status):
        raise ValueError(
            "Erreur dans les statuts OK / REORDER."
        )

    # --------------------------------------------------------------
    # Règle 6 : demande totale
    # --------------------------------------------------------------

    total_forecast = round(
        df["forecast_7d"].sum(),
        2,
    )

    if total_forecast != 577.39:
        raise ValueError(
            f"Forecast total inattendu : {total_forecast}."
        )

    # --------------------------------------------------------------
    # Règle 7 : quantité totale à commander
    # --------------------------------------------------------------

    total_reorder = int(
        df["reorder_quantity"].sum()
    )

    if total_reorder != 205:
        raise ValueError(
            f"Quantité totale inattendue : {total_reorder}."
        )

    # --------------------------------------------------------------
    # Règle 8 : valeur du réapprovisionnement
    # --------------------------------------------------------------

    expected_value = (
        df["reorder_quantity"]
        * df["unit_price"]
    ).round(2)

    total_value = round(
        expected_value.sum(),
        2,
    )

    if total_value != 555800.00:
        raise ValueError(
            f"Valeur totale inattendue : {total_value} AR."
        )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    """Run final J8.5 validation."""

    print("=" * 70)
    print("J8.5 — VALIDATION MÉTIER FINALE DU STOCK")
    print("=" * 70)

    # Vérification fichier principal
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"[OK] Recommandations chargées : "
        f"{len(df)} produits"
    )

    # Vérification colonnes
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing)}"
        )

    print("[PASS] Colonnes requises présentes")

    # Validation métier
    validate_business_rules(df)

    print("[PASS] Nombre de produits : 14")
    print("[PASS] Aucun doublon product_id")
    print("[PASS] Aucun NULL")
    print("[PASS] Prix unitaires valides")
    print("[PASS] Stocks actuels valides")
    print("[PASS] Forecast 7 jours valide")
    print("[PASS] Moyenne quotidienne valide")
    print("[PASS] Stock de sécurité = 20 %")
    print("[PASS] Stock cible valide")
    print("[PASS] Reorder quantity valide")
    print("[PASS] Statuts OK / REORDER valides")
    print("[PASS] Forecast total = 577.39")
    print("[PASS] Quantité totale à commander = 205")
    print("[PASS] Valeur totale = 555800.00 AR")

    # --------------------------------------------------------------
    # Résumé de validation
    # --------------------------------------------------------------

    validation = pd.DataFrame(
        [
            {
                "check": "products",
                "expected": 14,
                "actual": len(df),
                "status": "PASS",
            },
            {
                "check": "forecast_7d",
                "expected": 577.39,
                "actual": round(
                    df["forecast_7d"].sum(),
                    2,
                ),
                "status": "PASS",
            },
            {
                "check": "reorder_quantity",
                "expected": 205,
                "actual": int(
                    df["reorder_quantity"].sum()
                ),
                "status": "PASS",
            },
            {
                "check": "reorder_value",
                "expected": 555800.00,
                "actual": round(
                    (
                        df["reorder_quantity"]
                        * df["unit_price"]
                    ).sum(),
                    2,
                ),
                "status": "PASS",
            },
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("-" * 70)
    print(
        f"[OK] Rapport de validation sauvegardé : "
        f"{OUTPUT_FILE}"
    )

    print("=" * 70)
    print("J8.5 — VALIDATION STOCK : OK")
    print("=" * 70)


if __name__ == "__main__":
    main()