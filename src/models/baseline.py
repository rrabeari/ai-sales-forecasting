"""
J6.3 — Baseline de prévision.

Baseline :
    prédiction = lag_7

La demande prévue pour un produit est donc
égale à sa demande observée 7 jours auparavant.

Le dataset Test n'est volontairement pas utilisé ici.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_FILE = Path(
    "data/processed/ml_split/validation.csv"
)

OUTPUT_DIR = Path(
    "data/processed/ml_ready"
)

PREDICTIONS_FILE = OUTPUT_DIR / "baseline_validation.csv"

TARGET = "quantity"
BASELINE_FEATURE = "lag_7"


# ============================================================
# CHARGEMENT
# ============================================================

def load_validation_dataset() -> pd.DataFrame:
    """Charge le dataset de validation."""

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {VALIDATION_FILE}"
        )

    df = pd.read_csv(VALIDATION_FILE)

    required_columns = {
        TARGET,
        BASELINE_FEATURE,
        "date",
        "product_id",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            "Dates invalides détectées."
        )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_baseline_data(
    df: pd.DataFrame,
) -> None:
    """Valide les données nécessaires à la baseline."""

    if df.empty:
        raise ValueError(
            "Le dataset de validation est vide."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Valeurs NULL dans la target."
        )

    if df[BASELINE_FEATURE].isna().any():
        raise ValueError(
            "Valeurs NULL dans lag_7."
        )

    if (df[TARGET] < 0).any():
        raise ValueError(
            "Valeurs négatives dans la target."
        )

    if (df[BASELINE_FEATURE] < 0).any():
        raise ValueError(
            "Valeurs négatives dans lag_7."
        )


# ============================================================
# CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """Calcule les métriques de performance."""

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# SAUVEGARDE
# ============================================================

def save_predictions(
    df: pd.DataFrame,
) -> None:
    """Sauvegarde les prédictions de la baseline."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Exécute la baseline."""

    print("=" * 60)
    print("J6.3 — BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. CHARGEMENT
    # --------------------------------------------------------

    print("\n=== 1. CHARGEMENT ===")

    df = load_validation_dataset()

    print(
        f"[OK] Validation : {len(df):,} lignes"
    )

    print(
        f"[OK] Période : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print(
        f"[OK] Produits : "
        f"{df['product_id'].nunique()}"
    )

    # --------------------------------------------------------
    # 2. VALIDATION
    # --------------------------------------------------------

    print("\n=== 2. VALIDATION ===")

    validate_baseline_data(df)

    print(
        "[PASS] Target quantity valide"
    )

    print(
        "[PASS] lag_7 valide"
    )

    print(
        "[PASS] Aucune valeur NULL"
    )

    # --------------------------------------------------------
    # 3. PRÉDICTION
    # --------------------------------------------------------

    print("\n=== 3. BASELINE ===")

    y_true = df[TARGET]

    y_pred = df[BASELINE_FEATURE]

    print(
        f"[OK] Méthode : "
        f"prediction = {BASELINE_FEATURE}"
    )

    # --------------------------------------------------------
    # 4. MÉTRIQUES
    # --------------------------------------------------------

    print("\n=== 4. MÉTRIQUES ===")

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )

    # --------------------------------------------------------
    # 5. PRÉDICTIONS
    # --------------------------------------------------------

    print("\n=== 5. SAUVEGARDE ===")

    predictions = df[
        [
            "date",
            "product_id",
            TARGET,
        ]
    ].copy()

    predictions["prediction"] = y_pred

    predictions["error"] = (
        predictions[TARGET]
        - predictions["prediction"]
    )

    save_predictions(
        predictions
    )

    print(
        f"[OK] {PREDICTIONS_FILE}"
    )

    # --------------------------------------------------------
    # 6. RÉSUMÉ
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("J6.3 — RÉSUMÉ")
    print("=" * 60)

    print(
        f"Baseline : {BASELINE_FEATURE}"
    )

    print(
        f"MAE      : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE     : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²       : {metrics['R2']:.4f}"
    )

    print(
        "\nDataset utilisé : Validation"
    )

    print(
        "Dataset Test : NON UTILISÉ"
    )

    print("\n" + "=" * 60)
    print("J6.3 — BASELINE : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()