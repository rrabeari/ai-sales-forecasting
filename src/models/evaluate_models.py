"""
J6.5 — Évaluation des modèles Machine Learning.

Compare les modèles entraînés en J6.4 avec la baseline.

Métriques :
- MAE
- RMSE
- R²

IMPORTANT :
Le dataset Test n'est PAS utilisé.
L'évaluation porte uniquement sur Validation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_FILE = Path(
    "data/processed/ml_split/validation.csv"
)

PREDICTION_DIR = Path(
    "data/processed/ml_ready"
)

OUTPUT_FILE = (
    PREDICTION_DIR
    / "model_evaluation_validation.csv"
)


TARGET = "quantity"


MODEL_FILES = {
    "baseline_lag_7": "baseline_validation.csv",
    "random_forest": "random_forest_validation.csv",
    "gradient_boosting": "gradient_boosting_validation.csv",
    "hist_gradient_boosting": "hist_gradient_boosting_validation.csv",
}


# ============================================================
# CHARGEMENT VALIDATION
# ============================================================

def load_validation() -> pd.DataFrame:
    """Charge le dataset de validation."""

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {VALIDATION_FILE}"
        )

    df = pd.read_csv(
        VALIDATION_FILE
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Colonne target absente : {TARGET}"
        )

    return df


# ============================================================
# CHARGEMENT DES PRÉDICTIONS
# ============================================================

def load_predictions(
    file_path: Path,
) -> pd.DataFrame:
    """Charge un fichier de prédictions."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier de prédictions introuvable : "
            f"{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    required_columns = {
        "date",
        "product_id",
        TARGET,
        "prediction",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans "
            f"{file_path}: "
            f"{sorted(missing_columns)}"
        )

    return df


# ============================================================
# VALIDATION DES PRÉDICTIONS
# ============================================================

def validate_predictions(
    model_name: str,
    df: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Vérifie l'intégrité des prédictions."""

    if df.empty:
        raise ValueError(
            f"{model_name} : prédictions vides."
        )

    if len(df) != len(validation):
        raise ValueError(
            f"{model_name} : nombre de lignes "
            f"incompatible avec Validation."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            f"{model_name} : target NULL."
        )

    if df["prediction"].isna().any():
        raise ValueError(
            f"{model_name} : prédictions NULL."
        )

    if not np.isfinite(
        df["prediction"]
    ).all():
        raise ValueError(
            f"{model_name} : prédictions "
            f"non finies détectées."
        )

    if (df["prediction"] < 0).any():
        raise ValueError(
            f"{model_name} : prédictions "
            f"négatives détectées."
        )

    # Vérification de la structure temporelle
    expected_keys = validation[
        ["date", "product_id"]
    ].astype(str)

    prediction_keys = df[
        ["date", "product_id"]
    ].astype(str)

    expected_keys = set(
        zip(
            expected_keys["date"],
            expected_keys["product_id"],
        )
    )

    prediction_keys = set(
        zip(
            prediction_keys["date"],
            prediction_keys["product_id"],
        )
    )

    if expected_keys != prediction_keys:
        raise ValueError(
            f"{model_name} : les clés "
            f"date + product_id ne correspondent pas."
        )


# ============================================================
# MÉTRIQUES
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """Calcule MAE, RMSE et R²."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Évalue les modèles."""

    print("=" * 60)
    print("J6.5 — ÉVALUATION DES MODÈLES")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. CHARGEMENT
    # --------------------------------------------------------

    print("\n=== 1. CHARGEMENT ===")

    validation = load_validation()

    print(
        f"[OK] Validation : "
        f"{len(validation):,} lignes"
    )

    print(
        f"[OK] Target : {TARGET}"
    )

    # --------------------------------------------------------
    # 2. ÉVALUATION
    # --------------------------------------------------------

    print("\n=== 2. ÉVALUATION ===")

    results = []

    for model_name, filename in MODEL_FILES.items():

        print(
            f"\n--- {model_name} ---"
        )

        prediction_file = (
            PREDICTION_DIR
            / filename
        )

        predictions = load_predictions(
            prediction_file
        )

        validate_predictions(
            model_name,
            predictions,
            validation,
        )

        y_true = predictions[TARGET]

        y_pred = predictions["prediction"]

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        print(
            f"[OK] MAE  : {metrics['MAE']:.4f}"
        )

        print(
            f"[OK] RMSE : {metrics['RMSE']:.4f}"
        )

        print(
            f"[OK] R²   : {metrics['R2']:.4f}"
        )

        results.append(
            {
                "model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    # --------------------------------------------------------
    # 3. COMPARAISON
    # --------------------------------------------------------

    print("\n=== 3. COMPARAISON ===")

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="MAE",
        ascending=True,
    ).reset_index(
        drop=True
    )

    results_df.insert(
        0,
        "rank",
        range(
            1,
            len(results_df) + 1
        ),
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # 4. MEILLEUR MODÈLE
    # --------------------------------------------------------

    best_model = results_df.iloc[0]

    print("\n=== 4. MEILLEUR MODÈLE SELON MAE ===")

    print(
        f"[INFO] Modèle : "
        f"{best_model['model']}"
    )

    print(
        f"[INFO] MAE : "
        f"{best_model['MAE']:.4f}"
    )

    print(
        f"[INFO] RMSE : "
        f"{best_model['RMSE']:.4f}"
    )

    print(
        f"[INFO] R² : "
        f"{best_model['R2']:.4f}"
    )

    # --------------------------------------------------------
    # 5. COMPARAISON AVEC BASELINE
    # --------------------------------------------------------

    print(
        "\n=== 5. COMPARAISON AVEC BASELINE ==="
    )

    baseline = results_df[
        results_df["model"]
        == "baseline_lag_7"
    ].iloc[0]

    ml_results = results_df[
        results_df["model"]
        != "baseline_lag_7"
    ]

    best_ml = ml_results.iloc[0]

    mae_improvement = (
        (
            baseline["MAE"]
            - best_ml["MAE"]
        )
        / baseline["MAE"]
        * 100
    )

    rmse_improvement = (
        (
            baseline["RMSE"]
            - best_ml["RMSE"]
        )
        / baseline["RMSE"]
        * 100
    )

    print(
        f"[INFO] Baseline MAE : "
        f"{baseline['MAE']:.4f}"
    )

    print(
        f"[INFO] Meilleur ML MAE : "
        f"{best_ml['MAE']:.4f}"
    )

    print(
        f"[INFO] Amélioration MAE : "
        f"{mae_improvement:.2f}%"
    )

    print(
        f"[INFO] Amélioration RMSE : "
        f"{rmse_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # 6. SAUVEGARDE
    # --------------------------------------------------------

    print("\n=== 6. SAUVEGARDE ===")

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"[OK] {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # 7. VALIDATION
    # --------------------------------------------------------

    print("\n=== 7. VALIDATION ===")

    print(
        "[PASS] 4 méthodes évaluées"
    )

    print(
        "[PASS] MAE calculé"
    )

    print(
        "[PASS] RMSE calculé"
    )

    print(
        "[PASS] R² calculé"
    )

    print(
        "[PASS] Prédictions alignées avec Validation"
    )

    print(
        "[PASS] Aucune prédiction négative"
    )

    print(
        "[PASS] Dataset Test NON utilisé"
    )

    print("\n" + "=" * 60)
    print("J6.5 — ÉVALUATION : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()