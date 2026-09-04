"""
J6.4 — Entraînement des modèles de Machine Learning.

Objectifs :
- Charger Train et Validation
- Encoder product_id comme variable catégorielle
- Entraîner plusieurs modèles
- Produire les prédictions sur Validation
- Sauvegarder les modèles et les prédictions

IMPORTANT :
Le dataset Test n'est PAS utilisé à cette étape.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = Path(
    "data/processed/ml_split/train.csv"
)

VALIDATION_FILE = Path(
    "data/processed/ml_split/validation.csv"
)

MODEL_DIR = Path(
    "models"
)

PREDICTION_DIR = Path(
    "data/processed/ml_ready"
)


# ============================================================
# VARIABLES
# ============================================================

TARGET = "quantity"

CATEGORICAL_FEATURES = [
    "product_id",
]

NUMERICAL_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
]

FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
)


# ============================================================
# MODÈLES
# ============================================================

MODELS = {
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42,
    ),

    "hist_gradient_boosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    ),
}


# ============================================================
# CHARGEMENT
# ============================================================

def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """Charge un dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = set(
        FEATURES + [TARGET]
    )

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
# VALIDATION
# ============================================================

def validate_dataset(
    name: str,
    df: pd.DataFrame,
) -> None:
    """Valide un dataset."""

    if df.empty:
        raise ValueError(
            f"{name} est vide."
        )

    if df[FEATURES].isna().any().any():
        raise ValueError(
            f"Valeurs NULL détectées dans "
            f"les features de {name}."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            f"Valeurs NULL détectées dans "
            f"la target de {name}."
        )

    if (df[TARGET] < 0).any():
        raise ValueError(
            f"Valeurs négatives détectées "
            f"dans la target de {name}."
        )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor() -> ColumnTransformer:
    """Crée le préprocesseur."""

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )


# ============================================================
# PIPELINE
# ============================================================

def create_pipeline(model) -> Pipeline:
    """Crée un pipeline preprocessing + modèle."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Entraîne les modèles."""

    print("=" * 60)
    print("J6.4 — ENTRAÎNEMENT DES MODÈLES")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. CHARGEMENT
    # --------------------------------------------------------

    print("\n=== 1. CHARGEMENT ===")

    train = load_dataset(
        TRAIN_FILE
    )

    validation = load_dataset(
        VALIDATION_FILE
    )

    validate_dataset(
        "Train",
        train
    )

    validate_dataset(
        "Validation",
        validation
    )

    print(
        f"[OK] Train : {len(train):,} lignes"
    )

    print(
        f"[OK] Validation : "
        f"{len(validation):,} lignes"
    )

    print(
        f"[OK] Features : "
        f"{len(FEATURES)}"
    )

    print(
        f"[OK] Target : {TARGET}"
    )

    # --------------------------------------------------------
    # 2. X / y
    # --------------------------------------------------------

    print("\n=== 2. PRÉPARATION X / y ===")

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    print(
        f"[OK] X_train : {X_train.shape}"
    )

    print(
        f"[OK] y_train : {y_train.shape}"
    )

    print(
        f"[OK] X_validation : "
        f"{X_validation.shape}"
    )

    print(
        f"[OK] y_validation : "
        f"{y_validation.shape}"
    )

    # --------------------------------------------------------
    # 3. DOSSIERS
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 4. ENTRAÎNEMENT
    # --------------------------------------------------------

    print("\n=== 3. ENTRAÎNEMENT ===")

    results = []

    for model_name, model in MODELS.items():

        print(
            f"\n--- {model_name} ---"
        )

        pipeline = create_pipeline(
            model
        )

        print(
            "[INFO] Entraînement en cours..."
        )

        pipeline.fit(
            X_train,
            y_train
        )

        print(
            "[OK] Modèle entraîné"
        )

        # ----------------------------------------------------
        # PRÉDICTIONS VALIDATION
        # ----------------------------------------------------

        predictions = pipeline.predict(
            X_validation
        )

        predictions = np.maximum(
            predictions,
            0
        )

        prediction_file = (
            PREDICTION_DIR
            / f"{model_name}_validation.csv"
        )

        prediction_df = validation[
            [
                "date",
                "product_id",
                TARGET,
            ]
        ].copy()

        prediction_df[
            "prediction"
        ] = predictions

        prediction_df[
            "error"
        ] = (
            prediction_df[TARGET]
            - prediction_df["prediction"]
        )

        prediction_df.to_csv(
            prediction_file,
            index=False
        )

        # ----------------------------------------------------
        # SAUVEGARDE MODÈLE
        # ----------------------------------------------------

        model_file = (
            MODEL_DIR
            / f"{model_name}.joblib"
        )

        joblib.dump(
            pipeline,
            model_file
        )

        print(
            f"[OK] Prédictions : "
            f"{prediction_file}"
        )

        print(
            f"[OK] Modèle : "
            f"{model_file}"
        )

        # ----------------------------------------------------
        # RÉSULTATS
        # ----------------------------------------------------

        results.append(
            {
                "model": model_name,
                "predictions_file": str(
                    prediction_file
                ),
                "model_file": str(
                    model_file
                ),
                "mean_prediction": float(
                    np.mean(predictions)
                ),
                "min_prediction": float(
                    np.min(predictions)
                ),
                "max_prediction": float(
                    np.max(predictions)
                ),
            }
        )

    # --------------------------------------------------------
    # 5. RÉSUMÉ
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_file = (
        PREDICTION_DIR
        / "training_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    print("\n=== 4. RÉSUMÉ ===")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\n[OK] Résultats : "
        f"{results_file}"
    )

    print(
        "\n[PASS] Les modèles ont été "
        "entraînés sur Train uniquement."
    )

    print(
        "[PASS] Validation utilisée uniquement "
        "pour générer les prédictions."
    )

    print(
        "[PASS] Test NON utilisé."
    )

    print("\n" + "=" * 60)
    print("J6.4 — ENTRAÎNEMENT : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()