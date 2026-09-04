"""
J6.2 — Préparation des données pour le Machine Learning.

Objectifs :
- Charger les splits Train / Validation / Test
- Séparer les features X de la target y
- Identifier les variables numériques et catégorielles
- Vérifier l'intégrité des données
- Sauvegarder les jeux préparés
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/processed/ml_split")

TRAIN_FILE = INPUT_DIR / "train.csv"
VALIDATION_FILE = INPUT_DIR / "validation.csv"
TEST_FILE = INPUT_DIR / "test.csv"

OUTPUT_DIR = Path("data/processed/ml_ready")

TRAIN_X_FILE = OUTPUT_DIR / "X_train.csv"
TRAIN_Y_FILE = OUTPUT_DIR / "y_train.csv"

VALIDATION_X_FILE = OUTPUT_DIR / "X_validation.csv"
VALIDATION_Y_FILE = OUTPUT_DIR / "y_validation.csv"

TEST_X_FILE = OUTPUT_DIR / "X_test.csv"
TEST_Y_FILE = OUTPUT_DIR / "y_test.csv"


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


# ============================================================
# CHARGEMENT
# ============================================================

def load_split(file_path: Path) -> pd.DataFrame:
    """Charge un dataset Train, Validation ou Test."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            f"Dates invalides détectées dans {file_path}"
        )

    return df


# ============================================================
# VALIDATION DES COLONNES
# ============================================================

def validate_columns(df: pd.DataFrame) -> None:
    """Vérifie que toutes les colonnes nécessaires existent."""

    required_features = (
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
    )

    required_columns = set(
        required_features + [TARGET]
    )

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : "
            f"{sorted(missing_columns)}"
        )


# ============================================================
# PRÉPARATION X / y
# ============================================================

def prepare_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les features X de la target y."""

    validate_columns(df)

    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
    )

    X = df[feature_columns].copy()

    y = df[TARGET].copy()

    return X, y


# ============================================================
# VALIDATION DES DONNÉES ML
# ============================================================

def validate_ml_data(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Valide X et y."""

    if X.empty:
        raise ValueError(
            f"{name} : X est vide."
        )

    if y.empty:
        raise ValueError(
            f"{name} : y est vide."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{name} : X et y n'ont pas "
            f"le même nombre de lignes."
        )

    # Vérification des valeurs manquantes
    if X.isna().any().any():
        missing_columns = (
            X.columns[
                X.isna().any()
            ].tolist()
        )

        raise ValueError(
            f"{name} : valeurs NULL dans "
            f"les features : "
            f"{missing_columns}"
        )

    if y.isna().any():
        raise ValueError(
            f"{name} : target NULL détectée."
        )

    # Target non négative
    if (y < 0).any():
        raise ValueError(
            f"{name} : target négative détectée."
        )

    # Vérification des colonnes catégorielles
    for column in CATEGORICAL_FEATURES:

        if X[column].isna().any():
            raise ValueError(
                f"{name} : NULL dans "
                f"{column}."
            )

    # Vérification des features numériques
    for column in NUMERICAL_FEATURES:

        if not pd.api.types.is_numeric_dtype(
            X[column]
        ):
            raise ValueError(
                f"{name} : {column} "
                f"n'est pas numérique."
            )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_xy(
    X: pd.DataFrame,
    y: pd.Series,
    x_file: Path,
    y_file: Path,
) -> None:
    """Sauvegarde X et y."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X.to_csv(
        x_file,
        index=False
    )

    y.to_csv(
        y_file,
        index=False,
        header=[TARGET]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Exécute la préparation ML."""

    print("=" * 60)
    print("J6.2 — PRÉPARATION X / y")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. CHARGEMENT
    # --------------------------------------------------------

    print("\n=== 1. CHARGEMENT ===")

    train = load_split(TRAIN_FILE)
    validation = load_split(VALIDATION_FILE)
    test = load_split(TEST_FILE)

    print(
        f"[OK] Train      : {len(train):,} lignes"
    )

    print(
        f"[OK] Validation : {len(validation):,} lignes"
    )

    print(
        f"[OK] Test       : {len(test):,} lignes"
    )

    # --------------------------------------------------------
    # 2. PRÉPARATION
    # --------------------------------------------------------

    print("\n=== 2. PRÉPARATION X / y ===")

    X_train, y_train = prepare_xy(train)
    X_validation, y_validation = prepare_xy(validation)
    X_test, y_test = prepare_xy(test)

    print(
        f"[OK] X_train      : {X_train.shape}"
    )

    print(
        f"[OK] y_train      : {y_train.shape}"
    )

    print(
        f"[OK] X_validation : {X_validation.shape}"
    )

    print(
        f"[OK] y_validation : {y_validation.shape}"
    )

    print(
        f"[OK] X_test       : {X_test.shape}"
    )

    print(
        f"[OK] y_test       : {y_test.shape}"
    )

    # --------------------------------------------------------
    # 3. VALIDATION
    # --------------------------------------------------------

    print("\n=== 3. VALIDATION ===")

    validate_ml_data(
        "Train",
        X_train,
        y_train
    )

    validate_ml_data(
        "Validation",
        X_validation,
        y_validation
    )

    validate_ml_data(
        "Test",
        X_test,
        y_test
    )

    print(
        "[PASS] X et y ont le même nombre de lignes"
    )

    print(
        "[PASS] Aucune valeur NULL"
    )

    print(
        "[PASS] Target quantity valide"
    )

    print(
        "[PASS] Features numériques valides"
    )

    print(
        "[PASS] Product ID catégoriel valide"
    )

    # --------------------------------------------------------
    # 4. FEATURES
    # --------------------------------------------------------

    print("\n=== 4. FEATURES ===")

    print(
        f"[OK] Features catégorielles : "
        f"{CATEGORICAL_FEATURES}"
    )

    print(
        f"[OK] Features numériques : "
        f"{NUMERICAL_FEATURES}"
    )

    print(
        f"[OK] Nombre total de features : "
        f"{len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES)}"
    )

    print(
        f"[OK] Target : {TARGET}"
    )

    # --------------------------------------------------------
    # 5. SAUVEGARDE
    # --------------------------------------------------------

    print("\n=== 5. SAUVEGARDE ===")

    save_xy(
        X_train,
        y_train,
        TRAIN_X_FILE,
        TRAIN_Y_FILE
    )

    save_xy(
        X_validation,
        y_validation,
        VALIDATION_X_FILE,
        VALIDATION_Y_FILE
    )

    save_xy(
        X_test,
        y_test,
        TEST_X_FILE,
        TEST_Y_FILE
    )

    print(
        f"[OK] {TRAIN_X_FILE}"
    )

    print(
        f"[OK] {TRAIN_Y_FILE}"
    )

    print(
        f"[OK] {VALIDATION_X_FILE}"
    )

    print(
        f"[OK] {VALIDATION_Y_FILE}"
    )

    print(
        f"[OK] {TEST_X_FILE}"
    )

    print(
        f"[OK] {TEST_Y_FILE}"
    )

    # --------------------------------------------------------
    # 6. RÉSUMÉ
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("J6.2 — RÉSUMÉ")
    print("=" * 60)

    print(
        f"X_train      : {X_train.shape}"
    )

    print(
        f"X_validation : {X_validation.shape}"
    )

    print(
        f"X_test       : {X_test.shape}"
    )

    print(
        f"\ny_train      : {y_train.shape}"
    )

    print(
        f"y_validation : {y_validation.shape}"
    )

    print(
        f"y_test       : {y_test.shape}"
    )

    print(
        f"\nTarget : {TARGET}"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "J6.2 — PRÉPARATION X / y : OK"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()