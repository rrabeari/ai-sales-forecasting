"""
J6.9 — Évaluation finale du modèle.

Le modèle final est évalué sur le dataset TEST,
qui n'a pas été utilisé pour l'entraînement
ni pour la sélection du modèle.

Modèle :
    Gradient Boosting

Target :
    quantity

Test :
    2026-08-01 -> 2026-08-31
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = Path(
    "models/final_model.joblib"
)

X_TEST_FILE = Path(
    "data/processed/ml_ready/X_test.csv"
)

Y_TEST_FILE = Path(
    "data/processed/ml_ready/y_test.csv"
)

TEST_SPLIT_FILE = Path(
    "data/processed/ml_split/test.csv"
)

OUTPUT_PREDICTIONS = Path(
    "data/processed/ml_ready/final_model_test_predictions.csv"
)

OUTPUT_METRICS = Path(
    "data/processed/ml_ready/final_model_test_evaluation.csv"
)

TARGET = "quantity"


# ============================================================
# CHARGEMENT
# ============================================================

def load_model():
    """Charge le modèle final."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "[PASS] Modèle final chargé"
    )

    return model


def load_test_data():
    """Charge X_test et y_test."""

    if not X_TEST_FILE.exists():
        raise FileNotFoundError(
            f"X_test introuvable : {X_TEST_FILE}"
        )

    if not Y_TEST_FILE.exists():
        raise FileNotFoundError(
            f"y_test introuvable : {Y_TEST_FILE}"
        )

    if not TEST_SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Test split introuvable : {TEST_SPLIT_FILE}"
        )

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    y_test = pd.read_csv(
        Y_TEST_FILE
    )

    test_split = pd.read_csv(
        TEST_SPLIT_FILE
    )

    return X_test, y_test, test_split


# ============================================================
# VALIDATION DU TEST
# ============================================================

def validate_test_data(
    X_test,
    y_test,
    test_split,
):
    """Valide la structure du dataset Test."""

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test et y_test n'ont pas "
            "le même nombre de lignes."
        )

    if len(X_test) != 434:
        raise ValueError(
            f"Nombre de lignes Test inattendu : "
            f"{len(X_test)}"
        )

    if len(test_split) != 434:
        raise ValueError(
            f"Nombre de lignes du split Test "
            f"inattendu : {len(test_split)}"
        )

    if "quantity" not in y_test.columns:
        raise ValueError(
            "La colonne target 'quantity' "
            "est absente de y_test."
        )

    if y_test[TARGET].isna().any():
        raise ValueError(
            "Valeurs NULL détectées dans y_test."
        )

    if (y_test[TARGET] < 0).any():
        raise ValueError(
            "Quantités négatives détectées "
            "dans y_test."
        )

    print(
        "[PASS] X_test / y_test alignés"
    )

    print(
        "[PASS] 434 lignes Test"
    )

    print(
        "[PASS] Target quantity valide"
    )


# ============================================================
# VÉRIFICATION DE LA PÉRIODE
# ============================================================

def validate_test_period(
    test_split,
):
    """Vérifie la période du dataset Test."""

    if "date" not in test_split.columns:
        raise ValueError(
            "La colonne date est absente du Test."
        )

    dates = pd.to_datetime(
        test_split["date"]
    )

    min_date = dates.min()
    max_date = dates.max()

    expected_min = pd.Timestamp(
        "2026-08-01"
    )

    expected_max = pd.Timestamp(
        "2026-08-31"
    )

    if min_date != expected_min:
        raise ValueError(
            f"Date minimale inattendue : "
            f"{min_date}"
        )

    if max_date != expected_max:
        raise ValueError(
            f"Date maximale inattendue : "
            f"{max_date}"
        )

    print(
        "[PASS] Période Test : "
        "2026-08-01 → 2026-08-31"
    )


# ============================================================
# PRÉDICTION
# ============================================================

def predict(
    model,
    X_test,
):
    """Produit les prédictions Test."""

    predictions = model.predict(
        X_test
    )

    predictions = pd.Series(
        predictions,
        name="prediction",
    )

    return predictions


# ============================================================
# MÉTRIQUES
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """Calcule les métriques finales."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

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
# BASELINE
# ============================================================

def calculate_baseline(
    test_split,
):
    """
    Calcule les performances de la baseline lag_7
    directement sur le Test.
    """

    if "lag_7" not in test_split.columns:
        raise ValueError(
            "La colonne lag_7 est absente du Test."
        )

    baseline_df = test_split[
        ["quantity", "lag_7"]
    ].dropna()

    y_true = baseline_df[
        "quantity"
    ]

    y_baseline = baseline_df[
        "lag_7"
    ]

    return calculate_metrics(
        y_true,
        y_baseline,
    )


# ============================================================
# SAUVEGARDE DES PRÉDICTIONS
# ============================================================

def save_predictions(
    test_split,
    predictions,
):
    """Sauvegarde les prédictions finales."""

    result = test_split.copy()

    result["prediction"] = predictions

    result["prediction_error"] = (
        result["quantity"]
        - result["prediction"]
    )

    result["absolute_error"] = (
        result["prediction_error"]
        .abs()
    )

    OUTPUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print(
        f"[OK] Prédictions sauvegardées : "
        f"{OUTPUT_PREDICTIONS}"
    )


# ============================================================
# SAUVEGARDE DES MÉTRIQUES
# ============================================================

def save_metrics(
    final_metrics,
    baseline_metrics,
):
    """Sauvegarde les métriques finales."""

    mae_gain = (
        (
            baseline_metrics["MAE"]
            - final_metrics["MAE"]
        )
        / baseline_metrics["MAE"]
        * 100
    )

    rmse_gain = (
        (
            baseline_metrics["RMSE"]
            - final_metrics["RMSE"]
        )
        / baseline_metrics["RMSE"]
        * 100
    )

    results = pd.DataFrame(
        [
            {
                "model": "final_gradient_boosting",
                "MAE": final_metrics["MAE"],
                "RMSE": final_metrics["RMSE"],
                "R2": final_metrics["R2"],
                "test_rows": 434,
                "test_period": "2026-08-01_to_2026-08-31",
            },
            {
                "model": "baseline_lag_7",
                "MAE": baseline_metrics["MAE"],
                "RMSE": baseline_metrics["RMSE"],
                "R2": baseline_metrics["R2"],
                "test_rows": 434,
                "test_period": "2026-08-01_to_2026-08-31",
            },
        ]
    )

    OUTPUT_METRICS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    return mae_gain, rmse_gain


# ============================================================
# MAIN
# ============================================================

def main():
    """Exécute J6.9."""

    print("=" * 60)
    print("J6.9 — ÉVALUATION FINALE SUR LE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. CHARGEMENT
    # --------------------------------------------------------

    print("\n=== 1. CHARGEMENT ===")

    model = load_model()

    X_test, y_test, test_split = (
        load_test_data()
    )

    print(
        f"[OK] X_test : {X_test.shape}"
    )

    print(
        f"[OK] y_test : {y_test.shape}"
    )

    # --------------------------------------------------------
    # 2. VALIDATION
    # --------------------------------------------------------

    print("\n=== 2. VALIDATION DU TEST ===")

    validate_test_data(
        X_test,
        y_test,
        test_split,
    )

    validate_test_period(
        test_split
    )

    # --------------------------------------------------------
    # 3. PRÉDICTION
    # --------------------------------------------------------

    print("\n=== 3. PRÉDICTION ===")

    predictions = predict(
        model,
        X_test,
    )

    print(
        f"[OK] {len(predictions)} prédictions générées"
    )

    print(
        f"[INFO] Prédiction moyenne : "
        f"{predictions.mean():.4f}"
    )

    print(
        f"[INFO] Prédiction min : "
        f"{predictions.min():.4f}"
    )

    print(
        f"[INFO] Prédiction max : "
        f"{predictions.max():.4f}"
    )

    # --------------------------------------------------------
    # 4. ÉVALUATION MODÈLE FINAL
    # --------------------------------------------------------

    print("\n=== 4. ÉVALUATION MODÈLE FINAL ===")

    y_true = y_test[
        TARGET
    ]

    final_metrics = calculate_metrics(
        y_true,
        predictions,
    )

    print(
        f"[RESULT] MAE  : "
        f"{final_metrics['MAE']:.4f}"
    )

    print(
        f"[RESULT] RMSE : "
        f"{final_metrics['RMSE']:.4f}"
    )

    print(
        f"[RESULT] R²   : "
        f"{final_metrics['R2']:.4f}"
    )

    # --------------------------------------------------------
    # 5. BASELINE
    # --------------------------------------------------------

    print("\n=== 5. BASELINE LAG_7 ===")

    baseline_metrics = calculate_baseline(
        test_split
    )

    print(
        f"[RESULT] Baseline MAE  : "
        f"{baseline_metrics['MAE']:.4f}"
    )

    print(
        f"[RESULT] Baseline RMSE : "
        f"{baseline_metrics['RMSE']:.4f}"
    )

    print(
        f"[RESULT] Baseline R²   : "
        f"{baseline_metrics['R2']:.4f}"
    )

    # --------------------------------------------------------
    # 6. COMPARAISON
    # --------------------------------------------------------

    print("\n=== 6. COMPARAISON FINALE ===")

    mae_gain, rmse_gain = save_metrics(
        final_metrics,
        baseline_metrics,
    )

    print(
        f"[RESULT] Gain MAE  : "
        f"{mae_gain:.2f}%"
    )

    print(
        f"[RESULT] Gain RMSE : "
        f"{rmse_gain:.2f}%"
    )

    # --------------------------------------------------------
    # 7. SAUVEGARDE
    # --------------------------------------------------------

    print("\n=== 7. SAUVEGARDE ===")

    save_predictions(
        test_split,
        predictions,
    )

    print(
        f"[OK] Métriques sauvegardées : "
        f"{OUTPUT_METRICS}"
    )

    # --------------------------------------------------------
    # 8. VALIDATION FINALE
    # --------------------------------------------------------

    print("\n=== 8. VALIDATION FINALE ===")

    if len(predictions) != 434:
        raise ValueError(
            "Le nombre de prédictions est incorrect."
        )

    if predictions.isna().any():
        raise ValueError(
            "Des prédictions NULL ont été générées."
        )

    if pd.Series(y_true).isna().any():
        raise ValueError(
            "Des valeurs réelles NULL existent."
        )

    print(
        "[PASS] 434 prédictions générées"
    )

    print(
        "[PASS] Aucune prédiction NULL"
    )

    print(
        "[PASS] Test évalué uniquement à J6.9"
    )

    print(
        "[PASS] Évaluation finale terminée"
    )

    print("\n" + "=" * 60)
    print("J6.9 — ÉVALUATION FINALE : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()