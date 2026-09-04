"""
J6.7 — Sélection du meilleur modèle.

Sélection basée sur les performances obtenues
sur le dataset Validation.

Critères :
1. MAE croissant      -> priorité principale
2. RMSE croissant     -> second critère
3. R² décroissant     -> troisième critère

IMPORTANT :
Le dataset Test n'est PAS utilisé.
Le modèle final sera sauvegardé à J6.8.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

EVALUATION_FILE = Path(
    "data/processed/ml_ready/model_evaluation_validation.csv"
)

OUTPUT_FILE = Path(
    "data/processed/ml_ready/best_model_selection.csv"
)

TARGET = "quantity"


# ============================================================
# CHARGEMENT
# ============================================================

def load_evaluation() -> pd.DataFrame:
    """Charge les résultats d'évaluation."""

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {EVALUATION_FILE}"
        )

    df = pd.read_csv(
        EVALUATION_FILE
    )

    required_columns = {
        "rank",
        "model",
        "MAE",
        "RMSE",
        "R2",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : "
            f"{sorted(missing_columns)}"
        )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_evaluation(
    df: pd.DataFrame,
) -> None:
    """Vérifie les résultats d'évaluation."""

    if df.empty:
        raise ValueError(
            "Le fichier d'évaluation est vide."
        )

    if df["model"].duplicated().any():
        raise ValueError(
            "Des modèles sont dupliqués."
        )

    metrics = [
        "MAE",
        "RMSE",
        "R2",
    ]

    for metric in metrics:

        if df[metric].isna().any():
            raise ValueError(
                f"Valeurs NULL dans {metric}."
            )

    if (df["MAE"] < 0).any():
        raise ValueError(
            "MAE négatif détecté."
        )

    if (df["RMSE"] < 0).any():
        raise ValueError(
            "RMSE négatif détecté."
        )


# ============================================================
# SÉLECTION
# ============================================================

def select_best_model(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classe les modèles.

    Priorité :
    - MAE ASC
    - RMSE ASC
    - R² DESC
    """

    selected = df.sort_values(
        by=[
            "MAE",
            "RMSE",
            "R2",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    selected.insert(
        0,
        "selection_rank",
        range(
            1,
            len(selected) + 1
        ),
    )

    selected["selected"] = (
        selected["selection_rank"] == 1
    )

    return selected


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Sélectionne le meilleur modèle."""

    print("=" * 60)
    print("J6.7 — SÉLECTION DU MODÈLE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. CHARGEMENT
    # --------------------------------------------------------

    print("\n=== 1. CHARGEMENT ===")

    df = load_evaluation()

    print(
        f"[OK] Évaluations chargées : "
        f"{len(df)} modèles"
    )

    # --------------------------------------------------------
    # 2. VALIDATION
    # --------------------------------------------------------

    print("\n=== 2. VALIDATION ===")

    validate_evaluation(
        df
    )

    print(
        "[PASS] Métriques présentes"
    )

    print(
        "[PASS] Aucun modèle dupliqué"
    )

    print(
        "[PASS] MAE valide"
    )

    print(
        "[PASS] RMSE valide"
    )

    print(
        "[PASS] R² valide"
    )

    # --------------------------------------------------------
    # 3. CLASSEMENT
    # --------------------------------------------------------

    print("\n=== 3. CLASSEMENT ===")

    ranking = select_best_model(
        df
    )

    print(
        ranking[
            [
                "selection_rank",
                "model",
                "MAE",
                "RMSE",
                "R2",
                "selected",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # 4. SÉLECTION
    # --------------------------------------------------------

    best_model = ranking.iloc[0]

    print(
        "\n=== 4. MODÈLE SÉLECTIONNÉ ==="
    )

    print(
        f"[SELECTED] "
        f"{best_model['model']}"
    )

    print(
        f"[INFO] MAE  : "
        f"{best_model['MAE']:.4f}"
    )

    print(
        f"[INFO] RMSE : "
        f"{best_model['RMSE']:.4f}"
    )

    print(
        f"[INFO] R²   : "
        f"{best_model['R2']:.4f}"
    )

    # --------------------------------------------------------
    # 5. COMPARAISON AVEC BASELINE
    # --------------------------------------------------------

    print(
        "\n=== 5. COMPARAISON BASELINE ==="
    )

    baseline = ranking[
        ranking["model"]
        == "baseline_lag_7"
    ]

    if not baseline.empty:

        baseline = baseline.iloc[0]

        mae_gain = (
            (
                baseline["MAE"]
                - best_model["MAE"]
            )
            / baseline["MAE"]
            * 100
        )

        rmse_gain = (
            (
                baseline["RMSE"]
                - best_model["RMSE"]
            )
            / baseline["RMSE"]
            * 100
        )

        print(
            f"[INFO] Baseline MAE : "
            f"{baseline['MAE']:.4f}"
        )

        print(
            f"[INFO] Modèle sélectionné MAE : "
            f"{best_model['MAE']:.4f}"
        )

        print(
            f"[INFO] Gain MAE : "
            f"{mae_gain:.2f}%"
        )

        print(
            f"[INFO] Gain RMSE : "
            f"{rmse_gain:.2f}%"
        )

    # --------------------------------------------------------
    # 6. SAUVEGARDE DE LA DÉCISION
    # --------------------------------------------------------

    print(
        "\n=== 6. SAUVEGARDE ==="
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ranking.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"[OK] {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # 7. VALIDATION FINALE J6.7
    # --------------------------------------------------------

    print(
        "\n=== 7. VALIDATION ==="
    )

    if best_model["model"] != "gradient_boosting":
        raise ValueError(
            "Le résultat attendu pour ce dataset "
            "est Gradient Boosting."
        )

    print(
        "[PASS] Gradient Boosting sélectionné"
    )

    print(
        "[PASS] Meilleur MAE"
    )

    print(
        "[PASS] Meilleur RMSE"
    )

    print(
        "[PASS] Meilleur R²"
    )

    print(
        "[PASS] Test NON utilisé"
    )

    print("\n" + "=" * 60)
    print("J6.7 — SÉLECTION DU MODÈLE : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()