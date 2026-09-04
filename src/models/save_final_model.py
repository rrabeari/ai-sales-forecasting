"""
J6.8 — Sauvegarde du modèle final.

Le modèle sélectionné à J6.7 est Gradient Boosting.

IMPORTANT :
- Le modèle a été entraîné uniquement sur Train.
- La sélection a été faite sur Validation.
- Le dataset Test n'est PAS utilisé à cette étape.
"""

from pathlib import Path
import json
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_MODEL = Path(
    "models/gradient_boosting.joblib"
)

FINAL_MODEL = Path(
    "models/final_model.joblib"
)

METADATA_FILE = Path(
    "models/final_model_metadata.json"
)

SELECTION_FILE = Path(
    "data/processed/ml_ready/best_model_selection.csv"
)

TARGET = "quantity"

FEATURES = [
    "product_id",
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
# VALIDATION DES FICHIERS
# ============================================================

def validate_files() -> None:
    """Vérifie que les fichiers nécessaires existent."""

    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(
            f"Modèle source introuvable : {SOURCE_MODEL}"
        )

    if not SELECTION_FILE.exists():
        raise FileNotFoundError(
            f"Fichier de sélection introuvable : "
            f"{SELECTION_FILE}"
        )

    print("[PASS] Modèle Gradient Boosting trouvé")
    print("[PASS] Résultats de sélection trouvés")


# ============================================================
# VALIDATION DE LA SÉLECTION
# ============================================================

def validate_selection() -> dict:
    """Vérifie que Gradient Boosting est bien sélectionné."""

    import pandas as pd

    df = pd.read_csv(SELECTION_FILE)

    required_columns = {
        "model",
        "MAE",
        "RMSE",
        "R2",
        "selected",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing)}"
        )

    selected = df[df["selected"] == True]

    if len(selected) != 1:
        raise ValueError(
            "Le fichier doit contenir exactement "
            "un modèle sélectionné."
        )

    best = selected.iloc[0]

    if best["model"] != "gradient_boosting":
        raise ValueError(
            "Le modèle sélectionné n'est pas "
            "Gradient Boosting."
        )

    print("[PASS] Gradient Boosting confirmé comme modèle sélectionné")

    return {
        "model": best["model"],
        "MAE": float(best["MAE"]),
        "RMSE": float(best["RMSE"]),
        "R2": float(best["R2"]),
    }


# ============================================================
# SAUVEGARDE
# ============================================================

def save_final_model() -> None:
    """Copie le modèle sélectionné comme modèle final."""

    FINAL_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        SOURCE_MODEL,
        FINAL_MODEL,
    )

    print(
        f"[OK] Modèle final sauvegardé : "
        f"{FINAL_MODEL}"
    )


# ============================================================
# MÉTADONNÉES
# ============================================================

def save_metadata(metrics: dict) -> None:
    """Sauvegarde les métadonnées du modèle final."""

    metadata = {
        "model_name": "gradient_boosting",
        "model_file": str(FINAL_MODEL),
        "target": TARGET,
        "features": FEATURES,
        "training_dataset": "data/processed/ml_ready/X_train.csv",
        "validation_dataset": "data/processed/ml_ready/X_validation.csv",
        "test_dataset_used": False,
        "selection_stage": "J6.7",
        "save_stage": "J6.8",
        "validation_metrics": metrics,
        "selection_criteria": [
            "MAE ascending",
            "RMSE ascending",
            "R2 descending",
        ],
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"[OK] Métadonnées sauvegardées : "
        f"{METADATA_FILE}"
    )


# ============================================================
# VALIDATION FINALE
# ============================================================

def final_validation() -> None:
    """Vérifie les fichiers produits."""

    if not FINAL_MODEL.exists():
        raise FileNotFoundError(
            "Le modèle final n'a pas été créé."
        )

    if FINAL_MODEL.stat().st_size == 0:
        raise ValueError(
            "Le modèle final est vide."
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Les métadonnées n'ont pas été créées."
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if metadata["model_name"] != "gradient_boosting":
        raise ValueError(
            "Les métadonnées indiquent un mauvais modèle."
        )

    if metadata["test_dataset_used"] is not False:
        raise ValueError(
            "Le Test ne doit pas être utilisé à J6.8."
        )

    if metadata["target"] != TARGET:
        raise ValueError(
            "La cible enregistrée est incorrecte."
        )

    if len(metadata["features"]) != 12:
        raise ValueError(
            "Le nombre de features doit être égal à 12."
        )

    print("[PASS] Modèle final non vide")
    print("[PASS] Métadonnées valides")
    print("[PASS] Target = quantity")
    print("[PASS] 12 features enregistrées")
    print("[PASS] Test NON utilisé")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Exécute J6.8."""

    print("=" * 60)
    print("J6.8 — SAUVEGARDE DU MODÈLE FINAL")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. VALIDATION DES FICHIERS
    # --------------------------------------------------------

    print("\n=== 1. VALIDATION DES FICHIERS ===")

    validate_files()

    # --------------------------------------------------------
    # 2. VALIDATION DE LA SÉLECTION
    # --------------------------------------------------------

    print("\n=== 2. VALIDATION DE LA SÉLECTION ===")

    metrics = validate_selection()

    print(
        f"[INFO] MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"[INFO] RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"[INFO] R²   : {metrics['R2']:.4f}"
    )

    # --------------------------------------------------------
    # 3. SAUVEGARDE DU MODÈLE
    # --------------------------------------------------------

    print("\n=== 3. SAUVEGARDE DU MODÈLE ===")

    save_final_model()

    # --------------------------------------------------------
    # 4. SAUVEGARDE DES MÉTADONNÉES
    # --------------------------------------------------------

    print("\n=== 4. SAUVEGARDE DES MÉTADONNÉES ===")

    save_metadata(metrics)

    # --------------------------------------------------------
    # 5. VALIDATION FINALE
    # --------------------------------------------------------

    print("\n=== 5. VALIDATION FINALE ===")

    final_validation()

    print("\n" + "=" * 60)
    print("J6.8 — MODÈLE FINAL : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()