"""
AI Sales Forecasting
J5.7 - Validation finale du Feature Engineering

Objectif :
Valider l'ensemble du pipeline J5 et la cohérence
entre les datasets intermédiaires et le dataset ML final.

IMPORTANT :
    Ce script est un contrôle final.
    Il ne modifie aucun dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_clean.csv"
)

CALENDAR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_calendar_features.csv"
)

LAG_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_lag_features.csv"
)

ROLLING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_rolling_features.csv"
)

ML_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_ml_ready.csv"
)


# ============================================================
# CONSTANTES
# ============================================================

EXPECTED_CLEAN_ROWS = 5_110
EXPECTED_ML_ROWS = 4_690

EXPECTED_PRODUCTS = 14

EXPECTED_FULL_DAYS = 365
EXPECTED_ML_DAYS = 335

EXPECTED_REMOVED_ROWS = 420

EXPECTED_COLUMNS = 18

MIN_DATE = pd.Timestamp("2025-09-01")
MAX_DATE = pd.Timestamp("2026-08-31")

ML_MIN_DATE = pd.Timestamp("2025-10-01")


CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
]

LAG_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
]

ROLLING_FEATURES = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
]

FINAL_FEATURES = (
    CALENDAR_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
)


FINAL_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "revenue",
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
# UTILITAIRE
# ============================================================

def load_file(path, label):
    """Charge un fichier CSV."""

    print(
        f"[OK] Chargement {label} : {path.name}"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            f"Dates invalides dans {label}"
        )

    print(
        f"[OK] {len(df):,} lignes"
    )

    return df


# ============================================================
# CHARGEMENT DES DATASETS
# ============================================================

def load_all_datasets():

    print()
    print("=== 1. CHARGEMENT DES DATASETS J5 ===")

    clean = load_file(
        CLEAN_FILE,
        "J3 sales_clean"
    )

    calendar = load_file(
        CALENDAR_FILE,
        "J5.2 calendar"
    )

    lag = load_file(
        LAG_FILE,
        "J5.3 lag"
    )

    rolling = load_file(
        ROLLING_FILE,
        "J5.4 rolling"
    )

    ml = load_file(
        ML_FILE,
        "J5.5 ML"
    )

    print(
        "[PASS] Tous les datasets J5 sont disponibles"
    )

    return clean, calendar, lag, rolling, ml


# ============================================================
# STRUCTURE DES DATASETS
# ============================================================

def check_pipeline_structure(
    clean,
    calendar,
    lag,
    rolling,
    ml
):

    print()
    print("=== 2. STRUCTURE DU PIPELINE ===")

    if len(clean) != EXPECTED_CLEAN_ROWS:
        raise ValueError(
            "sales_clean.csv incorrect"
        )

    print(
        "[PASS] sales_clean : 5 110 lignes"
    )

    if len(calendar) != EXPECTED_CLEAN_ROWS:
        raise ValueError(
            "sales_calendar_features.csv incorrect"
        )

    print(
        "[PASS] calendar : 5 110 lignes"
    )

    if len(lag) != EXPECTED_CLEAN_ROWS:
        raise ValueError(
            "sales_lag_features.csv incorrect"
        )

    print(
        "[PASS] lag : 5 110 lignes"
    )

    if len(rolling) != EXPECTED_CLEAN_ROWS:
        raise ValueError(
            "sales_rolling_features.csv incorrect"
        )

    print(
        "[PASS] rolling : 5 110 lignes"
    )

    if len(ml) != EXPECTED_ML_ROWS:
        raise ValueError(
            "sales_ml_ready.csv incorrect"
        )

    print(
        "[PASS] ML : 4 690 lignes"
    )


# ============================================================
# NOMBRE DE PRODUITS ET PERIODES
# ============================================================

def check_products_and_periods(
    clean,
    calendar,
    lag,
    rolling,
    ml
):

    print()
    print("=== 3. PRODUITS ET PÉRIODES ===")

    datasets = {
        "clean": clean,
        "calendar": calendar,
        "lag": lag,
        "rolling": rolling,
        "ml": ml,
    }

    for name, df in datasets.items():

        products = df["product_id"].nunique()

        if products != EXPECTED_PRODUCTS:
            raise ValueError(
                f"{name}: nombre de produits incorrect"
            )

        print(
            f"[PASS] {name} : "
            f"{EXPECTED_PRODUCTS} produits"
        )

    # Historique complet

    for name, df in {
        "clean": clean,
        "calendar": calendar,
        "lag": lag,
        "rolling": rolling,
    }.items():

        if df["date"].min() != MIN_DATE:
            raise ValueError(
                f"{name}: date minimale incorrecte"
            )

        if df["date"].max() != MAX_DATE:
            raise ValueError(
                f"{name}: date maximale incorrecte"
            )

    print(
        "[PASS] Historique complet : "
        "2025-09-01 → 2026-08-31"
    )

    # Dataset ML

    if ml["date"].min() != ML_MIN_DATE:
        raise ValueError(
            "Date minimale ML incorrecte"
        )

    if ml["date"].max() != MAX_DATE:
        raise ValueError(
            "Date maximale ML incorrecte"
        )

    print(
        "[PASS] Dataset ML : "
        "2025-10-01 → 2026-08-31"
    )


# ============================================================
# CONSERVATION DES DONNEES SOURCES
# ============================================================

def check_source_integrity(
    clean,
    calendar,
    lag,
    rolling,
    ml
):

    print()
    print("=== 4. INTÉGRITÉ DES DONNÉES SOURCES ===")

    key = [
        "date",
        "product_id",
    ]

    source_columns = [
        "product_name",
        "category",
        "quantity",
        "unit_price",
        "revenue",
    ]

    clean_keyed = (
        clean
        .set_index(key)
        .sort_index()
    )

    calendar_keyed = (
        calendar
        .set_index(key)
        .sort_index()
    )

    lag_keyed = (
        lag
        .set_index(key)
        .sort_index()
    )

    rolling_keyed = (
        rolling
        .set_index(key)
        .sort_index()
    )

    # --------------------------------------------------------
    # Clean -> Calendar
    # --------------------------------------------------------

    for column in source_columns:

        if not np.allclose(
            clean_keyed[column],
            calendar_keyed[column],
            rtol=1e-10,
            atol=1e-10
        ) if pd.api.types.is_numeric_dtype(
            clean_keyed[column]
        ) else not (
            clean_keyed[column]
            == calendar_keyed[column]
        ).all():

            raise ValueError(
                f"Modification détectée : "
                f"{column} entre clean et calendar"
            )

    print(
        "[PASS] Données sources intactes après J5.2"
    )

    # --------------------------------------------------------
    # Calendar -> Lag
    # --------------------------------------------------------

    for column in source_columns + CALENDAR_FEATURES:

        if pd.api.types.is_numeric_dtype(
            calendar_keyed[column]
        ):

            equal = np.allclose(
                calendar_keyed[column],
                lag_keyed[column],
                rtol=1e-10,
                atol=1e-10
            )

        else:

            equal = (
                calendar_keyed[column]
                == lag_keyed[column]
            ).all()

        if not equal:
            raise ValueError(
                f"Modification détectée : "
                f"{column} entre calendar et lag"
            )

    print(
        "[PASS] Données intactes après J5.3"
    )

    # --------------------------------------------------------
    # Lag -> Rolling
    # --------------------------------------------------------

    for column in (
        source_columns
        + CALENDAR_FEATURES
        + LAG_FEATURES
    ):

        if pd.api.types.is_numeric_dtype(
            lag_keyed[column]
        ):

            equal = np.allclose(
                lag_keyed[column].fillna(-999999),
                rolling_keyed[column].fillna(-999999),
                rtol=1e-10,
                atol=1e-10
            )

        else:

            equal = (
                lag_keyed[column]
                == rolling_keyed[column]
            ).all()

        if not equal:
            raise ValueError(
                f"Modification détectée : "
                f"{column} entre lag et rolling"
            )

    print(
        "[PASS] Données intactes après J5.4"
    )

    # --------------------------------------------------------
    # Rolling -> ML
    # --------------------------------------------------------

    ml_keyed = (
        ml
        .set_index(key)
        .sort_index()
    )

    ml_keys = ml_keyed.index

    rolling_subset = (
        rolling_keyed
        .loc[ml_keys]
    )

    for column in source_columns + FINAL_FEATURES:

        if pd.api.types.is_numeric_dtype(
            rolling_subset[column]
        ):

            equal = np.allclose(
                rolling_subset[column],
                ml_keyed[column],
                rtol=1e-10,
                atol=1e-10
            )

        else:

            equal = (
                rolling_subset[column]
                == ml_keyed[column]
            ).all()

        if not equal:
            raise ValueError(
                f"Modification détectée : "
                f"{column} entre rolling et ML"
            )

    print(
        "[PASS] Données ML cohérentes avec J5.4"
    )


# ============================================================
# FEATURES
# ============================================================

def check_feature_presence(
    calendar,
    lag,
    rolling,
    ml
):

    print()
    print("=== 5. PRÉSENCE DES FEATURES ===")

    for feature in CALENDAR_FEATURES:

        if feature not in calendar.columns:
            raise ValueError(
                f"Feature absente : {feature}"
            )

    print(
        "[PASS] 5 features calendaires"
    )

    for feature in LAG_FEATURES:

        if feature not in lag.columns:
            raise ValueError(
                f"Feature absente : {feature}"
            )

    print(
        "[PASS] 3 features LAG"
    )

    for feature in ROLLING_FEATURES:

        if feature not in rolling.columns:
            raise ValueError(
                f"Feature absente : {feature}"
            )

    print(
        "[PASS] 3 features Rolling"
    )

    if list(ml.columns) != FINAL_COLUMNS:
        raise ValueError(
            "Structure finale ML incorrecte"
        )

    print(
        "[PASS] 18 colonnes finales ML"
    )


# ============================================================
# NULLS STRUCTURELS
# ============================================================

def check_structural_nulls(
    lag,
    rolling,
    ml
):

    print()
    print("=== 6. NULLS STRUCTURELS ===")

    expected_lag_nulls = {
        "lag_1": 14,
        "lag_7": 98,
        "lag_14": 196,
    }

    for column, expected in expected_lag_nulls.items():

        actual = lag[column].isna().sum()

        if actual != expected:
            raise ValueError(
                f"{column}: "
                f"{actual} NULL au lieu de {expected}"
            )

        print(
            f"[PASS] {column} : "
            f"{actual} NULL structurels"
        )

    expected_rolling_nulls = {
        "rolling_mean_7": 98,
        "rolling_mean_14": 196,
        "rolling_mean_30": 420,
    }

    for column, expected in expected_rolling_nulls.items():

        actual = rolling[column].isna().sum()

        if actual != expected:
            raise ValueError(
                f"{column}: "
                f"{actual} NULL au lieu de {expected}"
            )

        print(
            f"[PASS] {column} : "
            f"{actual} NULL structurels"
        )

    if ml[FINAL_FEATURES].isna().sum().sum() != 0:
        raise ValueError(
            "NULL détecté dans les features ML"
        )

    print(
        "[PASS] Aucun NULL dans le dataset ML final"
    )


# ============================================================
# REDUCTION ML
# ============================================================

def check_ml_reduction(
    rolling,
    ml
):

    print()
    print("=== 7. RÉDUCTION DU DATASET ML ===")

    removed = len(rolling) - len(ml)

    if removed != EXPECTED_REMOVED_ROWS:
        raise ValueError(
            f"Lignes retirées incorrectes : {removed}"
        )

    print(
        "[PASS] 420 lignes retirées"
    )

    expected_days = (
        EXPECTED_FULL_DAYS
        - 30
    )

    if expected_days != EXPECTED_ML_DAYS:
        raise ValueError(
            "Calcul des jours ML incohérent"
        )

    print(
        "[PASS] 30 premiers jours exclus "
        "pour chaque produit"
    )

    print(
        "[PASS] 335 jours ML par produit"
    )


# ============================================================
# TARGET
# ============================================================

def check_target_integrity(
    clean,
    ml
):

    print()
    print("=== 8. INTÉGRITÉ DE LA TARGET ===")

    clean_keyed = (
        clean
        .set_index(
            ["date", "product_id"]
        )
        .sort_index()
    )

    ml_keyed = (
        ml
        .set_index(
            ["date", "product_id"]
        )
        .sort_index()
    )

    expected_quantity = (
        clean_keyed
        .loc[ml_keyed.index, "quantity"]
    )

    if not (
        expected_quantity
        == ml_keyed["quantity"]
    ).all():

        raise ValueError(
            "quantity modifiée pendant J5"
        )

    print(
        "[PASS] Target quantity conservée"
    )

    if (ml["quantity"] < 0).any():
        raise ValueError(
            "quantity négative détectée"
        )

    print(
        "[PASS] Target quantity valide"
    )


# ============================================================
# ABSENCE DE LEAKAGE
# ============================================================

def check_no_leakage(ml):

    print()
    print("=== 9. VALIDATION ANTI-LEAKAGE ===")

    ml_sorted = (
        ml
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # LAG
    # --------------------------------------------------------

    for product_id, group in ml_sorted.groupby(
        "product_id"
    ):

        group = group.reset_index(drop=True)

        # Toutes les lignes ML ont suffisamment
        # d'historique.

        for i in range(len(group)):

            if i >= 1:

                expected = group.loc[
                    i - 1,
                    "quantity"
                ]

                actual = group.loc[
                    i,
                    "lag_1"
                ]

                if actual != expected:
                    raise ValueError(
                        f"lag_1 incorrect "
                        f"product_id={product_id}"
                    )

            if i >= 7:

                expected = group.loc[
                    i - 7:i - 1,
                    "quantity"
                ].mean()

                actual = group.loc[
                    i,
                    "rolling_mean_7"
                ]

                if not np.isclose(
                    actual,
                    expected
                ):
                    raise ValueError(
                        f"rolling_mean_7 incorrect "
                        f"product_id={product_id}"
                    )

    print(
        "[PASS] LAG et Rolling utilisent uniquement "
        "l'historique"
    )

    print(
        "[PASS] Aucun data leakage détecté"
    )


# ============================================================
# COHERENCE FINALE
# ============================================================

def check_final_consistency(ml):

    print()
    print("=== 10. COHÉRENCE FINALE ===")

    if len(ml.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Nombre final de colonnes incorrect"
        )

    print(
        "[PASS] 18 colonnes finales"
    )

    if len(ml) != EXPECTED_ML_ROWS:
        raise ValueError(
            "Nombre final de lignes incorrect"
        )

    print(
        "[PASS] 4 690 observations ML"
    )

    if ml["product_id"].nunique() != EXPECTED_PRODUCTS:
        raise ValueError(
            "Nombre final de produits incorrect"
        )

    print(
        "[PASS] 14 produits"
    )

    counts = (
        ml.groupby("product_id")
        .size()
    )

    if not (
        counts == EXPECTED_ML_DAYS
    ).all():

        raise ValueError(
            "Nombre de jours incorrect par produit"
        )

    print(
        "[PASS] 335 observations par produit"
    )


# ============================================================
# RESUME FINAL
# ============================================================

def print_summary():

    print()
    print("============================================================")
    print("J5.7 — VALIDATION FINALE FEATURE ENGINEERING")
    print("============================================================")

    print()
    print("PIPELINE VALIDÉ :")

    print(
        "  sales_clean.csv"
        "                 → 5 110 lignes"
    )

    print(
        "  sales_calendar_features.csv"
        "  → 5 110 lignes"
    )

    print(
        "  sales_lag_features.csv"
        "       → 5 110 lignes"
    )

    print(
        "  sales_rolling_features.csv"
        "   → 5 110 lignes"
    )

    print(
        "  sales_ml_ready.csv"
        "            → 4 690 lignes"
    )

    print()
    print("FEATURES FINALES :")

    for feature in FINAL_FEATURES:
        print(
            f"  - {feature}"
        )

    print()
    print("RÉSULTAT :")

    print(
        "  Target               : quantity"
    )

    print(
        "  Produits             : 14"
    )

    print(
        "  Historique           : 365 jours"
    )

    print(
        "  Dataset ML           : 335 jours/produit"
    )

    print(
        "  Observations ML      : 4 690"
    )

    print(
        "  Colonnes finales     : 18"
    )

    print(
        "  Data leakage         : Aucun"
    )

    print(
        "  NULL ML              : Aucun"
    )

    print()
    print("============================================================")
    print("J5.7 — VALIDATION FINALE : OK")
    print("============================================================")


# ============================================================
# MAIN
# ============================================================

def main():

    print("============================================================")
    print("AI Sales Forecasting")
    print("J5.7 - Validation finale du Feature Engineering")
    print("============================================================")

    (
        clean,
        calendar,
        lag,
        rolling,
        ml,
    ) = load_all_datasets()

    check_pipeline_structure(
        clean,
        calendar,
        lag,
        rolling,
        ml
    )

    check_products_and_periods(
        clean,
        calendar,
        lag,
        rolling,
        ml
    )

    check_source_integrity(
        clean,
        calendar,
        lag,
        rolling,
        ml
    )

    check_feature_presence(
        calendar,
        lag,
        rolling,
        ml
    )

    check_structural_nulls(
        lag,
        rolling,
        ml
    )

    check_ml_reduction(
        rolling,
        ml
    )

    check_target_integrity(
        clean,
        ml
    )

    check_no_leakage(
        ml
    )

    check_final_consistency(
        ml
    )

    print_summary()


if __name__ == "__main__":
    main()