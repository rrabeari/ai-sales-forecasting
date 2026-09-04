"""
AI Sales Forecasting
J5.6 - Quality Control du Dataset ML final

Objectif :
Effectuer un contrôle qualité indépendant du dataset
destiné au Machine Learning.

Source :
    data/processed/sales_ml_ready.csv

IMPORTANT :
    Ce script ne modifie jamais le dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_ml_ready.csv"
)

HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_rolling_features.csv"
)

EXPECTED_ROWS = 4_690
EXPECTED_PRODUCTS = 14
EXPECTED_DAYS_PER_PRODUCT = 335

EXPECTED_MIN_DATE = pd.Timestamp("2025-10-01")
EXPECTED_MAX_DATE = pd.Timestamp("2026-08-31")

TARGET = "quantity"

ORIGINAL_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "revenue",
]

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

FEATURE_COLUMNS = (
    CALENDAR_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
)

EXPECTED_COLUMNS = (
    ORIGINAL_COLUMNS
    + FEATURE_COLUMNS
)


# ============================================================
# CHARGEMENT
# ============================================================

def load_dataset():
    """Charge le dataset ML final."""

    print()
    print("=== CHARGEMENT DU DATASET ML ===")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"[OK] Fichier chargé : {INPUT_FILE}"
    )

    print(
        f"[OK] Nombre de lignes : {len(df):,}"
    )

    return df


# ============================================================
# STRUCTURE
# ============================================================

def check_structure(df):
    """Contrôle la structure générale."""

    print()
    print("=== 1. CONTRÔLE STRUCTURE ===")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Nombre de lignes incorrect : "
            f"{len(df)}"
        )

    print(
        f"[PASS] {EXPECTED_ROWS:,} lignes"
    )

    if len(df.columns) != 18:
        raise ValueError(
            f"Nombre de colonnes incorrect : "
            f"{len(df.columns)}"
        )

    print(
        "[PASS] 18 colonnes"
    )

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Ordre ou noms des colonnes incorrects"
        )

    print(
        "[PASS] Structure des 18 colonnes correcte"
    )


# ============================================================
# TYPES
# ============================================================

def check_types(df):
    """Contrôle les types de données."""

    print()
    print("=== 2. CONTRÔLE DES TYPES ===")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            "Dates invalides détectées"
        )

    print(
        "[PASS] date : datetime valide"
    )

    if not pd.api.types.is_numeric_dtype(
        df["product_id"]
    ):
        raise ValueError(
            "product_id n'est pas numérique"
        )

    print(
        "[PASS] product_id : numérique"
    )

    numeric_columns = [
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

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise ValueError(
                f"{column} n'est pas numérique"
            )

    print(
        "[PASS] Variables numériques valides"
    )


# ============================================================
# NULL
# ============================================================

def check_nulls(df):
    """Vérifie l'absence de NULL."""

    print()
    print("=== 3. CONTRÔLE DES NULL ===")

    null_counts = df.isna().sum()

    total_nulls = null_counts.sum()

    if total_nulls != 0:
        print(null_counts[null_counts > 0])

        raise ValueError(
            f"{total_nulls} NULL détectés"
        )

    print(
        "[PASS] Aucun NULL dans le dataset ML"
    )


# ============================================================
# DOUBLONS
# ============================================================

def check_duplicates(df):
    """Contrôle les doublons."""

    print()
    print("=== 4. CONTRÔLE DES DOUBLONS ===")

    duplicate_rows = df.duplicated().sum()

    if duplicate_rows != 0:
        raise ValueError(
            f"{duplicate_rows} doublons complets détectés"
        )

    print(
        "[PASS] Aucun doublon complet"
    )

    duplicate_keys = df.duplicated(
        subset=["date", "product_id"]
    ).sum()

    if duplicate_keys != 0:
        raise ValueError(
            f"{duplicate_keys} doublons date + product_id"
        )

    print(
        "[PASS] Aucune clé date + product_id dupliquée"
    )


# ============================================================
# PRODUITS ET DATES
# ============================================================

def check_temporal_structure(df):
    """Contrôle la structure temporelle."""

    print()
    print("=== 5. CONTRÔLE TEMPOREL ===")

    products = df["product_id"].nunique()

    if products != EXPECTED_PRODUCTS:
        raise ValueError(
            f"Nombre de produits incorrect : {products}"
        )

    print(
        f"[PASS] {EXPECTED_PRODUCTS} produits"
    )

    min_date = df["date"].min()
    max_date = df["date"].max()

    if min_date != EXPECTED_MIN_DATE:
        raise ValueError(
            f"Date minimale incorrecte : "
            f"{min_date.date()}"
        )

    if max_date != EXPECTED_MAX_DATE:
        raise ValueError(
            f"Date maximale incorrecte : "
            f"{max_date.date()}"
        )

    print(
        f"[PASS] Période : "
        f"{min_date.date()} → {max_date.date()}"
    )

    days_per_product = (
        df.groupby("product_id")["date"]
        .nunique()
    )

    if not (
        days_per_product
        == EXPECTED_DAYS_PER_PRODUCT
    ).all():

        print(days_per_product)

        raise ValueError(
            "Nombre de jours incorrect "
            "pour au moins un produit"
        )

    print(
        "[PASS] 335 jours par produit"
    )


# ============================================================
# CONTINUITE TEMPORELLE
# ============================================================

def check_date_continuity(df):
    """Vérifie qu'il n'y a pas de trou temporel."""

    print()
    print("=== 6. CONTINUITÉ TEMPORELLE ===")

    df_sorted = (
        df
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    for product_id, group in df_sorted.groupby(
        "product_id"
    ):

        differences = (
            group["date"]
            .diff()
            .dropna()
        )

        if not (differences == pd.Timedelta(days=1)).all():
            raise ValueError(
                f"Trou temporel détecté "
                f"pour product_id={product_id}"
            )

    print(
        "[PASS] Aucun trou temporel détecté"
    )


# ============================================================
# QUANTITY
# ============================================================

def check_quantity(df):
    """Contrôle la target."""

    print()
    print("=== 7. CONTRÔLE TARGET : quantity ===")

    if df[TARGET].isna().any():
        raise ValueError(
            "quantity contient des NULL"
        )

    if (df[TARGET] < 0).any():
        raise ValueError(
            "quantity contient des valeurs négatives"
        )

    if not np.all(
        df[TARGET] == df[TARGET].round()
    ):
        raise ValueError(
            "quantity contient des valeurs non entières"
        )

    print(
        "[PASS] quantity sans NULL"
    )

    print(
        "[PASS] quantity >= 0"
    )

    print(
        "[PASS] quantity entière"
    )

    print()
    print("Distribution quantity :")

    print(
        df[TARGET].describe().to_string()
    )


# ============================================================
# UNIT PRICE
# ============================================================

def check_price(df):
    """Contrôle le prix."""

    print()
    print("=== 8. CONTRÔLE UNIT_PRICE ===")

    if df["unit_price"].isna().any():
        raise ValueError(
            "unit_price contient des NULL"
        )

    if (df["unit_price"] <= 0).any():
        raise ValueError(
            "unit_price contient des valeurs <= 0"
        )

    print(
        "[PASS] unit_price > 0"
    )


# ============================================================
# REVENUE
# ============================================================

def check_revenue(df):
    """Contrôle la cohérence du chiffre d'affaires."""

    print()
    print("=== 9. CONTRÔLE REVENUE ===")

    expected_revenue = (
        df["quantity"]
        * df["unit_price"]
    )

    if not np.allclose(
        df["revenue"],
        expected_revenue,
        rtol=1e-10,
        atol=1e-10
    ):
        raise ValueError(
            "revenue != quantity × unit_price"
        )

    print(
        "[PASS] revenue = quantity × unit_price"
    )


# ============================================================
# CALENDRIER
# ============================================================

def check_calendar_features(df):
    """Contrôle les variables calendaires."""

    print()
    print("=== 10. CONTRÔLE FEATURES CALENDAIRES ===")

    expected_day_of_week = (
        df["date"].dt.dayofweek
    )

    if not (
        df["day_of_week"]
        == expected_day_of_week
    ).all():

        raise ValueError(
            "day_of_week incohérent"
        )

    print(
        "[PASS] day_of_week cohérent"
    )

    expected_day = df["date"].dt.day

    if not (
        df["day_of_month"]
        == expected_day
    ).all():

        raise ValueError(
            "day_of_month incohérent"
        )

    print(
        "[PASS] day_of_month cohérent"
    )

    expected_month = df["date"].dt.month

    if not (
        df["month"]
        == expected_month
    ).all():

        raise ValueError(
            "month incohérent"
        )

    print(
        "[PASS] month cohérent"
    )

    expected_week = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    if not (
        df["week_of_year"]
        == expected_week
    ).all():

        raise ValueError(
            "week_of_year incohérent"
        )

    print(
        "[PASS] week_of_year cohérent"
    )

    expected_weekend = (
        df["date"]
        .dt.dayofweek
        >= 5
    ).astype(int)

    if not (
        df["is_weekend"]
        == expected_weekend
    ).all():

        raise ValueError(
            "is_weekend incohérent"
        )

    print(
        "[PASS] is_weekend cohérent"
    )


# ============================================================
# LAG
# ============================================================
def check_lags(df):
    """Contrôle les variables LAG à partir de l'historique complet."""

    print()
    print("=== 11. CONTRÔLE FEATURES LAG ===")

    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Historique introuvable : {HISTORY_FILE}"
        )

    history = pd.read_csv(HISTORY_FILE)

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce"
    )

    history = (
        history
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    ml = (
        df
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    expected_lags = {
        "lag_1": 1,
        "lag_7": 7,
        "lag_14": 14,
    }

    for column, shift_value in expected_lags.items():

        expected = (
            history
            .groupby("product_id")["quantity"]
            .shift(shift_value)
        )

        history_check = history[
            ["date", "product_id"]
        ].copy()

        history_check["expected"] = expected

        comparison = ml[
            ["date", "product_id", column]
        ].merge(
            history_check,
            on=["date", "product_id"],
            how="left",
            validate="one_to_one"
        )

        if comparison["expected"].isna().any():
            raise ValueError(
                f"Historique insuffisant pour vérifier {column}"
            )

        if not np.allclose(
            comparison[column],
            comparison["expected"],
            rtol=1e-10,
            atol=1e-10
        ):
            raise ValueError(
                f"{column} incohérent"
            )

        print(
            f"[PASS] {column} cohérent avec l'historique complet"
        )

    print(
        "[PASS] Toutes les variables LAG sont correctes"
    )

# ============================================================
# ROLLING
# ============================================================

def check_rolling(df):
    """Contrôle les Rolling à partir de l'historique complet."""

    print()
    print("=== 12. CONTRÔLE FEATURES ROLLING ===")

    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Historique introuvable : {HISTORY_FILE}"
        )

    history = pd.read_csv(HISTORY_FILE)

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce"
    )

    history = (
        history
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    ml = (
        df
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    expected_windows = {
        "rolling_mean_7": 7,
        "rolling_mean_14": 14,
        "rolling_mean_30": 30,
    }

    for column, window in expected_windows.items():

        expected = (
            history
            .groupby("product_id")["quantity"]
            .transform(
                lambda series: (
                    series
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window
                    )
                    .mean()
                )
            )
        )

        history_check = history[
            ["date", "product_id"]
        ].copy()

        history_check["expected"] = expected

        comparison = ml[
            ["date", "product_id", column]
        ].merge(
            history_check,
            on=["date", "product_id"],
            how="left",
            validate="one_to_one"
        )

        if comparison["expected"].isna().any():
            raise ValueError(
                f"Historique insuffisant pour vérifier {column}"
            )

        if not np.allclose(
            comparison[column],
            comparison["expected"],
            rtol=1e-10,
            atol=1e-10
        ):
            raise ValueError(
                f"{column} incohérente"
            )

        print(
            f"[PASS] {column} cohérente avec l'historique complet"
        )

    print(
        "[PASS] Toutes les variables Rolling sont correctes"
    )

# ============================================================
# DATA LEAKAGE
# ============================================================

def check_no_leakage(df):
    """Vérifie l'absence de fuite temporelle."""

    print()
    print("=== 13. CONTRÔLE DATA LEAKAGE ===")

    df_sorted = (
        df
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    # Les features historiques ne doivent jamais
    # utiliser quantity du jour courant.

    for product_id, group in df_sorted.groupby(
        "product_id"
    ):

        group = group.reset_index(drop=True)

        for index in range(len(group)):

            current_quantity = group.loc[
                index,
                "quantity"
            ]

            if index >= 1:

                lag_1 = group.loc[
                    index,
                    "lag_1"
                ]

                previous_quantity = group.loc[
                    index - 1,
                    "quantity"
                ]

                if lag_1 != previous_quantity:
                    raise ValueError(
                        f"Leakage/incohérence lag_1 "
                        f"product_id={product_id}"
                    )

            if index >= 7:

                rolling_7 = group.loc[
                    index,
                    "rolling_mean_7"
                ]

                previous_7 = group.loc[
                    index - 7:index - 1,
                    "quantity"
                ].mean()

                if not np.isclose(
                    rolling_7,
                    previous_7
                ):
                    raise ValueError(
                        f"Leakage/incohérence "
                        f"rolling_mean_7 "
                        f"product_id={product_id}"
                    )

            # Vérification explicite :
            # la valeur actuelle n'est jamais utilisée
            # dans les calculs historiques.

            if index >= 30:

                rolling_30 = group.loc[
                    index,
                    "rolling_mean_30"
                ]

                previous_30 = group.loc[
                    index - 30:index - 1,
                    "quantity"
                ].mean()

                if not np.isclose(
                    rolling_30,
                    previous_30
                ):
                    raise ValueError(
                        f"Leakage/incohérence "
                        f"rolling_mean_30 "
                        f"product_id={product_id}"
                    )

            _ = current_quantity

    print(
        "[PASS] Aucun data leakage détecté"
    )


# ============================================================
# VALEURS EXTREMES
# ============================================================

def check_extremes(df):
    """Analyse les valeurs extrêmes sans les supprimer."""

    print()
    print("=== 14. ANALYSE DES VALEURS EXTREMES ===")

    q95 = df[TARGET].quantile(0.95)
    q99 = df[TARGET].quantile(0.99)

    count_p95 = (
        df[TARGET] > q95
    ).sum()

    count_p99 = (
        df[TARGET] > q99
    ).sum()

    print(
        f"P95 quantity : {q95:.2f}"
    )

    print(
        f"P99 quantity : {q99:.2f}"
    )

    print(
        f"Observations > P95 : {count_p95}"
    )

    print(
        f"Observations > P99 : {count_p99}"
    )

    if (df[TARGET] > q99).sum() == 0:
        raise ValueError(
            "Analyse extrême incohérente"
        )

    print(
        "[PASS] Valeurs extrêmes analysées"
    )

    print(
        "[INFO] Les valeurs extrêmes sont conservées"
    )


# ============================================================
# DISTRIBUTION PAR PRODUIT
# ============================================================

def check_product_distribution(df):
    """Analyse la distribution par produit."""

    print()
    print("=== 15. DISTRIBUTION PAR PRODUIT ===")

    counts = (
        df.groupby("product_id")
        .size()
    )

    print(counts.to_string())

    if not (
        counts == EXPECTED_DAYS_PER_PRODUCT
    ).all():

        raise ValueError(
            "Distribution incorrecte par produit"
        )

    print(
        "[PASS] Distribution équilibrée : "
        "335 lignes par produit"
    )


# ============================================================
# INTEGRITE FINALE
# ============================================================

def check_final_integrity(df):
    """Dernier contrôle général."""

    print()
    print("=== 16. INTÉGRITÉ FINALE ===")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            "Nombre de lignes modifié"
        )

    if len(df.columns) != 18:
        raise ValueError(
            "Nombre de colonnes modifié"
        )

    if df[EXPECTED_COLUMNS].isna().sum().sum() != 0:
        raise ValueError(
            "NULL détecté"
        )

    print(
        "[PASS] Dataset ML structurellement intact"
    )

    print(
        "[PASS] Dataset ML prêt pour J6"
    )


# ============================================================
# RESUME
# ============================================================

def print_summary(df):
    """Affiche le résumé du Quality Control."""

    print()
    print("============================================================")
    print("J5.6 — QUALITY CONTROL : RÉSUMÉ")
    print("============================================================")

    print(
        f"Lignes          : {len(df):,}"
    )

    print(
        f"Produits        : "
        f"{df['product_id'].nunique()}"
    )

    print(
        f"Jours/produit   : "
        f"{df.groupby('product_id').size().iloc[0]}"
    )

    print(
        f"Colonnes        : {len(df.columns)}"
    )

    print(
        f"Target          : {TARGET}"
    )

    print(
        f"Période         : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print()
    print("Features ML :")

    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")

    print()
    print("============================================================")
    print("J5.6 — QUALITY CONTROL : OK")
    print("============================================================")


# ============================================================
# MAIN
# ============================================================

def main():

    print("============================================================")
    print("AI Sales Forecasting")
    print("J5.6 - Quality Control du Dataset ML")
    print("============================================================")

    df = load_dataset()

    check_structure(df)

    check_types(df)

    check_nulls(df)

    check_duplicates(df)

    check_temporal_structure(df)

    check_date_continuity(df)

    check_quantity(df)

    check_price(df)

    check_revenue(df)

    check_calendar_features(df)

    check_lags(df)

    check_rolling(df)

    check_no_leakage(df)

    check_extremes(df)

    check_product_distribution(df)

    check_final_integrity(df)

    print_summary(df)


if __name__ == "__main__":
    main()