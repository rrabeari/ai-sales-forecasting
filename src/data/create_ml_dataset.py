"""
AI Sales Forecasting
J5.5 - Construction du dataset ML final

Objectif :
Préparer le dataset final destiné au Machine Learning.

Source :
    data/processed/sales_rolling_features.csv

Sortie :
    data/processed/sales_ml_ready.csv

Target :
    quantity

Les lignes ne disposant pas de l'historique nécessaire
pour les variables LAG / Rolling sont supprimées.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_rolling_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_ml_ready.csv"
)

EXPECTED_SOURCE_ROWS = 5_110
EXPECTED_PRODUCTS = 14
EXPECTED_DAYS = 365

EXPECTED_FINAL_ROWS = 4_690

FEATURE_COLUMNS = [
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

ORIGINAL_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "revenue",
]

EXPECTED_COLUMNS = ORIGINAL_COLUMNS + FEATURE_COLUMNS


# ============================================================
# CHARGEMENT
# ============================================================

def load_data():
    """Charge le dataset J5.4."""

    print()
    print("=== CHARGEMENT DU DATASET J5.4 ===")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"[OK] Fichier chargé : {INPUT_FILE}")
    print(f"[OK] Nombre de lignes : {len(df):,}")

    return df


# ============================================================
# VALIDATION SOURCE
# ============================================================

def validate_source(df):
    """Valide le dataset J5.4."""

    print()
    print("=== VALIDATION DU DATASET SOURCE ===")

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    print("[PASS] Toutes les colonnes J5.4 sont présentes")

    if len(df) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Nombre de lignes incorrect : {len(df)}"
        )

    print(
        f"[PASS] {EXPECTED_SOURCE_ROWS:,} lignes présentes"
    )

    df["date"] = pd.to_datetime(df["date"])

    if df["date"].isna().any():
        raise ValueError("Dates NULL détectées")

    print("[PASS] Toutes les dates sont valides")

    product_count = df["product_id"].nunique()

    if product_count != EXPECTED_PRODUCTS:
        raise ValueError(
            f"Nombre de produits incorrect : {product_count}"
        )

    print(
        f"[PASS] {EXPECTED_PRODUCTS} produits présents"
    )

    day_count = df["date"].nunique()

    if day_count != EXPECTED_DAYS:
        raise ValueError(
            f"Nombre de jours incorrect : {day_count}"
        )

    print(
        f"[PASS] {EXPECTED_DAYS} jours présents"
    )

    if df["quantity"].isna().any():
        raise ValueError(
            "quantity contient des NULL"
        )

    if (df["quantity"] < 0).any():
        raise ValueError(
            "quantity contient des valeurs négatives"
        )

    print("[PASS] quantity valide")


# ============================================================
# ANALYSE DES NULL
# ============================================================

def analyze_nulls(df):
    """Analyse les NULL avant suppression."""

    print()
    print("=== ANALYSE DES NULL AVANT NETTOYAGE ML ===")

    null_counts = df[FEATURE_COLUMNS].isna().sum()

    print(null_counts)

    expected_nulls = {
        "lag_1": 14,
        "lag_7": 98,
        "lag_14": 196,
        "rolling_mean_7": 98,
        "rolling_mean_14": 196,
        "rolling_mean_30": 420,
    }

    for column, expected in expected_nulls.items():

        actual = null_counts[column]

        if actual != expected:
            raise ValueError(
                f"{column}: {actual} NULL trouvés, "
                f"{expected} attendus"
            )

    print(
        "[PASS] NULL historiques conformes "
        "aux attentes J5.3/J5.4"
    )


# ============================================================
# TRI
# ============================================================

def sort_data(df):
    """Trie les données par produit et date."""

    print()
    print("=== TRI CHRONOLOGIQUE ===")

    df = (
        df
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    for product_id, group in df.groupby("product_id"):

        if not group["date"].is_monotonic_increasing:
            raise ValueError(
                f"Ordre chronologique incorrect "
                f"pour product_id={product_id}"
            )

    print(
        "[PASS] Ordre chronologique validé "
        "pour chaque produit"
    )

    return df


# ============================================================
# CREATION DATASET ML
# ============================================================

def create_ml_dataset(df):
    """Supprime uniquement les lignes sans historique complet."""

    print()
    print("=== CONSTRUCTION DU DATASET ML ===")

    before = len(df)

    print(
        f"[INFO] Lignes avant filtrage : {before:,}"
    )

    # Le rolling_mean_30 est la feature nécessitant
    # le plus d'historique.
    #
    # Si cette feature est disponible, les autres
    # LAG / Rolling nécessaires le sont également.

    df_ml = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    after = len(df_ml)

    removed = before - after

    print(
        f"[OK] Lignes après filtrage : {after:,}"
    )

    print(
        f"[OK] Lignes retirées : {removed:,}"
    )

    return df_ml


# ============================================================
# VALIDATION DATASET ML
# ============================================================

def validate_ml_dataset(df_ml, original_df):
    """Valide le dataset ML final."""

    print()
    print("=== VALIDATION DU DATASET ML FINAL ===")

    # --------------------------------------------------------
    # Nombre de lignes
    # --------------------------------------------------------

    if len(df_ml) != EXPECTED_FINAL_ROWS:
        raise ValueError(
            f"Nombre de lignes final incorrect : "
            f"{len(df_ml)} "
            f"(attendu : {EXPECTED_FINAL_ROWS})"
        )

    print(
        f"[PASS] Nombre de lignes final : "
        f"{len(df_ml):,}"
    )

    # --------------------------------------------------------
    # Produits
    # --------------------------------------------------------

    product_count = df_ml["product_id"].nunique()

    if product_count != EXPECTED_PRODUCTS:
        raise ValueError(
            f"Nombre de produits final incorrect : "
            f"{product_count}"
        )

    print(
        f"[PASS] {EXPECTED_PRODUCTS} produits conservés"
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    min_date = df_ml["date"].min()
    max_date = df_ml["date"].max()

    expected_min = pd.Timestamp("2025-10-01")
    expected_max = pd.Timestamp("2026-08-31")

    if min_date != expected_min:
        raise ValueError(
            f"Date minimale incorrecte : "
            f"{min_date.date()}"
        )

    if max_date != expected_max:
        raise ValueError(
            f"Date maximale incorrecte : "
            f"{max_date.date()}"
        )

    print(
        f"[PASS] Période ML : "
        f"{min_date.date()} → {max_date.date()}"
    )

    # --------------------------------------------------------
    # NULL
    # --------------------------------------------------------

    nulls = df_ml[FEATURE_COLUMNS].isna().sum()

    if nulls.sum() != 0:
        raise ValueError(
            "Des NULL sont encore présents dans "
            "les features ML"
        )

    print(
        "[PASS] Aucun NULL dans les features ML"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if df_ml["quantity"].isna().any():
        raise ValueError(
            "quantity contient des NULL"
        )

    print("[PASS] Target quantity sans NULL")

    if (df_ml["quantity"] < 0).any():
        raise ValueError(
            "quantity contient des valeurs négatives"
        )

    print("[PASS] Target quantity valide")

    # --------------------------------------------------------
    # Unicité
    # --------------------------------------------------------

    duplicates = df_ml.duplicated(
        subset=["date", "product_id"]
    ).sum()

    if duplicates != 0:
        raise ValueError(
            f"{duplicates} doublons date + product_id"
        )

    print(
        "[PASS] Aucune clé date + product_id dupliquée"
    )

    # --------------------------------------------------------
    # Nombre de jours par produit
    # --------------------------------------------------------

    days_per_product = (
        df_ml
        .groupby("product_id")["date"]
        .nunique()
    )

    if not (days_per_product == 335).all():
        raise ValueError(
            "Chaque produit doit avoir exactement "
            "335 jours dans le dataset ML"
        )

    print(
        "[PASS] 335 jours ML par produit"
    )

    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    if list(df_ml.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Structure des colonnes incorrecte"
        )

    print(
        "[PASS] Structure finale : "
        "18 colonnes"
    )

    # --------------------------------------------------------
    # Vérification des données originales
    # --------------------------------------------------------

    print()
    print("=== VÉRIFICATION INTÉGRITÉ DES DONNÉES ===")

    original_sorted = (
        original_df[
            ["date", "product_id"]
            + ORIGINAL_COLUMNS[2:]
        ]
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    ml_sorted = (
        df_ml[
            ["date", "product_id"]
            + ORIGINAL_COLUMNS[2:]
        ]
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    # Vérifier que chaque ligne ML existe bien
    # dans le dataset source.

    source_keys = set(
        zip(
            original_df["date"],
            original_df["product_id"]
        )
    )

    ml_keys = set(
        zip(
            df_ml["date"],
            df_ml["product_id"]
        )
    )

    if not ml_keys.issubset(source_keys):
        raise ValueError(
            "Le dataset ML contient des lignes "
            "absentes de la source"
        )

    print(
        "[PASS] Toutes les lignes ML proviennent "
        "du dataset J5.4"
    )

    # --------------------------------------------------------
    # Vérification target
    # --------------------------------------------------------

    if df_ml["quantity"].sum() <= 0:
        raise ValueError(
            "Somme quantity invalide"
        )

    print(
        "[PASS] Target quantity cohérente"
    )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_data(df_ml):
    """Sauvegarde le dataset ML."""

    print()
    print("=== SAUVEGARDE ===")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_ml.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"[OK] Fichier sauvegardé : "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# RESUME
# ============================================================

def print_summary(df_ml, original_rows):
    """Affiche le résumé final."""

    removed = original_rows - len(df_ml)

    print()
    print("============================================================")
    print("J5.5 — RÉSUMÉ FINAL")
    print("============================================================")

    print(
        f"Lignes source           : {original_rows:,}"
    )

    print(
        f"Lignes dataset ML       : {len(df_ml):,}"
    )

    print(
        f"Lignes retirées         : {removed:,}"
    )

    print(
        f"Produits                : "
        f"{df_ml['product_id'].nunique()}"
    )

    print(
        f"Jours par produit       : "
        f"{df_ml.groupby('product_id')['date'].nunique().iloc[0]}"
    )

    print(
        f"Colonnes                : "
        f"{len(df_ml.columns)}"
    )

    print(
        f"Période ML              : "
        f"{df_ml['date'].min().date()} → "
        f"{df_ml['date'].max().date()}"
    )

    print()
    print("Target ML : quantity")

    print()
    print("Features ML :")

    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")

    print()
    print("============================================================")
    print("J5.5 — DATASET ML FINAL : OK")
    print("============================================================")


# ============================================================
# MAIN
# ============================================================

def main():

    print("============================================================")
    print("AI Sales Forecasting")
    print("J5.5 - Construction du Dataset ML final")
    print("============================================================")

    # Chargement
    original_df = load_data()

    # Validation source
    validate_source(original_df)

    # Analyse NULL
    analyze_nulls(original_df)

    # Tri
    df = sort_data(original_df.copy())

    # Construction ML
    df_ml = create_ml_dataset(df)

    # Validation finale
    validate_ml_dataset(
        df_ml,
        original_df
    )

    # Sauvegarde
    save_data(df_ml)

    # Résumé
    print_summary(
        df_ml,
        len(original_df)
    )


if __name__ == "__main__":
    main()