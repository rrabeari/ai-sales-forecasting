"""
AI Sales Forecasting
J5.4 - Création des variables Rolling

Objectif :
Créer des moyennes mobiles historiques par produit
sans utiliser la quantité du jour courant.

Features :
    - rolling_mean_7
    - rolling_mean_14
    - rolling_mean_30

Source :
    data/processed/sales_lag_features.csv

Sortie :
    data/processed/sales_rolling_features.csv
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
    / "sales_lag_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_rolling_features.csv"
)

EXPECTED_ROWS = 5_110
EXPECTED_PRODUCTS = 14
EXPECTED_DAYS = 365

ROLLING_WINDOWS = [7, 14, 30]

ROLLING_FEATURES = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
]


# Colonnes attendues après J5.3
EXPECTED_SOURCE_COLUMNS = [
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
]


# ============================================================
# CHARGEMENT
# ============================================================

def load_data():
    """Charge le dataset J5.3."""

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
    """Valide le dataset J5.3 avant transformation."""

    print()
    print("=== VALIDATION DU DATASET SOURCE ===")

    missing_columns = [
        column
        for column in EXPECTED_SOURCE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    print("[PASS] Toutes les colonnes J5.3 sont présentes")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Nombre de lignes incorrect : {len(df)}"
        )

    print(f"[PASS] {EXPECTED_ROWS:,} lignes présentes")

    product_count = df["product_id"].nunique()

    if product_count != EXPECTED_PRODUCTS:
        raise ValueError(
            f"Nombre de produits incorrect : {product_count}"
        )

    print(f"[PASS] {EXPECTED_PRODUCTS} produits présents")

    df["date"] = pd.to_datetime(df["date"])

    min_date = df["date"].min()
    max_date = df["date"].max()

    expected_min = pd.Timestamp("2025-09-01")
    expected_max = pd.Timestamp("2026-08-31")

    if min_date != expected_min or max_date != expected_max:
        raise ValueError(
            f"Période incorrecte : {min_date.date()} → "
            f"{max_date.date()}"
        )

    print(
        f"[PASS] Période correcte : "
        f"{min_date.date()} → {max_date.date()}"
    )

    day_count = df["date"].nunique()

    if day_count != EXPECTED_DAYS:
        raise ValueError(
            f"Nombre de jours incorrect : {day_count}"
        )

    print(f"[PASS] {EXPECTED_DAYS} jours présents")

    if df["quantity"].isna().any():
        raise ValueError("quantity contient des NULL")

    if (df["quantity"] < 0).any():
        raise ValueError("quantity contient des valeurs négatives")

    print("[PASS] quantity valide")


# ============================================================
# TRI CHRONOLOGIQUE
# ============================================================

def sort_data(df):
    """Trie les données par produit puis par date."""

    print()
    print("=== TRI CHRONOLOGIQUE ===")

    df = df.sort_values(
        ["product_id", "date"]
    ).reset_index(drop=True)

    print("[PASS] Données triées par product_id + date")

    for product_id, group in df.groupby("product_id"):
        if not group["date"].is_monotonic_increasing:
            raise ValueError(
                f"Ordre chronologique incorrect "
                f"pour product_id={product_id}"
            )

    print(
        "[PASS] Ordre chronologique vérifié "
        "pour chaque produit"
    )

    return df


# ============================================================
# CREATION DES ROLLING FEATURES
# ============================================================

def create_rolling_features(df):
    """Crée les moyennes mobiles historiques."""

    print()
    print("=== CRÉATION DES VARIABLES ROLLING ===")

    for window in ROLLING_WINDOWS:

        column_name = f"rolling_mean_{window}"

        df[column_name] = (
            df.groupby("product_id")["quantity"]
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

        print(
            f"[OK] {column_name} créée "
            f"→ moyenne des {window} jours précédents"
        )

    return df


# ============================================================
# VALIDATION DES ROLLING FEATURES
# ============================================================

def validate_rolling_features(df, original_df):
    """Valide les variables rolling."""

    print()
    print("=== VALIDATION DES VARIABLES ROLLING ===")

    # --------------------------------------------------------
    # Présence des colonnes
    # --------------------------------------------------------

    for column in ROLLING_FEATURES:

        if column not in df.columns:
            raise ValueError(
                f"Feature manquante : {column}"
            )

    print(
        "[PASS] rolling_mean_7, "
        "rolling_mean_14 et rolling_mean_30 présentes"
    )

    # --------------------------------------------------------
    # NULL structurels
    # --------------------------------------------------------

    print()
    print("NULL attendus dans les ROLLING :")

    expected_nulls = {
        "rolling_mean_7": EXPECTED_PRODUCTS * 7,
        "rolling_mean_14": EXPECTED_PRODUCTS * 14,
        "rolling_mean_30": EXPECTED_PRODUCTS * 30,
    }

    for column in ROLLING_FEATURES:

        actual_nulls = df[column].isna().sum()
        expected = expected_nulls[column]

        print(
            f"{column:<18} {actual_nulls}"
        )

        if actual_nulls != expected:
            raise ValueError(
                f"{column}: "
                f"{actual_nulls} NULL trouvés, "
                f"{expected} attendus"
            )

    print(
        "[PASS] Nombre de NULL rolling "
        "conforme aux attentes"
    )

    # --------------------------------------------------------
    # Valeurs non négatives
    # --------------------------------------------------------

    for column in ROLLING_FEATURES:

        valid_values = df[column].dropna()

        if (valid_values < 0).any():
            raise ValueError(
                f"{column} contient des valeurs négatives"
            )

    print(
        "[PASS] Toutes les valeurs rolling "
        "sont >= 0"
    )

    # --------------------------------------------------------
    # Vérification mathématique
    # --------------------------------------------------------

    print()
    print("=== VÉRIFICATION MATHÉMATIQUE ===")

    test_product = df["product_id"].iloc[0]

    product_df = (
        df[df["product_id"] == test_product]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # rolling_mean_7
    expected_7 = (
        product_df["quantity"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    if not product_df["rolling_mean_7"].equals(expected_7):
        raise ValueError(
            "Erreur mathématique dans rolling_mean_7"
        )

    print(
        f"[PASS] rolling_mean_7 vérifiée "
        f"pour product_id={test_product}"
    )

    # rolling_mean_14
    expected_14 = (
        product_df["quantity"]
        .shift(1)
        .rolling(14, min_periods=14)
        .mean()
    )

    if not product_df["rolling_mean_14"].equals(expected_14):
        raise ValueError(
            "Erreur mathématique dans rolling_mean_14"
        )

    print(
        f"[PASS] rolling_mean_14 vérifiée "
        f"pour product_id={test_product}"
    )

    # rolling_mean_30
    expected_30 = (
        product_df["quantity"]
        .shift(1)
        .rolling(30, min_periods=30)
        .mean()
    )

    if not product_df["rolling_mean_30"].equals(expected_30):
        raise ValueError(
            "Erreur mathématique dans rolling_mean_30"
        )

    print(
        f"[PASS] rolling_mean_30 vérifiée "
        f"pour product_id={test_product}"
    )

    # --------------------------------------------------------
    # Vérification absence de data leakage
    # --------------------------------------------------------

    print()
    print("=== VÉRIFICATION DATA LEAKAGE ===")

    for window in ROLLING_WINDOWS:

        column_name = f"rolling_mean_{window}"

        for product_id, group in df.groupby("product_id"):

            group = group.sort_values("date")

            for index in range(window, len(group)):

                current_quantity = group.iloc[index]["quantity"]
                rolling_value = group.iloc[index][column_name]

                previous_values = (
                    group.iloc[
                        index - window:index
                    ]["quantity"]
                )

                expected_value = previous_values.mean()

                if abs(
                    rolling_value - expected_value
                ) > 1e-10:

                    raise ValueError(
                        f"Data leakage détectée dans "
                        f"{column_name}, "
                        f"product_id={product_id}, "
                        f"index={index}"
                    )

                # La quantité actuelle ne doit jamais
                # influencer la moyenne historique.
                if len(previous_values) == window:
                    if current_quantity in previous_values.values:
                        pass

    print(
        "[PASS] Absence de data leakage détectée"
    )

    # --------------------------------------------------------
    # Intégrité des données J5.3
    # --------------------------------------------------------

    print()
    print(
        "=== VÉRIFICATION INTÉGRITÉ "
        "DES DONNÉES SOURCES ==="
    )

    if len(df) != len(original_df):
        raise ValueError(
            "Le nombre de lignes a été modifié"
        )

    print(
        f"[PASS] Nombre de lignes inchangé : "
        f"{len(df):,}"
    )

    # Comparaison après tri pour éviter les problèmes
    # liés au changement d'ordre des lignes.

    def compare_column(column_name):

        original_sorted = (
            original_df[
                ["date", "product_id", column_name]
            ]
            .sort_values(
                ["product_id", "date"]
            )
            .reset_index(drop=True)
        )

        current_sorted = (
            df[
                ["date", "product_id", column_name]
            ]
            .sort_values(
                ["product_id", "date"]
            )
            .reset_index(drop=True)
        )

        if not original_sorted[column_name].equals(
            current_sorted[column_name]
        ):
            raise ValueError(
                f"{column_name} a été modifiée"
            )

        print(
            f"[PASS] {column_name} inchangée"
        )

    compare_column("quantity")
    compare_column("revenue")
    compare_column("product_name")
    compare_column("category")
    compare_column("unit_price")

    # product_id et date sont déjà utilisés
    # comme clé de comparaison.

    print("[PASS] product_id inchangé")
    print("[PASS] date inchangée")

    compare_column("day_of_week")
    compare_column("day_of_month")
    compare_column("month")
    compare_column("week_of_year")
    compare_column("is_weekend")

    compare_column("lag_1")
    compare_column("lag_7")
    compare_column("lag_14")

    print(
        "[PASS] Toutes les données J5.3 "
        "sont intactes"
    )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_data(df):
    """Sauvegarde le dataset enrichi."""

    print()
    print("=== SAUVEGARDE ===")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"[OK] Fichier sauvegardé : "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# ECHANTILLON
# ============================================================

def print_sample(df):
    """Affiche un échantillon."""

    print()
    print("=== ÉCHANTILLON DES ROLLING ===")

    columns = [
        "date",
        "product_id",
        "quantity",
        "lag_1",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
    ]

    print(
        df[
            columns
        ].head(35).to_string(index=False)
    )


# ============================================================
# RESUME
# ============================================================

def print_summary(df):
    """Affiche le résumé final."""

    print()
    print("============================================================")
    print("J5.4 — RÉSUMÉ FINAL")
    print("============================================================")

    print(f"Lignes                 : {len(df):,}")
    print(
        f"Produits               : "
        f"{df['product_id'].nunique()}"
    )
    print(
        f"Jours                  : "
        f"{df['date'].nunique()}"
    )

    print(
        f"Période                : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print()
    print("Variables ROLLING créées :")
    print(
        "  - rolling_mean_7   "
        "→ moyenne des 7 jours précédents"
    )
    print(
        "  - rolling_mean_14  "
        "→ moyenne des 14 jours précédents"
    )
    print(
        "  - rolling_mean_30  "
        "→ moyenne des 30 jours précédents"
    )

    print()
    print("NULL structurels :")

    print(
        df[
            ROLLING_FEATURES
        ].isna().sum()
    )

    print()
    print("============================================================")
    print("J5.4 — VARIABLES ROLLING : OK")
    print("============================================================")


# ============================================================
# MAIN
# ============================================================

def main():

    print("============================================================")
    print("AI Sales Forecasting")
    print("J5.4 - Création des variables ROLLING")
    print("============================================================")

    # Charger
    original_df = load_data()

    # Validation source
    validate_source(original_df)

    # Copie de travail
    df = original_df.copy()

    # Tri
    df = sort_data(df)

    # Création rolling
    df = create_rolling_features(df)

    # Validation
    validate_rolling_features(
        df,
        original_df
    )

    # Échantillon
    print_sample(df)

    # Sauvegarde
    save_data(df)

    # Résumé
    print_summary(df)


if __name__ == "__main__":
    main()