"""
AI Sales Forecasting
J5.2 - Création des variables calendaires

Objectif
--------
Créer les variables calendaires à partir de la colonne `date`
du dataset nettoyé.

Features créées :
    - day_of_week
    - day_of_month
    - month
    - week_of_year
    - is_weekend

Important
---------
Cette étape ne crée PAS encore les variables LAG ou rolling.
Elles seront créées dans les étapes J5.3 et J5.4.

Le dataset source `sales_clean.csv` ne doit jamais être modifié.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

# Racine du projet.
# Le fichier est situé dans :
# src/data/create_calendar_features.py
#
# parents[2] permet de remonter jusqu'à :
# ai-sales-forecasting/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset nettoyé produit pendant J3.
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "sales_clean.csv"

# Fichier intermédiaire produit par J5.2.
# Il sera utilisé par J5.3 pour créer les LAG.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_calendar_features.csv"
)


# ============================================================
# 2. COLONNES ATTENDUES
# ============================================================

# Colonnes qui doivent obligatoirement être présentes
# dans le dataset source.
EXPECTED_SOURCE_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "revenue",
]

# Les cinq nouvelles variables créées pendant J5.2.
CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
]


# ============================================================
# 3. CHARGEMENT DU DATASET
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Charge le dataset nettoyé.

    Returns
    -------
    pd.DataFrame
        Dataset source.

    Raises
    ------
    FileNotFoundError
        Si sales_clean.csv n'existe pas.
    """

    print("\n=== CHARGEMENT DU DATASET ===")

    # Vérification de l'existence du fichier source.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {INPUT_FILE}"
        )

    # Lecture du CSV.
    df = pd.read_csv(INPUT_FILE)

    print(f"[OK] Fichier chargé : {INPUT_FILE}")
    print(f"[OK] Nombre de lignes : {len(df):,}")

    return df


# ============================================================
# 4. VALIDATION DU DATASET SOURCE
# ============================================================

def validate_source(df: pd.DataFrame) -> None:
    """
    Vérifie que le dataset source respecte les prérequis J5.2.

    Les données du dataset CLEAN doivent rester intactes.
    """

    print("\n=== VALIDATION DU DATASET SOURCE ===")

    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in EXPECTED_SOURCE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            f"{missing_columns}"
        )

    print("[PASS] Colonnes sources présentes")

    # --------------------------------------------------------
    # Vérification du nombre de lignes
    # --------------------------------------------------------

    if len(df) != 5110:
        raise ValueError(
            f"Nombre de lignes inattendu : {len(df)} "
            f"(attendu : 5110)"
        )

    print("[PASS] 5 110 lignes présentes")

    # --------------------------------------------------------
    # Vérification de la colonne date
    # --------------------------------------------------------

    # On convertit temporairement la date pour vérifier
    # qu'elle est correctement interprétable.
    parsed_dates = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Si une date ne peut pas être convertie,
    # elle devient NaT.
    invalid_dates = parsed_dates.isna().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"{invalid_dates} date(s) invalide(s)"
        )

    print("[PASS] Toutes les dates sont valides")

    # --------------------------------------------------------
    # Vérification de la période
    # --------------------------------------------------------

    min_date = parsed_dates.min()
    max_date = parsed_dates.max()

    expected_min = pd.Timestamp("2025-09-01")
    expected_max = pd.Timestamp("2026-08-31")

    if min_date != expected_min:
        raise ValueError(
            f"Date minimale incorrecte : {min_date.date()}"
        )

    if max_date != expected_max:
        raise ValueError(
            f"Date maximale incorrecte : {max_date.date()}"
        )

    print("[PASS] Période correcte : 2025-09-01 → 2026-08-31")

    # --------------------------------------------------------
    # Vérification des produits
    # --------------------------------------------------------

    product_count = df["product_id"].nunique()

    if product_count != 14:
        raise ValueError(
            f"Nombre de produits inattendu : {product_count}"
        )

    print("[PASS] 14 produits présents")

    # --------------------------------------------------------
    # Vérification des jours
    # --------------------------------------------------------

    day_count = parsed_dates.dt.normalize().nunique()

    if day_count != 365:
        raise ValueError(
            f"Nombre de jours inattendu : {day_count}"
        )

    print("[PASS] 365 jours présents")


# ============================================================
# 5. CRÉATION DES VARIABLES CALENDAIRES
# ============================================================

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les cinq variables calendaires définies dans J5.2.

    Les variables sont calculées uniquement à partir de `date`.
    Aucune donnée de vente future n'est utilisée.
    """

    print("\n=== CRÉATION DES VARIABLES CALENDAIRES ===")

    # --------------------------------------------------------
    # Conversion de la date
    # --------------------------------------------------------

    # Conversion en datetime.
    # errors="raise" permet d'arrêter le programme si
    # une date est invalide.
    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    # --------------------------------------------------------
    # day_of_week
    # --------------------------------------------------------
    # Pandas utilise :
    # lundi = 0
    # mardi = 1
    # mercredi = 2
    # jeudi = 3
    # vendredi = 4
    # samedi = 5
    # dimanche = 6

    df["day_of_week"] = df["date"].dt.dayofweek

    # --------------------------------------------------------
    # day_of_month
    # --------------------------------------------------------
    # Numéro du jour dans le mois.
    # Exemple :
    # 2026-08-01 → 1
    # 2026-08-15 → 15
    # 2026-08-31 → 31

    df["day_of_month"] = df["date"].dt.day

    # --------------------------------------------------------
    # month
    # --------------------------------------------------------
    # Numéro du mois :
    # janvier = 1
    # ...
    # décembre = 12

    df["month"] = df["date"].dt.month

    # --------------------------------------------------------
    # week_of_year
    # --------------------------------------------------------
    # Numéro de semaine ISO.
    #
    # isocalendar() permet d'utiliser la convention
    # ISO 8601 pour les semaines.

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # --------------------------------------------------------
    # is_weekend
    # --------------------------------------------------------
    # Samedi = 5
    # Dimanche = 6
    #
    # On crée :
    # 0 = jour de semaine
    # 1 = samedi ou dimanche

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    print("[OK] day_of_week créé")
    print("[OK] day_of_month créé")
    print("[OK] month créé")
    print("[OK] week_of_year créé")
    print("[OK] is_weekend créé")

    return df


# ============================================================
# 6. VALIDATION DES FEATURES
# ============================================================

def validate_calendar_features(
    df: pd.DataFrame,
    original_df: pd.DataFrame
) -> None:
    """
    Vérifie la qualité des cinq nouvelles variables.

    Vérifie également que les colonnes originales importantes
    n'ont pas été modifiées.
    """

    print("\n=== VALIDATION DES FEATURES CALENDAIRES ===")

    # --------------------------------------------------------
    # Vérification de la présence des features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in CALENDAR_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features manquantes : {missing_features}"
        )

    print("[PASS] Les 5 features sont présentes")

    # --------------------------------------------------------
    # Vérification des NULL
    # --------------------------------------------------------

    null_counts = df[CALENDAR_FEATURES].isna().sum()

    total_nulls = int(null_counts.sum())

    if total_nulls > 0:
        raise ValueError(
            "NULL détectés dans les features : "
            f"{null_counts.to_dict()}"
        )

    print("[PASS] Aucun NULL dans les features")

    # --------------------------------------------------------
    # Validation day_of_week
    # --------------------------------------------------------

    valid_day_of_week = df["day_of_week"].between(0, 6).all()

    if not valid_day_of_week:
        raise ValueError(
            "Valeur invalide dans day_of_week"
        )

    print("[PASS] day_of_week valide : 0 → 6")

    # --------------------------------------------------------
    # Validation day_of_month
    # --------------------------------------------------------

    valid_day_of_month = df["day_of_month"].between(1, 31).all()

    if not valid_day_of_month:
        raise ValueError(
            "Valeur invalide dans day_of_month"
        )

    print("[PASS] day_of_month valide : 1 → 31")

    # --------------------------------------------------------
    # Validation month
    # --------------------------------------------------------

    valid_month = df["month"].between(1, 12).all()

    if not valid_month:
        raise ValueError(
            "Valeur invalide dans month"
        )

    print("[PASS] month valide : 1 → 12")

    # --------------------------------------------------------
    # Validation week_of_year
    # --------------------------------------------------------

    valid_week = df["week_of_year"].between(1, 53).all()

    if not valid_week:
        raise ValueError(
            "Valeur invalide dans week_of_year"
        )

    print("[PASS] week_of_year valide : 1 → 53")

    # --------------------------------------------------------
    # Validation is_weekend
    # --------------------------------------------------------

    valid_weekend = df["is_weekend"].isin([0, 1]).all()

    if not valid_weekend:
        raise ValueError(
            "Valeur invalide dans is_weekend"
        )

    print("[PASS] is_weekend valide : 0 / 1")

    # --------------------------------------------------------
    # Validation day_of_week ↔ is_weekend
    # --------------------------------------------------------

    expected_weekend = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    weekend_consistency = (
        df["is_weekend"] == expected_weekend
    ).all()

    if not weekend_consistency:
        raise ValueError(
            "Incohérence entre day_of_week et is_weekend"
        )

    print(
        "[PASS] Cohérence day_of_week ↔ is_weekend"
    )

    # ========================================================
    # Vérification de l'intégrité des données originales
    # ========================================================

    # Le nombre de lignes doit être identique.
    if len(df) != len(original_df):
        raise ValueError(
            "Le nombre de lignes a été modifié"
        )

    print("[PASS] Nombre de lignes inchangé : 5 110")

    # --------------------------------------------------------
    # quantity doit rester identique
    # --------------------------------------------------------

    quantity_unchanged = (
        df["quantity"].reset_index(drop=True)
        == original_df["quantity"].reset_index(drop=True)
    ).all()

    if not quantity_unchanged:
        raise ValueError(
            "La colonne quantity a été modifiée"
        )

    print("[PASS] quantity inchangée")

    # --------------------------------------------------------
    # revenue doit rester identique
    # --------------------------------------------------------

    revenue_unchanged = (
        df["revenue"].reset_index(drop=True)
        == original_df["revenue"].reset_index(drop=True)
    ).all()

    if not revenue_unchanged:
        raise ValueError(
            "La colonne revenue a été modifiée"
        )

    print("[PASS] revenue inchangé")

    # --------------------------------------------------------
    # product_id doit rester identique
    # --------------------------------------------------------

    product_unchanged = (
        df["product_id"].reset_index(drop=True)
        == original_df["product_id"].reset_index(drop=True)
    ).all()

    if not product_unchanged:
        raise ValueError(
            "La colonne product_id a été modifiée"
        )

    print("[PASS] product_id inchangé")

    # --------------------------------------------------------
    # date doit rester identique
    # --------------------------------------------------------

    original_dates = pd.to_datetime(
        original_df["date"]
    ).reset_index(drop=True)

    new_dates = pd.to_datetime(
        df["date"]
    ).reset_index(drop=True)

    dates_unchanged = (
        new_dates == original_dates
    ).all()

    if not dates_unchanged:
        raise ValueError(
            "La colonne date a été modifiée"
        )

    print("[PASS] date inchangée")

    # --------------------------------------------------------
    # Vérification des produits et jours
    # --------------------------------------------------------

    product_count = df["product_id"].nunique()

    if product_count != 14:
        raise ValueError(
            f"Nombre de produits incorrect : {product_count}"
        )

    print("[PASS] 14 produits conservés")

    day_count = df["date"].dt.normalize().nunique()

    if day_count != 365:
        raise ValueError(
            f"Nombre de jours incorrect : {day_count}"
        )

    print("[PASS] 365 jours conservés")


# ============================================================
# 7. VALIDATION DE QUELQUES DATES CONNUES
# ============================================================

def validate_sample_dates(df: pd.DataFrame) -> None:
    """
    Vérifie quelques dates connues afin de confirmer
    manuellement la logique des variables calendaires.
    """

    print("\n=== VÉRIFICATION DATES EXEMPLES ===")

    # --------------------------------------------------------
    # 2025-09-01 = lundi
    # --------------------------------------------------------

    row = df.loc[
        df["date"] == pd.Timestamp("2025-09-01")
    ]

    if row.empty:
        raise ValueError(
            "Date 2025-09-01 introuvable"
        )

    day_values = row["day_of_week"].unique()

    if not (day_values == [0]).all():
        raise ValueError(
            "2025-09-01 devrait être un lundi"
        )

    print(
        "[PASS] 2025-09-01 → lundi → day_of_week = 0"
    )

    # --------------------------------------------------------
    # 2025-09-06 = samedi
    # --------------------------------------------------------

    row = df.loc[
        df["date"] == pd.Timestamp("2025-09-06")
    ]

    if row.empty:
        raise ValueError(
            "Date 2025-09-06 introuvable"
        )

    if not (
        (row["day_of_week"] == 5)
        & (row["is_weekend"] == 1)
    ).all():
        raise ValueError(
            "2025-09-06 devrait être samedi/weekend"
        )

    print(
        "[PASS] 2025-09-06 → samedi → weekend = 1"
    )

    # --------------------------------------------------------
    # 2025-09-07 = dimanche
    # --------------------------------------------------------

    row = df.loc[
        df["date"] == pd.Timestamp("2025-09-07")
    ]

    if row.empty:
        raise ValueError(
            "Date 2025-09-07 introuvable"
        )

    if not (
        (row["day_of_week"] == 6)
        & (row["is_weekend"] == 1)
    ).all():
        raise ValueError(
            "2025-09-07 devrait être dimanche/weekend"
        )

    print(
        "[PASS] 2025-09-07 → dimanche → weekend = 1"
    )


# ============================================================
# 8. SAUVEGARDE
# ============================================================

def save_data(df: pd.DataFrame) -> None:
    """
    Sauvegarde le dataset enrichi.

    Le fichier source sales_clean.csv reste intact.
    """

    print("\n=== SAUVEGARDE ===")

    # Création du dossier de sortie si nécessaire.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Sauvegarde sans index Pandas.
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"[OK] Fichier sauvegardé : {OUTPUT_FILE}"
    )


# ============================================================
# 9. RAPPORT FINAL
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    """
    Affiche un résumé final de J5.2.
    """

    print("\n" + "=" * 60)
    print("J5.2 — RÉSUMÉ FINAL")
    print("=" * 60)

    print(f"Lignes                 : {len(df):,}")
    print(f"Produits               : {df['product_id'].nunique()}")
    print(
        f"Période                : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print("\nFeatures créées :")

    for feature in CALENDAR_FEATURES:
        print(f"  - {feature}")

    print("\nExemple de distribution :")

    print(
        "\nday_of_week :"
    )
    print(
        df["day_of_week"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nis_weekend :"
    )
    print(
        df["is_weekend"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n" + "=" * 60)
    print("J5.2 — VARIABLES CALENDAIRES : OK")
    print("=" * 60)


# ============================================================
# 10. PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    """
    Pipeline principal J5.2.
    """

    print("=" * 60)
    print("AI Sales Forecasting")
    print("J5.2 - Création des variables calendaires")
    print("=" * 60)

    # --------------------------------------------------------
    # Étape 1 : chargement
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Conservation d'une copie du dataset original.
    #
    # Cette copie permet de vérifier que les colonnes
    # importantes n'ont pas été modifiées.
    # --------------------------------------------------------

    original_df = df.copy(deep=True)

    # --------------------------------------------------------
    # Étape 2 : validation source
    # --------------------------------------------------------

    validate_source(df)

    # --------------------------------------------------------
    # Étape 3 : création des features
    # --------------------------------------------------------

    df = create_calendar_features(df)

    # --------------------------------------------------------
    # Étape 4 : validation des nouvelles features
    # --------------------------------------------------------

    validate_calendar_features(
        df,
        original_df
    )

    # --------------------------------------------------------
    # Étape 5 : validation de quelques dates
    # --------------------------------------------------------

    validate_sample_dates(df)

    # --------------------------------------------------------
    # Étape 6 : sauvegarde
    # --------------------------------------------------------

    save_data(df)

    # --------------------------------------------------------
    # Étape 7 : résumé
    # --------------------------------------------------------

    print_summary(df)


# ============================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================

if __name__ == "__main__":
    main()