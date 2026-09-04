"""
AI Sales Forecasting
J5.3 - Création des variables LAG

Objectif
--------
Créer des variables historiques permettant au futur modèle
de connaître la demande passée d'un produit.

Variables créées :
    - lag_1
    - lag_7
    - lag_14

Définition :
    lag_1  = quantité du produit à J-1
    lag_7  = quantité du produit à J-7
    lag_14 = quantité du produit à J-14

Important
---------
Les LAG sont calculés séparément pour chaque produit.

Aucune information future ne doit être utilisée.

Le dataset source de J5.2 n'est jamais modifié.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

# Racine du projet :
# ai-sales-forecasting/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset produit par J5.2.
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_calendar_features.csv"
)

# Dataset intermédiaire produit par J5.3.
# Il sera utilisé ensuite par J5.4.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_lag_features.csv"
)


# ============================================================
# 2. CONFIGURATION DES VARIABLES
# ============================================================

# Colonnes obligatoires provenant de J5.2.
EXPECTED_COLUMNS = [
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
]

# Nouvelles variables créées pendant J5.3.
LAG_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
]

# Nombre de lignes attendu.
EXPECTED_ROWS = 5110

# Nombre de produits attendu.
EXPECTED_PRODUCTS = 14

# Nombre de jours attendu.
EXPECTED_DAYS = 365


# ============================================================
# 3. CHARGEMENT DU DATASET
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Charge le dataset enrichi avec les variables calendaires.

    Returns
    -------
    pd.DataFrame
        Dataset J5.2.
    """

    print("\n=== CHARGEMENT DU DATASET J5.2 ===")

    # Vérifie que le fichier existe.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {INPUT_FILE}"
        )

    # Lecture du CSV.
    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    print(
        f"[OK] Fichier chargé : {INPUT_FILE}"
    )

    print(
        f"[OK] Nombre de lignes : {len(df):,}"
    )

    return df


# ============================================================
# 4. VALIDATION DU DATASET SOURCE
# ============================================================

def validate_source(df: pd.DataFrame) -> None:
    """
    Vérifie que le dataset J5.2 est conforme
    avant de créer les LAG.
    """

    print("\n=== VALIDATION DU DATASET SOURCE ===")

    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            f"{missing_columns}"
        )

    print("[PASS] Toutes les colonnes J5.2 sont présentes")

    # --------------------------------------------------------
    # Vérification du nombre de lignes
    # --------------------------------------------------------

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Nombre de lignes incorrect : {len(df)} "
            f"(attendu : {EXPECTED_ROWS})"
        )

    print("[PASS] 5 110 lignes présentes")

    # --------------------------------------------------------
    # Vérification des produits
    # --------------------------------------------------------

    product_count = df["product_id"].nunique()

    if product_count != EXPECTED_PRODUCTS:
        raise ValueError(
            f"Nombre de produits incorrect : {product_count}"
        )

    print("[PASS] 14 produits présents")

    # --------------------------------------------------------
    # Vérification des jours
    # --------------------------------------------------------

    day_count = df["date"].dt.normalize().nunique()

    if day_count != EXPECTED_DAYS:
        raise ValueError(
            f"Nombre de jours incorrect : {day_count}"
        )

    print("[PASS] 365 jours présents")

    # --------------------------------------------------------
    # Vérification des dates
    # --------------------------------------------------------

    if df["date"].isna().any():
        raise ValueError(
            "Des dates NULL sont présentes"
        )

    print("[PASS] Aucune date NULL")

    # --------------------------------------------------------
    # Vérification de quantity
    # --------------------------------------------------------

    if df["quantity"].isna().any():
        raise ValueError(
            "Des valeurs NULL sont présentes dans quantity"
        )

    if (df["quantity"] < 0).any():
        raise ValueError(
            "Des quantités négatives sont présentes"
        )

    print("[PASS] quantity valide")


# ============================================================
# 5. TRI CHRONOLOGIQUE
# ============================================================

def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trie les données par produit puis par date.

    Ce tri est indispensable avant de calculer les LAG.

    Exemple :

        Produit A
            2025-09-01
            2025-09-02
            2025-09-03

        Produit B
            2025-09-01
            2025-09-02
            2025-09-03
    """

    print("\n=== TRI CHRONOLOGIQUE ===")

    # Tri par produit puis date.
    df = df.sort_values(
        by=["product_id", "date"]
    ).reset_index(drop=True)

    # Vérification que chaque produit est bien chronologique.
    is_sorted = (
        df.groupby("product_id")["date"]
        .apply(lambda x: x.is_monotonic_increasing)
        .all()
    )

    if not is_sorted:
        raise ValueError(
            "Les dates ne sont pas correctement triées"
        )

    print(
        "[PASS] Données triées par product_id + date"
    )

    print(
        "[PASS] Ordre chronologique vérifié pour chaque produit"
    )

    return df


# ============================================================
# 6. CRÉATION DES VARIABLES LAG
# ============================================================

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables lag_1, lag_7 et lag_14.

    Le calcul est effectué séparément pour chaque produit.

    Exemple :

        quantity : 5  7  4  8

        lag_1    : NaN 5  7  4

    Ainsi, lag_1 représente toujours la quantité
    du jour précédent pour le même produit.
    """

    print("\n=== CRÉATION DES VARIABLES LAG ===")

    # --------------------------------------------------------
    # lag_1
    # --------------------------------------------------------
    # Décale la quantité d'une ligne vers le bas
    # pour chaque produit.
    #
    # Comme les données sont quotidiennes et complètes,
    # une ligne précédente correspond à J-1.

    df["lag_1"] = (
        df.groupby("product_id")["quantity"]
        .shift(1)
    )

    print(
        "[OK] lag_1 créé → demande à J-1"
    )

    # --------------------------------------------------------
    # lag_7
    # --------------------------------------------------------
    # Décale la quantité de 7 observations.
    #
    # Notre dataset possède une observation par jour
    # et par produit.
    #
    # Donc 7 observations = 7 jours avant.

    df["lag_7"] = (
        df.groupby("product_id")["quantity"]
        .shift(7)
    )

    print(
        "[OK] lag_7 créé → demande à J-7"
    )

    # --------------------------------------------------------
    # lag_14
    # --------------------------------------------------------
    # Décale la quantité de 14 observations.
    #
    # 14 observations = 14 jours avant.

    df["lag_14"] = (
        df.groupby("product_id")["quantity"]
        .shift(14)
    )

    print(
        "[OK] lag_14 créé → demande à J-14"
    )

    return df


# ============================================================
# 7. VALIDATION DES VARIABLES LAG
# ============================================================

def validate_lag_features(
    df: pd.DataFrame,
    original_df: pd.DataFrame
) -> None:
    """
    Valide les variables LAG et vérifie l'intégrité
    des données originales.
    """

    print("\n=== VALIDATION DES VARIABLES LAG ===")

    # --------------------------------------------------------
    # Vérification de la présence des nouvelles colonnes
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in LAG_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Variables LAG manquantes : {missing_features}"
        )

    print("[PASS] lag_1, lag_7 et lag_14 présents")

    # --------------------------------------------------------
    # Les LAG peuvent contenir des NULL au début.
    #
    # C'est NORMAL.
    #
    # lag_1  : 1 première observation par produit
    # lag_7  : 7 premières observations par produit
    # lag_14 : 14 premières observations par produit
    #
    # Avec 14 produits :
    #
    # lag_1  → 14 NULL
    # lag_7  → 98 NULL
    # lag_14 → 196 NULL
    # --------------------------------------------------------

    lag_nulls = df[LAG_FEATURES].isna().sum()

    print("\nNULL attendus dans les LAG :")
    print(lag_nulls.to_string())

    expected_lag_1_nulls = EXPECTED_PRODUCTS * 1
    expected_lag_7_nulls = EXPECTED_PRODUCTS * 7
    expected_lag_14_nulls = EXPECTED_PRODUCTS * 14

    if lag_nulls["lag_1"] != expected_lag_1_nulls:
        raise ValueError(
            "Nombre de NULL inattendu dans lag_1 : "
            f"{lag_nulls['lag_1']} "
            f"(attendu : {expected_lag_1_nulls})"
        )

    if lag_nulls["lag_7"] != expected_lag_7_nulls:
        raise ValueError(
            "Nombre de NULL inattendu dans lag_7 : "
            f"{lag_nulls['lag_7']} "
            f"(attendu : {expected_lag_7_nulls})"
        )

    if lag_nulls["lag_14"] != expected_lag_14_nulls:
        raise ValueError(
            "Nombre de NULL inattendu dans lag_14 : "
            f"{lag_nulls['lag_14']} "
            f"(attendu : {expected_lag_14_nulls})"
        )

    print(
        "[PASS] Nombre de NULL LAG conforme aux attentes"
    )

    # --------------------------------------------------------
    # Les valeurs non NULL des LAG doivent être positives
    # ou nulles, puisque quantity >= 0.
    # --------------------------------------------------------

    for feature in LAG_FEATURES:

        valid_values = df[feature].dropna()

        if (valid_values < 0).any():
            raise ValueError(
                f"Valeur négative détectée dans {feature}"
            )

    print(
        "[PASS] Toutes les valeurs LAG sont >= 0"
    )

    # ========================================================
    # Vérification manuelle de la logique
    # ========================================================

    print("\n=== VÉRIFICATION LOGIQUE DES LAG ===")

    # On prend le premier produit.
    first_product = (
        df["product_id"]
        .iloc[0]
    )

    product_df = (
        df[df["product_id"] == first_product]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Vérification lag_1
    # --------------------------------------------------------

    lag_1_expected = (
        product_df["quantity"]
        .shift(1)
    )

    lag_1_match = (
        product_df["lag_1"]
        .equals(lag_1_expected)
    )

    if not lag_1_match:
        raise ValueError(
            "Erreur dans le calcul de lag_1"
        )

    print(
        f"[PASS] lag_1 vérifié pour product_id={first_product}"
    )

    # --------------------------------------------------------
    # Vérification lag_7
    # --------------------------------------------------------

    lag_7_expected = (
        product_df["quantity"]
        .shift(7)
    )

    lag_7_match = (
        product_df["lag_7"]
        .equals(lag_7_expected)
    )

    if not lag_7_match:
        raise ValueError(
            "Erreur dans le calcul de lag_7"
        )

    print(
        f"[PASS] lag_7 vérifié pour product_id={first_product}"
    )

    # --------------------------------------------------------
    # Vérification lag_14
    # --------------------------------------------------------

    lag_14_expected = (
        product_df["quantity"]
        .shift(14)
    )

    lag_14_match = (
        product_df["lag_14"]
        .equals(lag_14_expected)
    )

    if not lag_14_match:
        raise ValueError(
            "Erreur dans le calcul de lag_14"
        )

    print(
        f"[PASS] lag_14 vérifié pour product_id={first_product}"
    )

    # ========================================================
    # Vérification importante :
    # les LAG ne doivent jamais utiliser le futur.
    # ========================================================

    # Pour chaque produit, on vérifie que le premier
    # lag_1 non NULL correspond bien à la quantité
    # de la date précédente.

    for product_id, group in df.groupby("product_id"):

        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )

        # Vérification lag_1.
        for index in range(1, len(group)):

            current_lag = group.loc[index, "lag_1"]
            previous_quantity = group.loc[
                index - 1,
                "quantity"
            ]

            if current_lag != previous_quantity:
                raise ValueError(
                    f"Data leakage ou erreur lag_1 "
                    f"pour product_id={product_id}"
                )

    print(
        "[PASS] Absence de data leakage détectée"
    )

        # ========================================================
    # Vérification de l'intégrité des données sources
    # ========================================================
    #
    # IMPORTANT :
    # Le dataset a été trié par product_id + date avant
    # le calcul des LAG.
    #
    # L'ordre des lignes peut donc être différent de celui
    # du dataset original.
    #
    # Il ne faut donc PAS comparer les colonnes ligne par ligne
    # selon leur position.
    #
    # Nous allons comparer les données à l'aide de la clé
    # métier unique :
    #
    #     product_id + date
    #
    # Cela garantit que nous comparons le même produit
    # au même jour.
    # ========================================================

    print(
        "\n=== VÉRIFICATION INTÉGRITÉ DES DONNÉES SOURCES ==="
    )

    # --------------------------------------------------------
    # Vérification du nombre de lignes
    # --------------------------------------------------------

    if len(df) != len(original_df):
        raise ValueError(
            "Le nombre de lignes a été modifié"
        )

    print(
        "[PASS] Nombre de lignes inchangé : 5 110"
    )

    # --------------------------------------------------------
    # Création des copies pour comparaison
    # --------------------------------------------------------

    # On convertit les dates dans le même format.
    original_compare = original_df.copy()

    new_compare = df.copy()

    original_compare["date"] = pd.to_datetime(
        original_compare["date"]
    )

    new_compare["date"] = pd.to_datetime(
        new_compare["date"]
    )

    # --------------------------------------------------------
    # Vérification de la clé date + product_id
    # --------------------------------------------------------

    original_keys = (
        original_compare[
            ["date", "product_id"]
        ]
        .sort_values(
            by=["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    new_keys = (
        new_compare[
            ["date", "product_id"]
        ]
        .sort_values(
            by=["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    keys_match = original_keys.equals(new_keys)

    if not keys_match:
        raise ValueError(
            "Les clés date + product_id ont été modifiées"
        )

    print(
        "[PASS] Clés date + product_id inchangées"
    )

    # --------------------------------------------------------
    # Fonction utilitaire pour comparer une colonne
    # --------------------------------------------------------

    def compare_column(column_name: str) -> None:
        """
        Compare une colonne source entre le dataset original
        et le dataset enrichi.

        La comparaison est effectuée après tri par
        product_id + date afin d'ignorer le changement d'ordre.
        """

        original_values = (
            original_compare[
                ["date", "product_id", column_name]
            ]
            .sort_values(
                by=["product_id", "date"]
            )
            .reset_index(drop=True)
        )

        new_values = (
            new_compare[
                ["date", "product_id", column_name]
            ]
            .sort_values(
                by=["product_id", "date"]
            )
            .reset_index(drop=True)
        )

        if not original_values.equals(new_values):
            raise ValueError(
                f"{column_name} a été modifiée"
            )

        print(
            f"[PASS] {column_name} inchangée"
        )

    # --------------------------------------------------------
    # quantity
    # --------------------------------------------------------

    compare_column("quantity")

    # --------------------------------------------------------
    # revenue
    # --------------------------------------------------------

    compare_column("revenue")

    # --------------------------------------------------------
    # product_id
    # --------------------------------------------------------

    # product_id fait déjà partie de la clé de comparaison
    # date + product_id. Il a donc déjà été validé plus haut.
    print("[PASS] product_id inchangé")

    # --------------------------------------------------------
    # product_name
    # --------------------------------------------------------

    compare_column("product_name")

    # --------------------------------------------------------
    # category
    # --------------------------------------------------------

    compare_column("category")

    # --------------------------------------------------------
    # unit_price
    # --------------------------------------------------------

    compare_column("unit_price")

    # --------------------------------------------------------
    # date
    # --------------------------------------------------------

    # date fait déjà partie de la clé de comparaison
    # date + product_id. Elle a donc déjà été validée plus haut.
    print("[PASS] date inchangée")

    # --------------------------------------------------------
    # Vérification des cinq variables calendaires
    # --------------------------------------------------------
    #
    # Elles doivent également être conservées après le tri.
    # --------------------------------------------------------

    for feature in [
        "day_of_week",
        "day_of_month",
        "month",
        "week_of_year",
        "is_weekend",
    ]:

        compare_column(feature)

    print(
        "[PASS] Toutes les données J5.2 sont intactes"
    )


# ============================================================
# 8. SAUVEGARDE
# ============================================================

def save_data(df: pd.DataFrame) -> None:
    """
    Sauvegarde le dataset enrichi avec les LAG.

    Le dataset J5.2 reste intact.
    """

    print("\n=== SAUVEGARDE ===")

    # Création du dossier si nécessaire.
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
# 9. AFFICHAGE D'UN ÉCHANTILLON
# ============================================================

def print_sample(df: pd.DataFrame) -> None:
    """
    Affiche quelques lignes pour contrôler visuellement
    le résultat des LAG.
    """

    print("\n=== ÉCHANTILLON DES LAG ===")

    # Sélection du premier produit.
    first_product = df["product_id"].iloc[0]

    sample = (
        df[df["product_id"] == first_product]
        .sort_values("date")
        .head(20)
    )

    # Colonnes utiles pour la lecture.
    columns = [
        "date",
        "product_id",
        "quantity",
        "lag_1",
        "lag_7",
        "lag_14",
    ]

    print(
        sample[columns].to_string(index=False)
    )


# ============================================================
# 10. RAPPORT FINAL
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    """
    Affiche le résumé final de J5.3.
    """

    print("\n" + "=" * 60)
    print("J5.3 — RÉSUMÉ FINAL")
    print("=" * 60)

    print(
        f"Lignes                 : {len(df):,}"
    )

    print(
        f"Produits               : "
        f"{df['product_id'].nunique()}"
    )

    print(
        f"Jours                  : "
        f"{df['date'].dt.normalize().nunique()}"
    )

    print(
        f"Période                : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print("\nVariables LAG créées :")

    print(
        "  - lag_1   → quantité à J-1"
    )

    print(
        "  - lag_7   → quantité à J-7"
    )

    print(
        "  - lag_14  → quantité à J-14"
    )

    print("\nNULL structurels :")

    print(
        df[LAG_FEATURES]
        .isna()
        .sum()
        .to_string()
    )

    print("\n" + "=" * 60)
    print("J5.3 — VARIABLES LAG : OK")
    print("=" * 60)


# ============================================================
# 11. PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    """
    Pipeline principal J5.3.
    """

    print("=" * 60)
    print("AI Sales Forecasting")
    print("J5.3 - Création des variables LAG")
    print("=" * 60)

    # --------------------------------------------------------
    # Étape 1 : chargement du dataset J5.2
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Conservation d'une copie du dataset original.
    #
    # Elle permettra de vérifier que les données sources
    # n'ont pas été modifiées.
    # --------------------------------------------------------

    original_df = df.copy(deep=True)

    # --------------------------------------------------------
    # Étape 2 : validation
    # --------------------------------------------------------

    validate_source(df)

    # --------------------------------------------------------
    # Étape 3 : tri chronologique
    # --------------------------------------------------------

    df = sort_data(df)

    # --------------------------------------------------------
    # Étape 4 : création des LAG
    # --------------------------------------------------------

    df = create_lag_features(df)

    # --------------------------------------------------------
    # Étape 5 : validation des LAG
    # --------------------------------------------------------

    validate_lag_features(
        df,
        original_df
    )

    # --------------------------------------------------------
    # Étape 6 : affichage d'un échantillon
    # --------------------------------------------------------

    print_sample(df)

    # --------------------------------------------------------
    # Étape 7 : sauvegarde
    # --------------------------------------------------------

    save_data(df)

    # --------------------------------------------------------
    # Étape 8 : résumé final
    # --------------------------------------------------------

    print_summary(df)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()