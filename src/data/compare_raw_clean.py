"""
AI Sales Forecasting
J3.5 - RAW vs CLEAN comparison

Compare le dataset original (RAW) avec le dataset nettoyé (CLEAN).
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kshop_sales_synthetic.csv"
)

CLEAN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_clean.csv"
)


def main() -> None:
    """Compare les datasets RAW et CLEAN."""

    print("AI Sales Forecasting")
    print("J3.5 - RAW vs CLEAN")
    print("-" * 50)

    # -----------------------------------------------------------------------
    # 1. Vérification des fichiers
    # -----------------------------------------------------------------------

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Fichier RAW introuvable : {RAW_FILE}"
        )

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Fichier CLEAN introuvable : {CLEAN_FILE}"
        )

    # -----------------------------------------------------------------------
    # 2. Chargement
    # -----------------------------------------------------------------------

    raw = pd.read_csv(RAW_FILE)
    clean = pd.read_csv(CLEAN_FILE)

    # -----------------------------------------------------------------------
    # 3. Structure
    # -----------------------------------------------------------------------

    print()
    print("=== STRUCTURE ===")

    print(f"RAW lignes   : {len(raw):,}")
    print(f"CLEAN lignes : {len(clean):,}")

    print(f"RAW colonnes   : {len(raw.columns)}")
    print(f"CLEAN colonnes : {len(clean.columns)}")

    # -----------------------------------------------------------------------
    # 4. Colonnes
    # -----------------------------------------------------------------------

    print()
    print("=== COLONNES ===")

    columns_identical = list(raw.columns) == list(clean.columns)

    print(
        f"Colonnes identiques : "
        f"{'OUI' if columns_identical else 'NON'}"
    )

    if not columns_identical:
        print(f"RAW   : {list(raw.columns)}")
        print(f"CLEAN : {list(clean.columns)}")

    # -----------------------------------------------------------------------
    # 5. Nombre de lignes
    # -----------------------------------------------------------------------

    print()
    print("=== LIGNES ===")

    row_difference = len(clean) - len(raw)

    print(f"Différence de lignes : {row_difference:+,}")

    if row_difference == 0:
        print("[PASS] Aucune ligne perdue")
    else:
        print("[INFO] Le nombre de lignes a changé")

    # -----------------------------------------------------------------------
    # 6. Clé métier date + produit
    # -----------------------------------------------------------------------

    print()
    print("=== CLÉ DATE + PRODUIT ===")

    raw_keys = set(
        zip(
            raw["date"].astype(str),
            raw["product_id"],
        )
    )

    clean_keys = set(
        zip(
            clean["date"].astype(str),
            clean["product_id"],
        )
    )

    missing_from_clean = raw_keys - clean_keys
    added_to_clean = clean_keys - raw_keys

    print(
        f"Lignes RAW absentes du CLEAN : "
        f"{len(missing_from_clean):,}"
    )

    print(
        f"Lignes ajoutées dans CLEAN    : "
        f"{len(added_to_clean):,}"
    )

    if not missing_from_clean and not added_to_clean:
        print("[PASS] Clés date + produit identiques")

    # -----------------------------------------------------------------------
    # 7. Quantité
    # -----------------------------------------------------------------------

    print()
    print("=== QUANTITÉ ===")

    raw_quantity = raw["quantity"].sum()
    clean_quantity = clean["quantity"].sum()

    print(f"RAW quantité totale   : {raw_quantity:,}")
    print(f"CLEAN quantité totale : {clean_quantity:,}")
    print(
        f"Différence            : "
        f"{clean_quantity - raw_quantity:+,}"
    )

    if raw_quantity == clean_quantity:
        print("[PASS] Quantité totale identique")

    # -----------------------------------------------------------------------
    # 8. Revenue
    # -----------------------------------------------------------------------

    print()
    print("=== REVENUE ===")

    raw_revenue = raw["revenue"].sum()
    clean_revenue = clean["revenue"].sum()

    print(f"RAW CA total   : {raw_revenue:,} AR")
    print(f"CLEAN CA total : {clean_revenue:,} AR")
    print(
        f"Différence     : "
        f"{clean_revenue - raw_revenue:+,} AR"
    )

    if raw_revenue == clean_revenue:
        print("[PASS] CA total identique")

    # -----------------------------------------------------------------------
    # 9. Nombre de produits
    # -----------------------------------------------------------------------

    print()
    print("=== PRODUITS ===")

    raw_products = set(raw["product_id"].unique())
    clean_products = set(clean["product_id"].unique())

    print(f"Produits RAW   : {len(raw_products)}")
    print(f"Produits CLEAN : {len(clean_products)}")

    if raw_products == clean_products:
        print("[PASS] Liste des produits identique")

    # -----------------------------------------------------------------------
    # 10. Période
    # -----------------------------------------------------------------------

    print()
    print("=== PÉRIODE ===")

    raw["date"] = pd.to_datetime(raw["date"])
    clean["date"] = pd.to_datetime(clean["date"])

    raw_start = raw["date"].min()
    raw_end = raw["date"].max()

    clean_start = clean["date"].min()
    clean_end = clean["date"].max()

    print(f"RAW   : {raw_start.date()} -> {raw_end.date()}")
    print(f"CLEAN : {clean_start.date()} -> {clean_end.date()}")

    if raw_start == clean_start and raw_end == clean_end:
        print("[PASS] Période identique")

    # -----------------------------------------------------------------------
    # 11. Différences cellule par cellule
    # -----------------------------------------------------------------------

    print()
    print("=== DIFFÉRENCES ===")

    comparison_columns = [
        "date",
        "product_id",
        "product_name",
        "category",
        "quantity",
        "unit_price",
        "revenue",
    ]

    raw_compare = raw[comparison_columns].copy()
    clean_compare = clean[comparison_columns].copy()

    # Même ordre pour permettre une comparaison ligne par ligne
    raw_compare = raw_compare.sort_values(
        ["date", "product_id"]
    ).reset_index(drop=True)

    clean_compare = clean_compare.sort_values(
        ["date", "product_id"]
    ).reset_index(drop=True)

    if len(raw_compare) == len(clean_compare):
        differences = raw_compare.ne(clean_compare)

        # Une ligne est considérée différente si au moins une colonne diffère
        different_rows = int(differences.any(axis=1).sum())

        print(
            f"Lignes présentant une différence : "
            f"{different_rows:,}"
        )

        if different_rows == 0:
            print(
                "[PASS] Aucune différence de contenu "
                "entre RAW et CLEAN"
            )
        else:
            print(
                "[INFO] Des différences ont été détectées "
                "et doivent être analysées."
            )

            changed_columns = [
                column
                for column in comparison_columns
                if differences[column].any()
            ]

            print(
                "Colonnes concernées : "
                f"{changed_columns}"
            )

    # -----------------------------------------------------------------------
    # 12. Résumé final
    # -----------------------------------------------------------------------

    print()
    print("=" * 50)
    print("RÉSUMÉ J3.5")
    print("=" * 50)

    print(f"RAW lignes       : {len(raw):,}")
    print(f"CLEAN lignes     : {len(clean):,}")
    print(f"Quantité RAW     : {raw_quantity:,}")
    print(f"Quantité CLEAN   : {clean_quantity:,}")
    print(f"CA RAW           : {raw_revenue:,} AR")
    print(f"CA CLEAN         : {clean_revenue:,} AR")
    print(f"Produits RAW     : {len(raw_products)}")
    print(f"Produits CLEAN   : {len(clean_products)}")

    print()
    print("J3.5 — COMPARAISON RAW vs CLEAN : TERMINÉE")


if __name__ == "__main__":
    main()