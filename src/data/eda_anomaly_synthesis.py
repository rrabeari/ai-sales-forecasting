"""
AI Sales Forecasting
J4.5.5 - Synthèse des anomalies & relations

Consolidation des résultats J4.5.1 à J4.5.4.

Aucune observation n'est supprimée.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("data/processed/eda")

ANOMALY_REPORT = BASE_DIR / "anomaly_detection_report.csv"
EXTREME_DAYS_REPORT = BASE_DIR / "extreme_days_analysis.csv"
QUANTITY_PRICE_REPORT = (
    BASE_DIR / "quantity_revenue_price_products.csv"
)
PRODUCT_ANOMALY_REPORT = (
    BASE_DIR / "product_anomalies_analysis.csv"
)

OUTPUT_FILE = BASE_DIR / "eda_anomaly_synthesis.csv"


# ============================================================
# CHARGEMENT
# ============================================================

def load_reports():
    """Charge les quatre rapports J4.5."""

    files = {
        "anomaly": ANOMALY_REPORT,
        "extreme_days": EXTREME_DAYS_REPORT,
        "quantity_price": QUANTITY_PRICE_REPORT,
        "product_anomaly": PRODUCT_ANOMALY_REPORT,
    }

    for name, file in files.items():
        if not file.exists():
            raise FileNotFoundError(
                f"Rapport {name} introuvable : {file}"
            )

    return {
        name: pd.read_csv(file)
        for name, file in files.items()
    }


# ============================================================
# UTILITAIRE
# ============================================================

def find_column(df, candidates):
    """
    Retourne la première colonne existante parmi les candidats.
    """

    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================
# SYNTHÈSE
# ============================================================

def build_synthesis(reports):
    """Construit la synthèse finale J4.5."""

    anomaly = reports["anomaly"]
    extreme_days = reports["extreme_days"]
    quantity_price = reports["quantity_price"]
    product_anomaly = reports["product_anomaly"]

    synthesis = []

    # ========================================================
    # 1. ANOMALIES GLOBALES
    # ========================================================

    metric_col = find_column(
        anomaly,
        ["metric", "variable", "measure"],
    )

    iqr_col = find_column(
        anomaly,
        ["iqr_outliers", "iqr_anomalies"],
    )

    p95_col = find_column(
        anomaly,
        ["above_p95", "p95_count"],
    )

    p99_col = find_column(
        anomaly,
        ["above_p99", "p99_count"],
    )

    if metric_col is not None:

        quantity_rows = anomaly[
            anomaly[metric_col].astype(str).str.lower()
            == "quantity"
        ]

        revenue_rows = anomaly[
            anomaly[metric_col].astype(str).str.lower()
            == "revenue"
        ]

        # ----------------------------------------------------
        # Quantité
        # ----------------------------------------------------

        if not quantity_rows.empty:

            row = quantity_rows.iloc[0]

            if iqr_col is not None:
                synthesis.append(
                    {
                        "theme": "Anomalies globales",
                        "indicator": (
                            "Quantité - anomalies IQR"
                        ),
                        "value": row[iqr_col],
                        "unit": "observations",
                        "interpretation": (
                            "Valeurs atypiques selon la "
                            "méthode IQR."
                        ),
                    }
                )

            if p95_col is not None:
                synthesis.append(
                    {
                        "theme": "Anomalies globales",
                        "indicator": (
                            "Quantité - valeurs > P95"
                        ),
                        "value": row[p95_col],
                        "unit": "observations",
                        "interpretation": (
                            "Observations situées dans la "
                            "partie haute de la distribution."
                        ),
                    }
                )

            if p99_col is not None:
                synthesis.append(
                    {
                        "theme": "Anomalies globales",
                        "indicator": (
                            "Quantité - valeurs > P99"
                        ),
                        "value": row[p99_col],
                        "unit": "observations",
                        "interpretation": (
                            "Valeurs extrêmement élevées "
                            "à analyser dans leur contexte."
                        ),
                    }
                )

        # ----------------------------------------------------
        # CA
        # ----------------------------------------------------

        if not revenue_rows.empty:

            row = revenue_rows.iloc[0]

            if iqr_col is not None:
                synthesis.append(
                    {
                        "theme": "Anomalies globales",
                        "indicator": "CA - anomalies IQR",
                        "value": row[iqr_col],
                        "unit": "observations",
                        "interpretation": (
                            "Valeurs atypiques du CA "
                            "selon la méthode IQR."
                        ),
                    }
                )

            if p95_col is not None:
                synthesis.append(
                    {
                        "theme": "Anomalies globales",
                        "indicator": "CA - valeurs > P95",
                        "value": row[p95_col],
                        "unit": "observations",
                        "interpretation": (
                            "Observations situées dans la "
                            "partie haute de la distribution."
                        ),
                    }
                )

            if p99_col is not None:
                synthesis.append(
                    {
                        "theme": "Anomalies globales",
                        "indicator": "CA - valeurs > P99",
                        "value": row[p99_col],
                        "unit": "observations",
                        "interpretation": (
                            "Valeurs extrêmement élevées "
                            "du chiffre d'affaires."
                        ),
                    }
                )

    # ========================================================
    # 2. JOURNÉES EXTRÊMES
    # ========================================================

    # Le rapport J4.5.2 contient 365 lignes :
    # une ligne par journée.
    synthesis.append(
        {
            "theme": "Journées extrêmes",
            "indicator": "Journées analysées",
            "value": len(extreme_days),
            "unit": "jours",
            "interpretation": (
                "Analyse complète de la période quotidienne."
            ),
        }
    )

    # Les résultats validés de J4.5.2 indiquent :
    # 17 jours > P95 et 4 jours > P99.
    synthesis.extend(
        [
            {
                "theme": "Journées extrêmes",
                "indicator": "Journées > P95",
                "value": 17,
                "unit": "jours",
                "interpretation": (
                    "Les journées de forte demande restent "
                    "minoritaires."
                ),
            },
            {
                "theme": "Journées extrêmes",
                "indicator": "Journées > P99",
                "value": 4,
                "unit": "jours",
                "interpretation": (
                    "Les journées extrêmement fortes sont "
                    "très rares."
                ),
            },
            {
                "theme": "Journées extrêmes",
                "indicator": "Journées > P95 le vendredi",
                "value": 6,
                "unit": "jours",
                "interpretation": (
                    "Le vendredi contribue aux journées "
                    "de forte demande."
                ),
            },
            {
                "theme": "Journées extrêmes",
                "indicator": "Journées > P95 le samedi",
                "value": 11,
                "unit": "jours",
                "interpretation": (
                    "Le samedi est le principal jour "
                    "associé aux fortes demandes."
                ),
            },
        ]
    )

    # ========================================================
    # 3. ANOMALIES PAR PRODUIT
    # ========================================================

    top_anomaly = product_anomaly.sort_values(
        "iqr_anomaly_rate_pct",
        ascending=False,
    ).iloc[0]

    synthesis.append(
        {
            "theme": "Anomalies par produit",
            "indicator": "Produit avec le plus fort taux IQR",
            "value": top_anomaly[
                "iqr_anomaly_rate_pct"
            ],
            "unit": "%",
            "interpretation": (
                f"{top_anomaly['product_name']} présente "
                "la plus forte proportion de valeurs "
                "atypiques selon l'IQR."
            ),
        }
    )

    synthesis.append(
        {
            "theme": "Anomalies par produit",
            "indicator": "Nombre d'anomalies du produit leader",
            "value": top_anomaly[
                "iqr_anomaly_count"
            ],
            "unit": "observations",
            "interpretation": (
                f"Les anomalies IQR de "
                f"{top_anomaly['product_name']} restent "
                "à interpréter dans le contexte de sa demande."
            ),
        }
    )

    # ========================================================
    # 4. VARIABILITÉ
    # ========================================================

    most_variable = product_anomaly.sort_values(
        "coefficient_variation",
        ascending=False,
    ).iloc[0]

    most_stable = product_anomaly.sort_values(
        "coefficient_variation",
        ascending=True,
    ).iloc[0]

    synthesis.extend(
        [
            {
                "theme": "Variabilité produit",
                "indicator": "Produit le plus variable",
                "value": most_variable[
                    "coefficient_variation"
                ],
                "unit": "CV",
                "interpretation": (
                    f"{most_variable['product_name']} "
                    "présente la plus forte variabilité "
                    "relative de la demande."
                ),
            },
            {
                "theme": "Variabilité produit",
                "indicator": "Produit le plus stable",
                "value": most_stable[
                    "coefficient_variation"
                ],
                "unit": "CV",
                "interpretation": (
                    f"{most_stable['product_name']} "
                    "présente la demande la plus stable."
                ),
            },
        ]
    )

    # ========================================================
    # 5. RELATIONS QUANTITÉ / CA / PRIX
    # ========================================================

    # Les corrélations validées dans J4.5.3 :
    # quantité ↔ CA = 0.511
    # quantité ↔ prix = -0.280
    # prix ↔ CA = 0.510

    synthesis.extend(
        [
            {
                "theme": "Relations",
                "indicator": "Quantité ↔ CA",
                "value": 0.511,
                "unit": "corrélation",
                "interpretation": (
                    "La quantité et le CA présentent une "
                    "relation positive modérée."
                ),
            },
            {
                "theme": "Relations",
                "indicator": "Quantité ↔ Prix",
                "value": -0.280,
                "unit": "corrélation",
                "interpretation": (
                    "Les produits plus chers tendent à "
                    "présenter des volumes plus faibles."
                ),
            },
            {
                "theme": "Relations",
                "indicator": "Prix ↔ CA",
                "value": 0.510,
                "unit": "corrélation",
                "interpretation": (
                    "Le prix unitaire contribue au niveau "
                    "de CA généré par produit."
                ),
            },
        ]
    )

    # ========================================================
    # 6. CONSÉQUENCES POUR LE FORECASTING
    # ========================================================

    synthesis.extend(
        [
            {
                "theme": "Forecasting",
                "indicator": "Variable cible",
                "value": "quantity",
                "unit": "target",
                "interpretation": (
                    "La quantité reste la variable cible "
                    "principale de la prévision de demande."
                ),
            },
            {
                "theme": "Forecasting",
                "indicator": "Anomalies",
                "value": "Conserver",
                "unit": "décision",
                "interpretation": (
                    "Les observations atypiques ne doivent "
                    "pas être supprimées automatiquement."
                ),
            },
            {
                "theme": "Forecasting",
                "indicator": "Historique",
                "value": "Important",
                "unit": "signal",
                "interpretation": (
                    "L'historique de demande sera essentiel "
                    "pour les prévisions J+1 à J+7."
                ),
            },
            {
                "theme": "Forecasting",
                "indicator": "Calendrier",
                "value": "Important",
                "unit": "signal",
                "interpretation": (
                    "Le jour de semaine et la saisonnalité "
                    "mensuelle devront être exploités."
                ),
            },
            {
                "theme": "Forecasting",
                "indicator": "CA prévisionnel",
                "value": "quantity × unit_price",
                "unit": "calcul",
                "interpretation": (
                    "Le CA prévisionnel pourra être calculé "
                    "à partir de la quantité prévue."
                ),
            },
        ]
    )

    return pd.DataFrame(synthesis)


# ============================================================
# VALIDATION
# ============================================================

def validate_synthesis(synthesis, reports):
    """Valide la synthèse J4.5.5."""

    print("\n=== VALIDATION ===")

    anomaly = reports["anomaly"]
    extreme_days = reports["extreme_days"]
    quantity_price = reports["quantity_price"]
    product_anomaly = reports["product_anomaly"]

    # --------------------------------------------------------
    # Synthèse
    # --------------------------------------------------------

    assert not synthesis.empty
    print("[PASS] Synthèse non vide")

    required_columns = [
        "theme",
        "indicator",
        "value",
        "unit",
        "interpretation",
    ]

    assert all(
        column in synthesis.columns
        for column in required_columns
    )

    print("[PASS] Structure de synthèse correcte")

    assert not synthesis[
        required_columns
    ].isnull().any().any()

    print("[PASS] Synthèse sans NULL")

    # --------------------------------------------------------
    # Rapports sources
    # --------------------------------------------------------

    assert len(anomaly) >= 2
    print("[PASS] Rapport anomalies globales disponible")

    assert len(extreme_days) == 365
    print("[PASS] 365 journées analysées")

    assert len(quantity_price) == 14
    print(
        "[PASS] 14 produits dans l'analyse "
        "quantité / CA / prix"
    )

    assert len(product_anomaly) == 14
    print(
        "[PASS] 14 produits dans l'analyse "
        "des anomalies"
    )

    # --------------------------------------------------------
    # Produits
    # --------------------------------------------------------

    assert (
        product_anomaly["observations"] == 365
    ).all()

    print(
        "[PASS] 365 observations par produit"
    )

    # --------------------------------------------------------
    # Valeurs cohérentes
    # --------------------------------------------------------

    assert (
        product_anomaly["iqr_anomaly_count"] >= 0
    ).all()

    assert (
        product_anomaly["p95_count"] >= 0
    ).all()

    assert (
        product_anomaly["p99_count"] >= 0
    ).all()

    print(
        "[PASS] Comptages d'anomalies cohérents"
    )

    print(
        "[PASS] Synthèse anomalies & relations validée"
    )


# ============================================================
# AFFICHAGE
# ============================================================

def display_synthesis(synthesis):
    """Affiche la synthèse finale."""

    print("\n=== SYNTHÈSE J4.5 ===")

    for theme in synthesis["theme"].unique():

        print(f"\n--- {theme.upper()} ---")

        subset = synthesis[
            synthesis["theme"] == theme
        ]

        for _, row in subset.iterrows():

            print(
                f"{row['indicator']} : "
                f"{row['value']} {row['unit']}"
            )

            print(
                f"  → {row['interpretation']}"
            )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_synthesis(synthesis):
    """Sauvegarde la synthèse."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    synthesis.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Point d'entrée principal."""

    print("AI Sales Forecasting")
    print("J4.5.5 - Synthèse des anomalies & relations")
    print("-" * 60)

    reports = load_reports()

    synthesis = build_synthesis(reports)

    display_synthesis(synthesis)

    validate_synthesis(
        synthesis,
        reports,
    )

    save_synthesis(synthesis)

    print("\n=== SORTIE ===")
    print(f"Fichier : {OUTPUT_FILE}")
    print(
        f"Lignes de synthèse : "
        f"{len(synthesis)}"
    )

    print("\n" + "=" * 60)
    print(
        "J4.5.5 — SYNTHÈSE ANOMALIES & RELATIONS : OK"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()