"""
AI Sales Forecasting
J4.6 - Synthèse finale de l'EDA

Objectif :
    Consolider les résultats des analyses J4.2 à J4.5
    afin de produire une synthèse finale avant le Feature Engineering.

Entrées principales :
    data/processed/sales_clean.csv
    data/processed/eda/anomaly_detection_report.csv
    data/processed/eda/extreme_days_analysis.csv
    data/processed/eda/quantity_revenue_price_products.csv
    data/processed/eda/product_anomalies_analysis.csv
    data/processed/eda/eda_anomaly_synthesis.csv

Sortie :
    data/processed/eda/eda_final_synthesis.csv
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = BASE_DIR / "data" / "processed" / "sales_clean.csv"
EDA_DIR = BASE_DIR / "data" / "processed" / "eda"

ANOMALY_FILE = EDA_DIR / "anomaly_detection_report.csv"
EXTREME_DAYS_FILE = EDA_DIR / "extreme_days_analysis.csv"
PRICE_FILE = EDA_DIR / "quantity_revenue_price_products.csv"
PRODUCT_ANOMALY_FILE = EDA_DIR / "product_anomalies_analysis.csv"
ANOMALY_SYNTHESIS_FILE = EDA_DIR / "eda_anomaly_synthesis.csv"

OUTPUT_FILE = EDA_DIR / "eda_final_synthesis.csv"


# ============================================================
# UTILITAIRES
# ============================================================

def load_required_file(file_path):
    """Charge un fichier CSV et vérifie son existence."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier requis introuvable : {file_path}"
        )

    return pd.read_csv(file_path)


def format_number(value):
    """Formate un nombre pour l'affichage."""
    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f}".replace(",", " ")


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_data():
    """Charge les données principales et les rapports EDA."""

    sales = load_required_file(DATA_FILE)

    reports = {
        "anomaly": load_required_file(ANOMALY_FILE),
        "extreme_days": load_required_file(EXTREME_DAYS_FILE),
        "price": load_required_file(PRICE_FILE),
        "product_anomaly": load_required_file(PRODUCT_ANOMALY_FILE),
        "anomaly_synthesis": load_required_file(ANOMALY_SYNTHESIS_FILE),
    }

    return sales, reports


# ============================================================
# ANALYSE PRINCIPALE
# ============================================================

def build_final_synthesis(sales, reports):
    """Construit la synthèse finale de l'EDA."""

    sales["date"] = pd.to_datetime(sales["date"])

    # --------------------------------------------------------
    # PROFIL GLOBAL
    # --------------------------------------------------------

    total_rows = len(sales)
    total_quantity = sales["quantity"].sum()
    total_revenue = sales["revenue"].sum()

    start_date = sales["date"].min()
    end_date = sales["date"].max()

    product_count = sales["product_id"].nunique()
    day_count = sales["date"].nunique()

    avg_daily_quantity = (
        sales.groupby("date")["quantity"]
        .sum()
        .mean()
    )

    avg_daily_revenue = (
        sales.groupby("date")["revenue"]
        .sum()
        .mean()
    )

    # --------------------------------------------------------
    # TENDANCE
    # --------------------------------------------------------

    daily_quantity = (
        sales.groupby("date")["quantity"]
        .sum()
        .sort_index()
    )

    daily_revenue = (
        sales.groupby("date")["revenue"]
        .sum()
        .sort_index()
    )

    first_30_quantity = daily_quantity.head(30).mean()
    last_30_quantity = daily_quantity.tail(30).mean()

    first_90_quantity = daily_quantity.head(90).mean()
    last_90_quantity = daily_quantity.tail(90).mean()

    first_30_revenue = daily_revenue.head(30).mean()
    last_30_revenue = daily_revenue.tail(30).mean()

    first_90_revenue = daily_revenue.head(90).mean()
    last_90_revenue = daily_revenue.tail(90).mean()

    quantity_trend_30 = (
        (last_30_quantity - first_30_quantity)
        / first_30_quantity
        * 100
    )

    quantity_trend_90 = (
        (last_90_quantity - first_90_quantity)
        / first_90_quantity
        * 100
    )

    revenue_trend_30 = (
        (last_30_revenue - first_30_revenue)
        / first_30_revenue
        * 100
    )

    revenue_trend_90 = (
        (last_90_revenue - first_90_revenue)
        / first_90_revenue
        * 100
    )

    # --------------------------------------------------------
    # PRODUITS
    # --------------------------------------------------------

    product_summary = (
        sales.groupby(
            ["product_id", "product_name"],
            as_index=False
        )
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum")
        )
    )

    top_volume = product_summary.loc[
        product_summary["total_quantity"].idxmax()
    ]

    top_revenue = product_summary.loc[
        product_summary["total_revenue"].idxmax()
    ]

    # --------------------------------------------------------
    # VARIABILITÉ PRODUITS
    # --------------------------------------------------------

    product_variability = (
        sales.groupby(
            ["product_id", "product_name"]
        )["quantity"]
        .agg(["mean", "std"])
        .reset_index()
    )

    product_variability["cv"] = (
        product_variability["std"]
        / product_variability["mean"]
    )

    product_variability = product_variability[
        product_variability["mean"] > 0
    ]

    most_variable = product_variability.loc[
        product_variability["cv"].idxmax()
    ]

    most_stable = product_variability.loc[
        product_variability["cv"].idxmin()
    ]

    # --------------------------------------------------------
    # JOURNÉES EXTRÊMES
    # --------------------------------------------------------

    extreme_days = reports["extreme_days"]

    quantity_column = None

    possible_quantity_columns = [
        "daily_quantity",
        "total_quantity",
        "quantity"
    ]

    for column in possible_quantity_columns:
        if column in extreme_days.columns:
            quantity_column = column
            break

    if quantity_column is not None:
        p95 = extreme_days[quantity_column].quantile(0.95)
        p99 = extreme_days[quantity_column].quantile(0.99)
    else:
        daily_values = daily_quantity.values
        p95 = pd.Series(daily_values).quantile(0.95)
        p99 = pd.Series(daily_values).quantile(0.99)

    days_above_p95 = int(
        (daily_quantity > p95).sum()
    )

    days_above_p99 = int(
        (daily_quantity > p99).sum()
    )

    # --------------------------------------------------------
    # ANOMALIES PRODUITS
    # --------------------------------------------------------

    product_anomaly = reports["product_anomaly"]

    anomaly_rate_column = "iqr_anomaly_rate_pct"

    if anomaly_rate_column in product_anomaly.columns:
        anomaly_leader = product_anomaly.loc[
            product_anomaly[anomaly_rate_column].idxmax()
        ]
    else:
        anomaly_leader = None

    # --------------------------------------------------------
    # CORRÉLATIONS
    # --------------------------------------------------------

    correlations = sales[
        ["quantity", "unit_price", "revenue"]
    ].corr()

    quantity_revenue_corr = correlations.loc[
        "quantity", "revenue"
    ]

    quantity_price_corr = correlations.loc[
        "quantity", "unit_price"
    ]

    price_revenue_corr = correlations.loc[
        "unit_price", "revenue"
    ]

    # --------------------------------------------------------
    # CONSTRUCTION DE LA SYNTHÈSE
    # --------------------------------------------------------

    synthesis = [
        {
            "section": "Dataset",
            "indicator": "Nombre de lignes",
            "value": total_rows,
            "unit": "observations",
            "conclusion": "Dataset complet pour l'analyse exploratoire."
        },
        {
            "section": "Dataset",
            "indicator": "Nombre de produits",
            "value": product_count,
            "unit": "produits",
            "conclusion": "14 produits sont disponibles pour le forecasting."
        },
        {
            "section": "Dataset",
            "indicator": "Nombre de jours",
            "value": day_count,
            "unit": "jours",
            "conclusion": "Historique quotidien couvrant une année complète."
        },
        {
            "section": "Dataset",
            "indicator": "Période",
            "value": f"{start_date.date()} → {end_date.date()}",
            "unit": "",
            "conclusion": "Période d'analyse validée."
        },
        {
            "section": "Demande",
            "indicator": "Quantité totale",
            "value": total_quantity,
            "unit": "unités",
            "conclusion": "Volume total observé sur la période."
        },
        {
            "section": "Demande",
            "indicator": "Quantité moyenne quotidienne",
            "value": round(avg_daily_quantity, 2),
            "unit": "unités/jour",
            "conclusion": "Niveau moyen de demande quotidienne."
        },
        {
            "section": "CA",
            "indicator": "CA total",
            "value": total_revenue,
            "unit": "AR",
            "conclusion": "Chiffre d'affaires total observé."
        },
        {
            "section": "CA",
            "indicator": "CA moyen quotidien",
            "value": round(avg_daily_revenue, 2),
            "unit": "AR/jour",
            "conclusion": "Niveau moyen de CA quotidien."
        },
        {
            "section": "Tendance",
            "indicator": "Évolution quantité 30 jours",
            "value": round(quantity_trend_30, 2),
            "unit": "%",
            "conclusion": "Signal de tendance récente de la demande."
        },
        {
            "section": "Tendance",
            "indicator": "Évolution quantité 90 jours",
            "value": round(quantity_trend_90, 2),
            "unit": "%",
            "conclusion": "Signal de tendance moyen terme."
        },
        {
            "section": "Tendance",
            "indicator": "Évolution CA 30 jours",
            "value": round(revenue_trend_30, 2),
            "unit": "%",
            "conclusion": "Évolution récente du chiffre d'affaires."
        },
        {
            "section": "Tendance",
            "indicator": "Évolution CA 90 jours",
            "value": round(revenue_trend_90, 2),
            "unit": "%",
            "conclusion": "Évolution moyen terme du chiffre d'affaires."
        },
        {
            "section": "Produits",
            "indicator": "Leader volume",
            "value": top_volume["product_name"],
            "unit": "produit",
            "conclusion": (
                f"{top_volume['product_name']} est le produit "
                f"le plus vendu en volume."
            )
        },
        {
            "section": "Produits",
            "indicator": "Leader CA",
            "value": top_revenue["product_name"],
            "unit": "produit",
            "conclusion": (
                f"{top_revenue['product_name']} génère le plus "
                f"de chiffre d'affaires."
            )
        },
        {
            "section": "Produits",
            "indicator": "Produit le plus variable",
            "value": most_variable["product_name"],
            "unit": f"CV={most_variable['cv']:.2f}",
            "conclusion": (
                "Produit présentant la plus forte variabilité "
                "relative de la demande."
            )
        },
        {
            "section": "Produits",
            "indicator": "Produit le plus stable",
            "value": most_stable["product_name"],
            "unit": f"CV={most_stable['cv']:.2f}",
            "conclusion": (
                "Produit présentant la demande la plus régulière."
            )
        },
        {
            "section": "Anomalies",
            "indicator": "Jours > P95",
            "value": days_above_p95,
            "unit": "jours",
            "conclusion": (
                "Les journées de forte demande restent minoritaires."
            )
        },
        {
            "section": "Anomalies",
            "indicator": "Jours > P99",
            "value": days_above_p99,
            "unit": "jours",
            "conclusion": (
                "Les journées extrêmement fortes sont rares."
            )
        },
        {
            "section": "Anomalies",
            "indicator": "Produit avec plus fort taux IQR",
            "value": (
                anomaly_leader["product_name"]
                if anomaly_leader is not None
                else "N/A"
            ),
            "unit": (
                f"{anomaly_leader[anomaly_rate_column]:.2f}%"
                if anomaly_leader is not None
                else ""
            ),
            "conclusion": (
                "Les valeurs atypiques doivent être interprétées "
                "dans le contexte du produit."
            )
        },
        {
            "section": "Relations",
            "indicator": "Quantité ↔ CA",
            "value": round(quantity_revenue_corr, 3),
            "unit": "corrélation",
            "conclusion": "Relation positive modérée."
        },
        {
            "section": "Relations",
            "indicator": "Quantité ↔ Prix",
            "value": round(quantity_price_corr, 3),
            "unit": "corrélation",
            "conclusion": (
                "Les produits plus chers tendent à avoir des volumes "
                "plus faibles dans ce dataset."
            )
        },
        {
            "section": "Relations",
            "indicator": "Prix ↔ CA",
            "value": round(price_revenue_corr, 3),
            "unit": "corrélation",
            "conclusion": (
                "Le prix unitaire contribue au niveau de CA généré."
            )
        },
        {
            "section": "Forecasting",
            "indicator": "Variable cible",
            "value": "quantity",
            "unit": "target",
            "conclusion": (
                "La quantité sera la cible principale du modèle."
            )
        },
        {
            "section": "Forecasting",
            "indicator": "Historique",
            "value": "À conserver",
            "unit": "",
            "conclusion": (
                "Les historiques de demande seront utilisés pour "
                "les variables retardées et les moyennes mobiles."
            )
        },
        {
            "section": "Forecasting",
            "indicator": "Calendrier",
            "value": "À exploiter",
            "unit": "",
            "conclusion": (
                "Jour de semaine, mois et autres variables calendaires "
                "seront intégrés au Feature Engineering."
            )
        },
        {
            "section": "Forecasting",
            "indicator": "Anomalies",
            "value": "Conserver",
            "unit": "",
            "conclusion": (
                "Aucune suppression automatique des observations "
                "atypiques."
            )
        },
        {
            "section": "Forecasting",
            "indicator": "CA prévisionnel",
            "value": "quantity × unit_price",
            "unit": "",
            "conclusion": (
                "Le CA pourra être calculé après la prévision de quantité."
            )
        },
        {
            "section": "Limites",
            "indicator": "Nature des données",
            "value": "Synthétique",
            "unit": "",
            "conclusion": (
                "Les données ne représentent pas des transactions "
                "clients réelles."
            )
        },
        {
            "section": "Limites",
            "indicator": "Variables externes",
            "value": "Non disponibles",
            "unit": "",
            "conclusion": (
                "Promotions, ruptures de stock, météo et événements "
                "externes ne sont pas intégrés."
            )
        },
        {
            "section": "Limites",
            "indicator": "Catégorisation",
            "value": "À surveiller",
            "unit": "",
            "conclusion": (
                "La coexistence des catégories Alimentaire et "
                "Produits alimentaires doit être documentée."
            )
        },
        {
            "section": "Décision",
            "indicator": "EDA",
            "value": "PRÊTE POUR J5",
            "unit": "",
            "conclusion": (
                "Les résultats J4 sont suffisamment structurés "
                "pour commencer le Feature Engineering."
            )
        },
    ]

    return pd.DataFrame(synthesis)


# ============================================================
# VALIDATION
# ============================================================

def validate_final_synthesis(synthesis, sales):
    """Valide le rapport final."""

    print("\n=== VALIDATION J4.6 ===")

    checks = []

    checks.append(
        ("Synthèse non vide", not synthesis.empty)
    )

    checks.append(
        (
            "Colonnes obligatoires",
            all(
                column in synthesis.columns
                for column in [
                    "section",
                    "indicator",
                    "value",
                    "unit",
                    "conclusion",
                ]
            ),
        )
    )

    checks.append(
        (
            "Synthèse sans NULL critique",
            not synthesis[
                ["section", "indicator", "conclusion"]
            ].isnull().any().any(),
        )
    )

    checks.append(
        (
            "Dataset source non vide",
            not sales.empty,
        )
    )

    checks.append(
        (
            "14 produits présents",
            sales["product_id"].nunique() == 14,
        )
    )

    checks.append(
        (
            "365 jours présents",
            sales["date"].nunique() == 365,
        )
    )

    checks.append(
        (
            "Target quantity présente",
            "quantity" in sales.columns,
        )
    )

    all_pass = True

    for label, result in checks:
        status = "PASS" if result else "FAIL"

        print(f"[{status}] {label}")

        if not result:
            all_pass = False

    if not all_pass:
        raise ValueError(
            "La validation finale J4.6 a échoué."
        )

    print("\n[PASS] Synthèse finale EDA validée")

    return True


# ============================================================
# SAUVEGARDE
# ============================================================

def save_synthesis(synthesis):
    """Sauvegarde le rapport final."""

    EDA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    synthesis.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== SORTIE ===")
    print(f"Fichier : {OUTPUT_FILE}")
    print(f"Lignes  : {len(synthesis)}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Point d'entrée principal."""

    print("=" * 60)
    print("AI Sales Forecasting")
    print("J4.6 - Synthèse finale de l'EDA")
    print("=" * 60)

    sales, reports = load_data()

    print("\n=== DONNÉES CHARGÉES ===")
    print(f"Lignes source : {len(sales):,}")
    print(
        f"Période : "
        f"{pd.to_datetime(sales['date']).min().date()} "
        f"→ "
        f"{pd.to_datetime(sales['date']).max().date()}"
    )
    print(
        f"Produits : {sales['product_id'].nunique()}"
    )

    synthesis = build_final_synthesis(
        sales,
        reports
    )

    print("\n=== SYNTHÈSE FINALE ===")

    for _, row in synthesis.iterrows():
        print(
            f"[{row['section']}] "
            f"{row['indicator']} : "
            f"{row['value']} "
            f"{row['unit']}"
        )

    validate_final_synthesis(
        synthesis,
        sales
    )

    save_synthesis(synthesis)

    print("\n" + "=" * 60)
    print("J4.6 — SYNTHÈSE FINALE EDA : OK")
    print("=" * 60)


if __name__ == "__main__":
    main()