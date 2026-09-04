# AI Sales Forecasting — Project Checkpoint

> Dernière mise à jour : 2026-09-04
> Checkpoint : **J7.8**
> Statut global : **J7 — FORECASTING TERMINÉ**

---

# 1. Projet

**Nom :** AI Sales Forecasting

**Objectif :** construire une solution complète de prévision des ventes à partir des données KShop afin de :

* analyser l'historique des ventes ;
* identifier les tendances et comportements de demande ;
* préparer les données pour le Machine Learning ;
* entraîner et comparer plusieurs modèles ;
* sélectionner le meilleur modèle ;
* produire des prévisions J+1 → J+7 ;
* préparer des recommandations de stock ;
* construire un dashboard métier ;
* préparer une architecture exploitable en production.

---

# 2. Architecture du projet

```text
PostgreSQL / KShop
        │
        ▼
Python / Pandas
        │
        ▼
Data Cleaning
        │
        ▼
EDA
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning
        │
        ▼
Forecast J+1 → J+7
        │
        ▼
Stock Recommendation
        │
        ▼
Streamlit Dashboard
        │
        ▼
Production
```

---

# 3. Structure du projet

```text
ai-sales-forecasting/
├── config/
├── dashboard/
├── data/
│   ├── processed/
│   │   ├── eda/
│   │   ├── ml_ready/
│   │   ├── ml_split/
│   │   ├── forecast/
│   │   │   └── visualizations/
│   │   ├── sales_clean.csv
│   │   ├── sales_calendar_features.csv
│   │   ├── sales_lag_features.csv
│   │   ├── sales_rolling_features.csv
│   │   └── sales_ml_ready.csv
│   └── raw/
│       └── kshop_sales_synthetic.csv
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── forecasting/
│   │   ├── forecast_sales.py
│   │   ├── analyze_forecast.py
│   │   ├── visualize_forecast.py
│   │   └── validate_forecast.py
│   ├── models/
│   └── utils/
├── tests/
├── .env.example
├── .gitignore
├── PROJECT_CHECKPOINT.md
├── README.md
├── requirements.txt
└── src/main.py
```

---

# 4. Dataset

Le dataset utilisé pour le projet est un dataset **synthétique**, construit à partir de la structure métier de KShop.

Il ne représente pas de véritables transactions clients.

## Caractéristiques

* Période : **2025-09-01 → 2026-08-31**
* Durée : **365 jours**
* Produits : **14**
* Granularité : **1 ligne par produit et par jour**
* Nombre de lignes : **5 110**
* Variable cible : **quantity**

Fichier :

```text
data/raw/kshop_sales_synthetic.csv
```

Colonnes principales :

```text
date
product_id
product_name
category
quantity
unit_price
revenue
```

---

# 5. J1 — Initialisation

**Statut : ✅ TERMINÉ**

Environnement Python configuré avec Conda.

Version Python :

```text
Python 3.12.14
```

Principales dépendances installées et validées :

```text
numpy
pandas
scikit-learn
matplotlib
plotly
streamlit
SQLAlchemy
psycopg2
python-dotenv
joblib
openpyxl
jupyter
pytest
```

`src/main.py` validé.

---

# 6. J2 — Dataset

**Statut : ✅ TERMINÉ**

Dataset synthétique généré et validé.

Résultats :

```text
Lignes       : 5 110
Produits     : 14
Jours        : 365
Période      : 2025-09-01 → 2026-08-31
```

Contrôles réalisés :

* colonnes obligatoires présentes ;
* dates valides ;
* quantités valides ;
* prix positifs ;
* revenue cohérent avec quantity × unit_price ;
* absence de valeurs incohérentes.

---

# 7. J3 — Data Cleaning

**Statut : ✅ TERMINÉ**

Source :

```text
data/raw/kshop_sales_synthetic.csv
```

Sortie :

```text
data/processed/sales_clean.csv
```

Résultats :

```text
Lignes initiales          : 5 110
Lignes finales            : 5 110
NULL critiques supprimés : 0
Quantités invalides      : 0
Prix invalides            : 0
Doublons supprimés       : 0
```

Principes :

* dates validées ;
* product_id validé ;
* quantity validée ;
* unit_price validé ;
* revenue recalculé ;
* doublons `date + product_id` contrôlés ;
* texte nettoyé ;
* anomalies cohérentes conservées ;
* dataset RAW jamais modifié.

---

# 8. J4 — Exploratory Data Analysis

**Statut : ✅ TERMINÉ**

## Statistiques principales

Quantity :

```text
Total       : 28 076
Moyenne     : 5.49
Médiane     : 5
Écart-type  : 3.70
Minimum     : 0
Maximum     : 26
P95         : 13
P99         : 17
Zéros       : 142
```

Revenue :

```text
Total       : 67 966 700 AR
Moyenne     : 13 300.72 AR
Médiane     : 10 800 AR
Maximum     : 81 000 AR
```

Demande quotidienne :

```text
Moyenne     : 76.92 unités
Médiane     : 75
Minimum     : 36
Maximum     : 152
```

## Principales observations

* Samedi = jour le plus fort.
* Dimanche = jour le plus faible.
* Décembre 2025 = mois le plus fort.
* Février 2026 = mois le plus faible.
* Tendance positive sur l'année.
* Premier 30 jours : **71.17 unités/jour**
* Derniers 30 jours : **82.93 unités/jour**
* Croissance : **+16.53 %**

Produit leader en volume :

```text
Coca-Cola 33cl
```

Produit leader en CA :

```text
Huile alimentaire 1L
```

Les anomalies identifiées sont conservées pour analyse et modélisation.

---

# 9. J5 — Feature Engineering

**Statut : ✅ TERMINÉ**

Dataset final :

```text
data/processed/sales_ml_ready.csv
```

Résultats :

```text
Lignes : 4 690
Colonnes : 18
Produits : 14
Période : 2025-10-01 → 2026-08-31
```

## Features calendrier

```text
day_of_week
day_of_month
month
week_of_year
is_weekend
```

## Features lag

```text
lag_1
lag_7
lag_14
```

## Features rolling

```text
rolling_mean_7
rolling_mean_14
rolling_mean_30
```

Les lags et rolling sont calculés **par product_id**.

Aucune fuite de données n'a été introduite.

La normalisation n'est pas réalisée à cette étape.

---

# 10. J6 — Machine Learning

**Statut : ✅ TERMINÉ**

## J6.1 — Temporal Split

```text
TRAIN
2025-10-01 → 2026-06-30
3 822 lignes

VALIDATION
2026-07-01 → 2026-07-31
434 lignes

TEST
2026-08-01 → 2026-08-31
434 lignes
```

Aucun chevauchement temporel.

---

## J6.2 — Préparation X/y

12 features utilisées :

```text
product_id
day_of_week
day_of_month
month
week_of_year
is_weekend
lag_1
lag_7
lag_14
rolling_mean_7
rolling_mean_14
rolling_mean_30
```

Target :

```text
quantity
```

---

## J6.3 — Baseline

Baseline :

```text
lag_7
```

Validation :

```text
MAE  : 2.7442
RMSE : 3.6604
R²   : 0.0394
```

---

## J6.4 — Modèles entraînés

Modèles :

1. Random Forest
2. Gradient Boosting
3. HistGradientBoosting

Tous entraînés uniquement sur le Train.

---

## J6.5 — Évaluation Validation

| Modèle                |        MAE |       RMSE |         R² |
| --------------------- | ---------: | ---------: | ---------: |
| **Gradient Boosting** | **2.0337** | **2.6249** | **0.5060** |
| Random Forest         |     2.0640 |     2.6599 |     0.4928 |
| HistGradientBoosting  |     2.1287 |     2.7862 |     0.4434 |
| Baseline lag_7        |     2.7442 |     3.6604 |     0.0394 |

Gradient Boosting :

```text
Gain MAE  : +25.89 %
Gain RMSE : +28.29 %
```

---

## J6.6 — Comparaison approfondie

**⏭️ NON EXÉCUTÉE SÉPARÉMENT**

La sélection du modèle a néanmoins été effectuée sur les métriques disponibles.

---

## J6.7 — Sélection

Critères :

1. MAE croissant
2. RMSE croissant
3. R² décroissant

Modèle sélectionné :

```text
Gradient Boosting
```

---

## J6.8 — Modèle final

Fichiers :

```text
models/final_model.joblib
models/final_model_metadata.json
```

Target :

```text
quantity
```

Nombre de features :

```text
12
```

Le modèle final correspond au modèle Gradient Boosting sélectionné en J6.7.

---

## J6.9 — Évaluation finale Test

Test :

```text
2026-08-01 → 2026-08-31
434 lignes
14 produits
```

Résultats du modèle final :

```text
MAE  : 2.0928
RMSE : 2.7149
R²   : 0.5474
```

Baseline :

```text
MAE  : 2.9793
RMSE : 3.9691
R²   : 0.0327
```

Gains :

```text
Gain MAE  : 29.75 %
Gain RMSE : 31.60 %
```

Le Test n'a été utilisé qu'en J6.9 pour l'évaluation finale.

---

# 11. J7 — Forecasting J+1 → J+7

**Statut : ✅ TERMINÉ**

Objectif :

Produire une prévision récursive de la demande pour les 7 jours suivant la dernière date historique.

Dernière date historique :

```text
2026-08-31
```

Horizon :

```text
J+1 : 2026-09-01
J+2 : 2026-09-02
J+3 : 2026-09-03
J+4 : 2026-09-04
J+5 : 2026-09-05
J+6 : 2026-09-06
J+7 : 2026-09-07
```

Produits :

```text
14
```

Nombre total de prévisions :

```text
14 × 7 = 98
```

---

# 12. J7.1 — Forecast Contract

**Statut : ✅ VALIDÉ**

Modèle :

```text
models/final_model.joblib
```

Target :

```text
quantity
```

Features :

```text
product_id
day_of_week
day_of_month
month
week_of_year
is_weekend
lag_1
lag_7
lag_14
rolling_mean_7
rolling_mean_14
rolling_mean_30
```

Règle anti-leakage :

> Les quantités réelles futures ne sont jamais utilisées.

Les prévisions précédentes sont réinjectées récursivement pour calculer les features des jours suivants.

---

# 13. J7.2 — Préparation du moteur

**Statut : ✅ VALIDÉ**

Script :

```text
src/forecasting/forecast_sales.py
```

Contrôles :

* dataset historique disponible ;
* modèle final disponible ;
* interface `predict()` disponible ;
* métadonnées produits disponibles ;
* features compatibles avec le modèle.

---

# 14. J7.3 — Forecast récursif

**Statut : ✅ VALIDÉ**

Fichier généré :

```text
data/processed/forecast/sales_forecast_j1_j7.csv
```

Résultats :

```text
Prévisions générées : 98
Produits             : 14
Jours                : 7
```

Quantité totale prévue :

```text
577.39 unités
```

CA prévisionnel :

```text
1 416 793.77 AR
```

Moyenne par ligne :

```text
5.8918 unités
```

Contrôles :

```text
Aucune quantité future réelle utilisée : PASS
Forecast récursif J+1 → J+7            : PASS
Aucun doublon date + product_id        : PASS
Prévisions non négatives               : PASS
```

---

# 15. J7.4 — Validation du fichier Forecast

**Statut : ✅ VALIDÉ**

Résultats :

```text
Shape              : (98, 8)
Date minimum       : 2026-09-01
Date maximum       : 2026-09-07
Produits           : 14
Jours              : 7
NULL               : 0
Négatifs           : 0
Doublons           : 0
Erreurs CA         : 0
```

Cohérence :

```text
forecast_quantity × unit_price
=
forecast_revenue
```

Validée.

---

# 16. J7.5 — Analyse des prévisions

**Statut : ✅ VALIDÉ**

Fichiers :

```text
data/processed/forecast/forecast_daily_summary.csv
data/processed/forecast/forecast_product_summary.csv
```

## Résultat global

```text
Quantité totale : 577.39
CA total        : 1 416 793.77 AR
```

## Meilleur jour

```text
J+5
2026-09-05
```

Prévision :

```text
Quantité : 101.24
CA       : 247 769.42 AR
```

## Top 5 produits — quantité

```text
1. Coca-Cola 33cl       : 83.23
2. Fanta Orange 33cl    : 62.74
3. Eau Vive 1.5L        : 61.08
4. Riz 1kg              : 52.47
5. Biscuits Chocolat    : 48.76
```

## Top 5 produits — CA

```text
1. Huile alimentaire 1L : 194 427.03 AR
2. Lait en poudre 400g  : 189 618.96 AR
3. Riz 1kg              : 183 659.71 AR
4. Sucre 1kg            : 141 008.82 AR
5. Coca-Cola 33cl       : 124 849.90 AR
```

---

# 17. J7.6 — Visualisation

**Statut : ✅ VALIDÉ**

Script :

```text
src/forecasting/visualize_forecast.py
```

Dossier :

```text
data/processed/forecast/visualizations/
```

Visualisations générées :

```text
forecast_quantity_by_day.png
forecast_revenue_by_day.png
forecast_top_products_quantity.png
forecast_top_products_revenue.png
```

Les quatre fichiers ont été générés et contrôlés comme non vides.

---

# 18. J7.7 — Validation finale

**Statut : ✅ VALIDÉ**

Script :

```text
src/forecasting/validate_forecast.py
```

Contrôles réalisés :

```text
Fichier Forecast présent                 : PASS
Colonnes correctes                       : PASS
98 prévisions                            : PASS
14 produits                              : PASS
7 dates                                  : PASS
Dates J+1 → J+7                          : PASS
Horizons J+1 → J+7                       : PASS
Aucun NULL                               : PASS
Quantités non négatives                  : PASS
Prix unitaires positifs                  : PASS
Aucun doublon                            : PASS
Quantité × prix = CA                     : PASS
Analyse quotidienne présente             : PASS
Analyse produits présente                : PASS
Totaux cohérents                         : PASS
Modèle final présent                     : PASS
Métadonnées présentes                    : PASS
Visualisations présentes                 : PASS
```

Résumé final :

```text
Prévisions       : 98
Produits         : 14
Jours            : 7
Quantité totale  : 577.39
CA prévisionnel  : 1 416 793.77 AR
Période          : 2026-09-01 → 2026-09-07
```

---

# 19. J7.8 — Checkpoint

**Statut : ✅ VALIDÉ**

Le checkpoint est mis à jour pour enregistrer la clôture de J7.

Le projet est maintenant officiellement à :

```text
J7 — FORECASTING : TERMINÉ
```

---

# 20. Artefacts J7

## Scripts

```text
src/forecasting/forecast_sales.py
src/forecasting/analyze_forecast.py
src/forecasting/visualize_forecast.py
src/forecasting/validate_forecast.py
```

## Données Forecast

```text
data/processed/forecast/sales_forecast_j1_j7.csv
data/processed/forecast/forecast_daily_summary.csv
data/processed/forecast/forecast_product_summary.csv
```

## Visualisations

```text
data/processed/forecast/visualizations/
├── forecast_quantity_by_day.png
├── forecast_revenue_by_day.png
├── forecast_top_products_quantity.png
└── forecast_top_products_revenue.png
```

---

# 21. Git

Historique principal :

```text
07fd765 chore: initialize AI sales forecasting project
ff629be chore: complete J2 dataset validation
924bcb7 feat: complete J3 data cleaning
fe779a6 feat: complete J4 EDA
8acac3c feat: complete J5 feature engineering
2dcaa4d feat: complete J6 machine learning
```

Prochain commit :

```text
feat: complete J7 forecasting
```

---

# 22. État global du projet

| Phase                     | Statut |
| ------------------------- | ------ |
| J1 — Initialisation       | ✅      |
| J2 — Dataset              | ✅      |
| J3 — Data Cleaning        | ✅      |
| J4 — EDA                  | ✅      |
| J5 — Feature Engineering  | ✅      |
| J6 — Machine Learning     | ✅      |
| J7 — Forecasting          | ✅      |
| J8 — Stock Recommendation | ⏳      |
| J9 — Streamlit Dashboard  | ⏳      |
| J10 — Production          | ⏳      |

---

# 23. Prochaine étape

## J8 — Stock Recommendation

Objectif :

Transformer les prévisions de demande en recommandations opérationnelles de stock.

Prévu :

```text
Forecast J+1 → J+7
        │
        ▼
Analyse du stock actuel
        │
        ▼
Stock de sécurité
        │
        ▼
Point de commande
        │
        ▼
Quantité recommandée
        │
        ▼
Priorisation des produits
```

Le résultat attendu sera notamment :

```text
product_id
product_name
current_stock
forecast_demand
safety_stock
reorder_point
recommended_order_quantity
priority
```

---

# 24. Statut final du checkpoint

**Checkpoint actuel : J7.8**

**Statut global : J7 — FORECASTING TERMINÉ ✅**

**Prochaine phase : J8 — STOCK RECOMMENDATION**
