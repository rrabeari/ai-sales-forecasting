# AI Sales Forecasting — Project Checkpoint

> Dernière mise à jour : 2026-09-03
> Checkpoint : **J4.6**
> Statut global : **J4 — EDA VALIDÉ**

---

# 1. Projet

**Nom :** AI Sales Forecasting

**Objectif :** construire un système de prévision des ventes capable de prévoir la demande produit à **J+1 → J+7**, puis de proposer des recommandations de stock et de commande.

**Architecture cible :**

```text
KShop / PostgreSQL
        ↓
Python / Pandas
        ↓
Data Cleaning
        ↓
EDA
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Forecast J+1 → J+7
        ↓
Stock Recommendation
        ↓
Streamlit Dashboard
```

---

# 2. Stack technique

* Python 3.12.14
* Pandas 3.0.5
* NumPy 2.5.2
* Scikit-learn 1.9.0
* Matplotlib 3.11.0
* Plotly 6.9.0
* Streamlit 1.63.0
* SQLAlchemy 2.0.51
* PostgreSQL
* psycopg2
* python-dotenv
* Joblib 1.5.3
* OpenPyXL 3.1.5
* Jupyter
* Pytest 9.0.3
* Git / GitHub

---

# 3. Dataset

**Source métier :** structure/catalogue KShop

**Nature :** dataset synthétique professionnel généré pour le projet.

> Les données utilisées pour le forecasting sont synthétiques et ne représentent pas des transactions clients réelles.

**Dataset RAW :**

```text
data/raw/kshop_sales_synthetic.csv
```

**Dataset CLEAN :**

```text
data/processed/sales_clean.csv
```

### Caractéristiques

* Période : 2025-09-01 → 2026-08-31
* 365 jours
* 14 produits
* 5 110 observations
* granularité : produit × jour
* 7 colonnes :

  * date
  * product_id
  * product_name
  * category
  * quantity
  * unit_price
  * revenue

### Résultats globaux

* Quantité totale : **28 076**
* CA total : **67 966 700 AR**
* Quantité moyenne par jour : **76,92**
* CA moyen par jour : **186 210,14 AR**
* Quantité maximale par observation : **26**
* CA maximal par observation : **81 000 AR**
* NULL : **0**
* Doublons date + produit : **0**

---

# 4. J1 — INITIALISATION

**Statut : ✅ VALIDÉ**

## J1.1 — Structure du projet

**✅ VALIDÉ**

Structure principale :

```text
ai-sales-forecasting/
├── config/
├── dashboard/
├── data/
│   ├── processed/
│   │   ├── eda/
│   │   └── sales_clean.csv
│   └── raw/
│       └── kshop_sales_synthetic.csv
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── forecasting/
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

## J1.2 — Environnement Python

**✅ VALIDÉ**

Environnement :

```text
ai-sales-forecasting
```

Python :

```text
3.12.14
```

Les dépendances principales ont été installées et vérifiées.

## J1.3 — Git / GitHub

**✅ VALIDÉ**

Branche :

```text
main
```

Repository GitHub :

```text
rrabeari/ai-sales-forecasting
```

---

# 5. J2 — DATASET

**Statut : ✅ VALIDÉ**

## J2.1 — Définition des règles dataset

**✅ VALIDÉ**

## J2.2 — Générateur Python

```text
src/data/generate_synthetic_sales.py
```

**✅ VALIDÉ**

Random seed :

```text
42
```

## J2.3 — Génération dataset

**✅ VALIDÉ**

Résultat :

```text
5 110 lignes
14 produits
365 jours
```

## J2.4 — Quality Control

**✅ PASS**

* NULL : 0
* doublons : 0
* quantité négative : 0
* prix invalides : 0
* CA incohérent : 0
* période correcte
* 14 produits
* 365 jours

## J2.5 — Analyse statistique

**✅ VALIDÉ**

Quantité totale :

```text
28 076
```

CA total :

```text
67 966 700 AR
```

Produit #1 en volume :

```text
Coca-Cola 33cl
3 995 unités
```

Produit #1 en CA :

```text
Huile alimentaire 1L
9 190 500 AR
```

## J2.6 — Validation finale

**✅ PASS**

> **J2 — DATASET : VALIDÉ**

Commit :

```text
ff629be chore: complete J2 dataset validation
```

---

# 6. J3 — DATA CLEANING

**Statut : ✅ VALIDÉ**

## J3.1 — Définition des règles

**✅ VALIDÉ**

Règles principales :

* validation des dates
* validation de `product_id`
* quantité entière ≥ 0
* prix > 0
* recalcul du CA
* suppression des doublons date + produit
* nettoyage des espaces texte
* gestion des NULL critiques
* conservation des valeurs extrêmes cohérentes
* RAW jamais modifié

## J3.2 — Script de nettoyage

```text
src/data/clean_sales.py
```

**✅ PASS**

Résultat :

```text
Lignes initiales       : 5 110
Lignes finales         : 5 110
NULL critiques        : 0
Quantités invalides   : 0
Prix invalides        : 0
Doublons supprimés    : 0
```

## J3.3 — Exécution

**✅ VALIDÉ**

Sortie :

```text
data/processed/sales_clean.csv
```

## J3.4 — Quality Control

```text
src/data/quality_check_clean_sales.py
```

**✅ PASS**

Résultats :

```text
5 110 lignes
14 produits
365 jours
28 076 unités
67 966 700 AR
```

## J3.5 — RAW vs CLEAN

```text
src/data/compare_raw_clean.py
```

**✅ VALIDÉ**

* mêmes lignes
* mêmes colonnes
* mêmes clés
* même quantité totale
* même CA total
* mêmes produits
* même période
* 0 différence de contenu

## J3.6 — Validation finale

**✅ VALIDÉ**

> **J3 — DATA CLEANING : VALIDÉ**

Commit :

```text
924bcb7 feat: complete J3 data cleaning
```

Push GitHub :

```text
main → origin/main
```

---

# 7. J4 — EXPLORATORY DATA ANALYSIS

**Statut : ✅ VALIDÉ**

---

## J4.1 — Objectifs EDA

**✅ VALIDÉ**

Objectifs :

1. analyser l'évolution des ventes ;
2. identifier les produits leaders ;
3. analyser les catégories ;
4. analyser les jours de semaine ;
5. identifier la saisonnalité ;
6. identifier les produits à faible / moyen / fort volume ;
7. détecter les valeurs atypiques ;
8. identifier les signaux utiles au forecasting.

---

# 8. J4.2 — Analyse descriptive

**Statut : ✅ VALIDÉ**

Script :

```text
src/data/eda_descriptive.py
```

### Quantité

* moyenne : **5,49**
* médiane : **5**
* écart-type : **3,70**
* minimum : **0**
* maximum : **26**
* P95 : **13**
* P99 : **17**
* observations à zéro : **142**
* taux zéro : **2,78 %**

### CA

* moyenne : **13 300,72 AR**
* médiane : **10 800 AR**
* minimum : **0 AR**
* maximum : **81 000 AR**
* P95 : **33 000 AR**
* P99 : **49 500 AR**

### Produits

Leader volume :

```text
Coca-Cola 33cl
3 995 unités
```

Leader CA :

```text
Huile alimentaire 1L
9 190 500 AR
```

### Catégories

Moteur de volume :

```text
Boissons
```

Moteur financier :

```text
Produits alimentaires
```

---

# 9. J4.3 — Analyse temporelle

**Statut : ✅ VALIDÉ**

Scripts :

```text
src/data/eda_temporal_daily.py
src/data/eda_temporal_monthly.py
src/data/eda_temporal_trend.py
src/data/eda_temporal_weekday.py
```

### Analyse quotidienne

* moyenne : **76,92 unités/jour**
* médiane : **75**
* minimum : **36**
* maximum : **152**

Journée maximale :

```text
2025-12-26
152 unités
372 200 AR
```

### Analyse hebdomadaire

Jour le plus fort :

```text
Samedi
```

Jour le plus faible :

```text
Dimanche
```

### Analyse mensuelle

Mois le plus fort :

```text
Décembre 2025
```

Mois le plus faible :

```text
Février 2026
```

### Tendance

30 premiers jours :

```text
71,17 unités/jour
```

30 derniers jours :

```text
82,93 unités/jour
```

Évolution :

```text
+16,53 %
```

90 jours :

```text
+8,24 %
```

Conclusion :

> La demande présente un signal de tendance récente à la hausse ainsi qu'une saisonnalité hebdomadaire et mensuelle.

---

# 10. J4.4 — Produits & catégories

**Statut : ✅ VALIDÉ**

## J4.4.1 — Profil produits

```text
src/data/eda_product_profile.py
```

**✅ PASS**

Produit le plus stable :

```text
Coca-Cola 33cl
CV = 0,40
```

Produit le plus variable :

```text
Produit 001
CV = 0,74
```

## J4.4.2 — Performance produits

```text
src/data/eda_product_performance.py
```

**✅ PASS**

Répartition :

```text
Volume élevé + CA élevé : 4
Volume élevé + CA faible : 3
Volume faible + CA élevé : 3
Volume faible + CA faible : 4
```

Produits à faible volume mais fort CA :

* Huile alimentaire 1L
* Lait en poudre 400g
* Jus de fruit 1L

## J4.4.3 — Relations catégories / produits

```text
src/data/eda_category_product_relationship.py
```

**✅ PASS**

Analyse des contributions et de la concentration réalisée.

Point à surveiller :

```text
Alimentaire
Produits alimentaires
```

Ces deux catégories existent dans le dataset et leur distinction devra être documentée.

## J4.4.4 — Visualisations

```text
src/data/eda_product_visualizations.py
```

**✅ PASS**

Visualisations :

```text
data/processed/eda/top_products_quantity.png
data/processed/eda/top_products_revenue.png
data/processed/eda/product_volume_vs_revenue.png
data/processed/eda/category_product_contribution.png
data/processed/eda/product_demand_variability.png
```

## J4.4.5 — Synthèse

**✅ VALIDÉ**

Conclusion :

> Les boissons constituent le principal moteur de volume tandis que les produits alimentaires constituent le principal moteur financier.

---

# 11. J4.5 — Anomalies & Relations

**Statut : ✅ VALIDÉ**

## J4.5.1 — Détection des valeurs atypiques

```text
src/data/eda_anomaly_detection.py
```

**✅ PASS**

Quantité :

```text
IQR : 191
P95 : 191
P99 : 42
```

CA :

```text
IQR : 174
P95 : 249
P99 : 42
```

Conclusion :

> Les valeurs élevées ne sont pas automatiquement considérées comme des erreurs.

Aucune suppression automatique.

---

## J4.5.2 — Journées extrêmes

```text
src/data/eda_extreme_days.py
```

**✅ PASS**

```text
6 jours > borne IQR
17 jours > P95
4 jours > P99
```

Répartition des jours > P95 :

```text
Vendredi : 6
Samedi   : 11
```

Conclusion :

> Les journées extrêmes sont rares et principalement associées aux vendredis et samedis.

---

## J4.5.3 — Quantité / CA / Prix

```text
src/data/eda_quantity_revenue_price.py
```

**✅ PASS**

Validation :

```text
CA = quantité × prix
Écart maximum = 0 AR
```

Corrélations :

```text
Quantité ↔ CA   :  0,511
Quantité ↔ Prix : -0,280
Prix ↔ CA       :  0,510
```

Conclusion :

> La quantité est la variable cible principale du forecasting.

CA prévisionnel :

```text
CA prévisionnel = quantité prévue × prix unitaire
```

---

## J4.5.4 — Anomalies par produit

```text
src/data/eda_product_anomalies.py
```

**✅ PASS**

Produit avec le plus fort taux IQR :

```text
Biscuits Chocolat
4,38 %
```

Produit le plus variable :

```text
Produit 001
CV = 0,74
```

Produit le plus stable :

```text
Coca-Cola 33cl
CV = 0,40
```

---

## J4.5.5 — Synthèse anomalies & relations

```text
src/data/eda_anomaly_synthesis.py
```

**✅ PASS**

Principales conclusions :

* anomalies extrêmes rares ;
* forte demande principalement vendredi/samedi ;
* variabilité différente selon les produits ;
* `Produit 001` est le plus variable ;
* `Coca-Cola 33cl` est le plus stable ;
* anomalies conservées ;
* target = `quantity` ;
* historique et calendrier importants pour le forecasting.

> **J4.5 — ANOMALIES & RELATIONS : VALIDÉ**

---

# 12. J4.6 — Synthèse finale EDA

**Statut : ✅ VALIDÉ**

Script :

```text
src/data/eda_final_synthesis.py
```

Rapport :

```text
data/processed/eda/eda_final_synthesis.csv
```

### Validation

```text
[PASS] Synthèse non vide
[PASS] Colonnes obligatoires
[PASS] Synthèse sans NULL critique
[PASS] Dataset source non vide
[PASS] 14 produits présents
[PASS] 365 jours présents
[PASS] Target quantity présente
[PASS] Synthèse finale EDA validée
```

### Synthèse finale

#### Demande

```text
Quantité totale : 28 076
Moyenne/jour    : 76,92
```

#### CA

```text
CA total        : 67 966 700 AR
Moyenne/jour    : 186 210,14 AR
```

#### Tendance

```text
Quantité 30 jours : +16,53 %
Quantité 90 jours : +8,24 %
CA 30 jours       : +13,76 %
CA 90 jours       : +5,76 %
```

#### Produits

```text
Leader volume : Coca-Cola 33cl
Leader CA     : Huile alimentaire 1L

Plus stable   : Coca-Cola 33cl
Plus variable : Produit 001
```

#### Anomalies

```text
Jours > P95 : 17
Jours > P99 : 4
```

#### Relations

```text
Quantité ↔ CA   :  0,511
Quantité ↔ Prix : -0,280
Prix ↔ CA       :  0,510
```

---

# 13. Décisions finales de l'EDA

## Variable cible

```text
quantity
```

La prévision portera sur la demande en quantité.

## Signaux à exploiter dans J5

### Historique

```text
lag_1
lag_7
lag_14
```

### Moyennes mobiles

```text
rolling_mean_7
rolling_mean_14
rolling_mean_30
```

### Calendrier

```text
day_of_week
day_of_month
month
week_of_year
is_weekend
```

### Produit

```text
product_id
product_name
category
unit_price
```

### Anomalies

Les anomalies seront **conservées** et non supprimées automatiquement.

### CA

Le CA prévisionnel sera calculé après prévision :

```text
forecast_revenue = forecast_quantity × unit_price
```

---

# 14. Limites identifiées

Le dataset est adapté au développement du pipeline mais présente des limites.

### Données synthétiques

Les données ne représentent pas des transactions clients réelles.

### Variables externes absentes

Le dataset ne contient actuellement pas :

* promotions ;
* ruptures de stock ;
* météo ;
* événements ;
* jours fériés ;
* campagnes commerciales ;
* concurrence ;
* prix historiques variables.

### Catégorisation

La coexistence de :

```text
Alimentaire
Produits alimentaires
```

reste à documenter.

### Implication

Les résultats du modèle devront être présentés comme un **prototype de forecasting basé sur des données synthétiques**, et non comme une prévision de ventes réelles de KShop.

---

# 15. Fichiers EDA

## Scripts

```text
src/data/eda_descriptive.py
src/data/eda_temporal_daily.py
src/data/eda_temporal_monthly.py
src/data/eda_temporal_trend.py
src/data/eda_temporal_weekday.py
src/data/eda_product_profile.py
src/data/eda_product_performance.py
src/data/eda_category_product_relationship.py
src/data/eda_product_visualizations.py
src/data/eda_anomaly_detection.py
src/data/eda_extreme_days.py
src/data/eda_quantity_revenue_price.py
src/data/eda_product_anomalies.py
src/data/eda_anomaly_synthesis.py
src/data/eda_final_synthesis.py
```

## Rapports

```text
data/processed/eda/anomaly_detection_report.csv
data/processed/eda/extreme_days_analysis.csv
data/processed/eda/quantity_revenue_price_products.csv
data/processed/eda/product_anomalies_analysis.csv
data/processed/eda/eda_anomaly_synthesis.csv
data/processed/eda/eda_final_synthesis.csv
```

## Visualisations

```text
data/processed/eda/top_products_quantity.png
data/processed/eda/top_products_revenue.png
data/processed/eda/product_volume_vs_revenue.png
data/processed/eda/category_product_contribution.png
data/processed/eda/product_demand_variability.png
```

---

# 16. Git — Historique

### J1

Repository initialisé.

### J2

```text
ff629be chore: complete J2 dataset validation
```

### J3

```text
924bcb7 feat: complete J3 data cleaning
```

### J4.5

```text
65caa49 feat: complete J4.5 anomaly and relationship analysis
```

Push GitHub confirmé.

État avant J4.6 :

```text
main = origin/main
working tree = clean
```

---

# 17. État global du projet

| Étape                          | Statut        |
| ------------------------------ | ------------- |
| J1 — Initialisation            | ✅ VALIDÉ      |
| J2 — Dataset                   | ✅ VALIDÉ      |
| J3 — Data Cleaning             | ✅ VALIDÉ      |
| J4.1 — Objectifs EDA           | ✅ VALIDÉ      |
| J4.2 — Analyse descriptive     | ✅ VALIDÉ      |
| J4.3 — Analyse temporelle      | ✅ VALIDÉ      |
| J4.4 — Produits & catégories   | ✅ VALIDÉ      |
| J4.5.1 — Anomalies globales    | ✅ VALIDÉ      |
| J4.5.2 — Journées extrêmes     | ✅ VALIDÉ      |
| J4.5.3 — Quantité / CA / Prix  | ✅ VALIDÉ      |
| J4.5.4 — Anomalies par produit | ✅ VALIDÉ      |
| J4.5.5 — Synthèse anomalies    | ✅ VALIDÉ      |
| **J4.6 — Synthèse finale EDA** | **✅ VALIDÉ**  |
| **J4 — EDA**                   | **✅ TERMINÉ** |
| J5 — Feature Engineering       | ⏳ À FAIRE     |

---

# 18. Checkpoint actuel

**Projet :**

```text
AI Sales Forecasting
```

**Checkpoint :**

```text
J4.6
```

**Statut global :**

```text
J4 — EDA VALIDÉ
```

**Dernière étape validée :**

```text
J4.6 — Synthèse finale EDA
```

**Prochaine étape :**

```text
J5 — FEATURE ENGINEERING
```

---

# 19. Transition vers J5

J4 ayant été entièrement validé, le projet peut maintenant passer à la préparation des variables destinées au Machine Learning.

**J5 aura pour objectif de transformer l'historique nettoyé en dataset supervisé pour le forecasting J+1 → J+7.**

Les premières variables prévues sont :

```text
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

> **J4 — EDA : VALIDÉ ET TERMINÉ**

> **Prochaine étape : J5 — FEATURE ENGINEERING**
