# AI Sales Forecasting — Project Checkpoint

> Dernière mise à jour : 2026-09-03
> Checkpoint : **J4.5.5**
> Statut global : **J4 — EDA EN COURS**

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

**Fichier RAW :**

```text
data/raw/kshop_sales_synthetic.csv
```

**Fichier CLEAN :**

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
* Quantité maximale : **26**
* CA maximal par observation : **81 000 AR**
* Valeurs NULL : **0**
* Doublons date + produit : **0**

---

# 4. J1 — INITIALISATION

## J1.1 — Structure du projet

**Statut : ✅ VALIDÉ**

Structure créée :

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

**Statut : ✅ VALIDÉ**

Environnement Conda :

```text
ai-sales-forecasting
```

Python :

```text
3.12.14
```

Toutes les dépendances principales ont été vérifiées.

## J1.3 — Git

**Statut : ✅ VALIDÉ**

Repository Git initialisé.

Branche :

```text
main
```

Remote :

```text
origin
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

Fichier :

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

Contrôles :

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

Règles :

* validation des dates
* validation product_id
* quantité entière ≥ 0
* prix > 0
* recalcul du CA
* suppression des doublons date + produit
* nettoyage des espaces texte
* gestion explicite des NULL critiques
* conservation des valeurs extrêmes cohérentes
* fichier RAW jamais modifié

## J3.2 — Script de nettoyage

Fichier :

```text
src/data/clean_sales.py
```

**✅ PASS**

Résultat :

```text
Lignes initiales : 5 110
Lignes finales   : 5 110
NULL supprimés   : 0
Quantités invalides : 0
Prix invalides   : 0
Doublons supprimés : 0
```

## J3.3 — Exécution du nettoyage

**✅ VALIDÉ**

Fichier généré :

```text
data/processed/sales_clean.csv
```

## J3.4 — Quality Control

Fichier :

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

Fichier :

```text
src/data/compare_raw_clean.py
```

**✅ VALIDÉ**

Résultats :

* même nombre de lignes
* mêmes colonnes
* mêmes clés date + produit
* même quantité totale
* même CA total
* mêmes produits
* même période
* 0 différence de contenu

## J3.6 — Validation finale

**✅ VALIDÉ**

> **J3 — DATA CLEANING : VALIDÉ**

Commit Git :

```text
924bcb7 feat: complete J3 data cleaning
```

Push GitHub :

```text
main → origin/main
```

Working tree :

```text
clean
```

---

# 7. J4 — EXPLORATORY DATA ANALYSIS

**Statut : 🟡 EN COURS**

---

## J4.1 — Objectifs EDA

**✅ VALIDÉ**

Objectifs :

1. analyser l'évolution des ventes
2. identifier les produits leaders
3. analyser les catégories
4. analyser les jours de semaine
5. identifier la saisonnalité
6. identifier les produits à faible / moyen / fort volume
7. détecter les valeurs atypiques
8. identifier les signaux utiles au forecasting

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
* jours à zéro : **142 observations**
* taux zéro : **2,78 %**

### CA

* moyenne : **13 300,72 AR**
* médiane : **10 800 AR**
* minimum : **0 AR**
* maximum : **81 000 AR**
* P95 : **33 000 AR**
* P99 : **49 500 AR**

### Produits

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

Scripts principaux :

```text
src/data/eda_descriptive.py
```

### Analyse quotidienne

* moyenne : **76,92 unités/jour**
* médiane : **75**
* minimum : **36**
* maximum : **152**

Jour maximal :

```text
2025-12-26
152 unités
372 200 AR
```

### Analyse hebdomadaire

Jour le plus fort :

```text
Samedi
7,03 unités / observation
```

Jour le plus faible :

```text
Dimanche
4,17 unités / observation
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

Moyenne 30 premiers jours :

```text
71,17 unités
```

Moyenne 30 derniers jours :

```text
82,93 unités
```

Évolution :

```text
+16,53 %
```

Signal récent :

> tendance globalement croissante de la demande.

---

# 10. J4.4 — Produits & catégories

**Statut : ✅ VALIDÉ**

## J4.4.1 — Profil produits

Script :

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

Script :

```text
src/data/eda_product_performance.py
```

**✅ PASS**

Profils :

```text
Volume élevé + CA élevé : 4
Volume élevé + CA faible : 3
Volume faible + CA élevé : 3
Volume faible + CA faible : 4
```

Produits à fort volume et fort CA :

* Coca-Cola 33cl
* Riz 1kg
* Sucre 1kg
* Biscuits Chocolat

Produits à faible volume mais fort CA :

* Huile alimentaire 1L
* Lait en poudre 400g
* Jus de fruit 1L

## J4.4.3 — Relations catégories / produits

Script :

```text
src/data/eda_category_product_relationship.py
```

**✅ PASS**

Analyse de contribution et HHI réalisée.

Point à surveiller :

> Le dataset contient à la fois `Alimentaire` et `Produits alimentaires`.

Cette différence est conservée pour analyse ultérieure et n'est pas corrigée pendant l'EDA.

## J4.4.4 — Visualisations produits

Script :

```text
src/data/eda_product_visualizations.py
```

**✅ PASS**

Visualisations générées :

```text
data/processed/eda/top_products_quantity.png
data/processed/eda/top_products_revenue.png
data/processed/eda/product_volume_vs_revenue.png
data/processed/eda/category_product_contribution.png
data/processed/eda/product_demand_variability.png
```

## J4.4.5 — Synthèse produits & catégories

**✅ VALIDÉ**

Conclusion :

> Les boissons constituent le principal moteur de volume tandis que les produits alimentaires constituent le principal moteur financier.

---

# 11. J4.5 — Anomalies & Relations

**Statut : ✅ VALIDÉ**

---

## J4.5.1 — Détection des valeurs atypiques

Script :

```text
src/data/eda_anomaly_detection.py
```

**✅ PASS**

### Quantité

```text
IQR : 191 observations
P95 : 191 observations
P99 : 42 observations
```

### CA

```text
IQR : 174 observations
P95 : 249 observations
P99 : 42 observations
```

Conclusion :

> Les valeurs élevées ne sont pas automatiquement considérées comme des erreurs.

Aucune suppression automatique.

Rapport :

```text
data/processed/eda/anomaly_detection_report.csv
```

---

## J4.5.2 — Analyse des journées extrêmes

Script :

```text
src/data/eda_extreme_days.py
```

**✅ PASS**

### Références

```text
Moyenne : 76,92
Médiane : 75
P95 : 109
P99 : 124,36
```

### Journées extrêmes

```text
6 jours > borne IQR
17 jours > P95
4 jours > P99
```

Les 17 journées > P95 représentent :

```text
4,66 % des journées
```

Répartition :

```text
Vendredi : 6
Samedi   : 11
```

Conclusion :

> Les journées extrêmes sont rares et fortement associées aux vendredis et samedis.

Rapport :

```text
data/processed/eda/extreme_days_analysis.csv
```

---

## J4.5.3 — Quantité / CA / Prix

Script :

```text
src/data/eda_quantity_revenue_price.py
```

**✅ PASS**

Validation :

```text
CA = quantité × prix
Écart maximum = 0 AR
```

### Corrélations globales

```text
Quantité ↔ CA    :  0,511
Quantité ↔ Prix  : -0,280
Prix ↔ CA        :  0,510
```

### Corrélations au niveau produit

```text
Prix ↔ CA        :  0,781
Prix ↔ quantité  : -0,431
Quantité ↔ CA    :  0,096
```

Conclusion :

> La quantité reste la variable cible principale du forecasting.

Le CA pourra être recalculé après prévision :

```text
CA prévisionnel = quantité prévue × prix unitaire
```

Rapport :

```text
data/processed/eda/quantity_revenue_price_products.csv
```

---

## J4.5.4 — Anomalies par produit

Script :

```text
src/data/eda_product_anomalies.py
```

**✅ PASS**

14 produits analysés sur 365 observations chacun.

### Produits avec le plus fort taux d'anomalies IQR

```text
Biscuits Chocolat : 4,38 %
Jus de fruit 1L   : 2,74 %
Sucre 1kg         : 2,47 %
Dentifrice 75ml   : 2,47 %
Eau Vive 1.5L     : 1,92 %
```

### Produit le plus variable

```text
Produit 001
CV = 0,74
```

### Produit le plus stable

```text
Coca-Cola 33cl
CV = 0,40
```

Conclusion :

> Les anomalies doivent être interprétées relativement au comportement propre de chaque produit.

Rapport :

```text
data/processed/eda/product_anomalies_analysis.csv
```

---

# 12. J4.5.5 — Synthèse anomalies & relations

**Statut : ✅ VALIDÉ**

Script :

```text
src/data/eda_anomaly_synthesis.py
```

Rapport :

```text
data/processed/eda/eda_anomaly_synthesis.csv
```

Nombre d'indicateurs :

```text
23
```

### Synthèse finale

Les analyses montrent :

* les anomalies extrêmes sont rares ;
* les fortes demandes sont concentrées principalement le vendredi et le samedi ;
* certains produits présentent davantage de variabilité que d'autres ;
* `Produit 001` est le produit le plus variable ;
* `Coca-Cola 33cl` est le produit le plus stable ;
* `Biscuits Chocolat` présente le taux d'anomalies IQR le plus élevé ;
* quantité et CA ont une relation positive modérée ;
* les produits plus chers tendent à présenter des volumes plus faibles ;
* les anomalies ne doivent pas être supprimées automatiquement.

### Conséquences pour le forecasting

La variable cible principale sera :

```text
quantity
```

Les futurs modèles devront exploiter notamment :

```text
historique de demande
jour de semaine
mois
saisonnalité
tendance récente
produit
variabilité du produit
```

Les anomalies seront conservées afin de ne pas supprimer artificiellement des comportements de demande potentiellement utiles.

> **J4.5 — ANOMALIES & RELATIONS : VALIDÉ**

---

# 13. J4 — État actuel

| Étape                                   | Statut    |
| --------------------------------------- | --------- |
| J4.1 — Objectifs EDA                    | ✅ VALIDÉ  |
| J4.2 — Analyse descriptive              | ✅ VALIDÉ  |
| J4.3 — Analyse temporelle               | ✅ VALIDÉ  |
| J4.4 — Produits & catégories            | ✅ VALIDÉ  |
| J4.5.1 — Anomalies globales             | ✅ VALIDÉ  |
| J4.5.2 — Journées extrêmes              | ✅ VALIDÉ  |
| J4.5.3 — Quantité / CA / Prix           | ✅ VALIDÉ  |
| J4.5.4 — Anomalies par produit          | ✅ VALIDÉ  |
| J4.5.5 — Synthèse anomalies & relations | ✅ VALIDÉ  |
| **J4.6 — Synthèse finale EDA**          | ⏳ À FAIRE |

---

# 14. Fichiers EDA créés

```text
src/data/eda_descriptive.py
src/data/eda_product_profile.py
src/data/eda_product_performance.py
src/data/eda_category_product_relationship.py
src/data/eda_product_visualizations.py
src/data/eda_anomaly_detection.py
src/data/eda_extreme_days.py
src/data/eda_quantity_revenue_price.py
src/data/eda_product_anomalies.py
src/data/eda_anomaly_synthesis.py
```

Rapports :

```text
data/processed/eda/anomaly_detection_report.csv
data/processed/eda/extreme_days_analysis.csv
data/processed/eda/quantity_revenue_price_products.csv
data/processed/eda/product_anomalies_analysis.csv
data/processed/eda/eda_anomaly_synthesis.csv
```

Visualisations :

```text
data/processed/eda/top_products_quantity.png
data/processed/eda/top_products_revenue.png
data/processed/eda/product_volume_vs_revenue.png
data/processed/eda/category_product_contribution.png
data/processed/eda/product_demand_variability.png
```

---

# 15. Git — État

Dernier commit validé :

```text
924bcb7 feat: complete J3 data cleaning
```

J4.5 a été développé après ce commit.

### Prochain commit prévu

Après validation de l'ensemble des fichiers J4.5 :

```text
feat: complete J4.5 anomaly and relationship analysis
```

⚠️ Ne pas committer de données sensibles ou de fichiers `.sql`.

Le fichier :

```text
*.sql
```

reste ignoré par Git.

---

# 16. Prochaine étape

## J4.6 — Synthèse finale de l'EDA

Objectifs :

1. consolider toutes les conclusions J4 ;
2. identifier les variables pertinentes pour le forecasting ;
3. documenter les risques et limites du dataset ;
4. déterminer les éléments à conserver pour J5 ;
5. préparer la transition vers **Feature Engineering**.

Après validation de J4.6 :

```text
J4 — EDA : VALIDÉ
```

Puis démarrage de :

```text
J5 — FEATURE ENGINEERING
```

avec notamment :

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

---

# 17. Checkpoint actuel

**Projet :** AI Sales Forecasting

**Checkpoint :** **J4.5.5**

**Statut :** **J4 — EDA EN COURS**

**Dernière étape validée :**

```text
J4.5.5 — Synthèse anomalies & relations
```

**Prochaine étape :**

```text
J4.6 — Synthèse finale EDA
```

**J5 — Feature Engineering : NON COMMENCÉ**
