# AI Sales Forecasting — Project Checkpoint

> Dernière mise à jour : 2026-09-03
> Checkpoint : **J5.7**
> Statut global : **J5 — FEATURE ENGINEERING VALIDÉ**

---

# 1. Projet

**Nom :** AI Sales Forecasting

**Objectif :**
Développer une solution de prévision des ventes basée sur le Machine Learning afin de prévoir la demande future, puis proposer des recommandations de stock.

**Architecture cible :**

```text
PostgreSQL / KShop
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
Forecast J+1 à J+7
        ↓
Stock Recommendation
        ↓
Streamlit Dashboard
        ↓
Production
```

---

# 2. Progression globale

| Étape                     | Statut    |
| ------------------------- | --------- |
| J1 — Initialisation       | ✅ TERMINÉ |
| J2 — Dataset              | ✅ TERMINÉ |
| J3 — Data Cleaning        | ✅ TERMINÉ |
| J4 — EDA                  | ✅ TERMINÉ |
| J5 — Feature Engineering  | ✅ TERMINÉ |
| J6 — Machine Learning     | ⏳ À VENIR |
| J7 — Forecast             | ⏳ À VENIR |
| J8 — Stock Recommendation | ⏳ À VENIR |
| J9 — Streamlit Dashboard  | ⏳ À VENIR |
| J10 — Production          | ⏳ À VENIR |

---

# 3. J1 — Initialisation

**Statut : ✅ TERMINÉ**

Le projet Python `ai-sales-forecasting` a été initialisé.

Structure principale :

```text
ai-sales-forecasting/
├── config/
├── dashboard/
├── data/
│   ├── processed/
│   └── raw/
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

Point d'entrée :

```text
src/main.py
```

Validation :

```text
AI Sales Forecasting - Project initialized successfully.
```

---

# 4. J2 — Dataset

**Statut : ✅ TERMINÉ**

Dataset utilisé :

```text
data/raw/kshop_sales_synthetic.csv
```

Nature du dataset :

**Dataset synthétique**, construit à partir de la structure métier et du catalogue KShop.

Le dataset ne représente pas de vraies transactions clients.

### Caractéristiques

```text
Période        : 2025-09-01 → 2026-08-31
Durée          : 365 jours
Produits       : 14
Granularité    : journalière / produit
Observations   : 5 110
Colonnes       : 7
Random seed    : 42
```

Colonnes :

```text
date
product_id
product_name
category
quantity
unit_price
revenue
```

Target prévue pour le Machine Learning :

```text
quantity
```

### Validation J2

```text
Dataset généré                         ✅
Structure correcte                     ✅
Dates valides                          ✅
14 produits                            ✅
365 jours                              ✅
Quantités valides                      ✅
Prix valides                           ✅
Revenue cohérent                       ✅
Aucun problème critique                ✅
```

Commit J2 :

```text
ff629be chore: complete J2 dataset validation
```

---

# 5. J3 — Data Cleaning

**Statut : ✅ TERMINÉ**

Source :

```text
data/raw/kshop_sales_synthetic.csv
```

Output :

```text
data/processed/sales_clean.csv
```

Script principal :

```text
src/data/clean_sales.py
```

### Règles appliquées

* Validation des dates
* Validation de `product_id`
* Validation de `quantity`
* Validation de `unit_price`
* Recalcul de `revenue`
* Suppression des doublons `date + product_id`
* Nettoyage des espaces textuels
* Gestion explicite des valeurs NULL critiques
* Analyse des valeurs extrêmes
* Conservation des anomalies cohérentes

Le dataset RAW n'a pas été modifié.

### Résultats

```text
Lignes initiales          : 5 110
Lignes finales            : 5 110
NULL critiques supprimés : 0
Quantités invalides      : 0
Prix invalides            : 0
Doublons supprimés       : 0
```

Validation :

```text
J3 — DATA CLEANING : OK
```

Commit J3 :

```text
924bcb7 feat: complete J3 data cleaning
```

---

# 6. J4 — Exploratory Data Analysis

**Statut : ✅ TERMINÉ**

J4 a été entièrement validé.

## 6.1 Vue générale

```text
Lignes          : 5 110
Produits        : 14
Jours            : 365
Colonnes         : 7
```

## 6.2 Quantity

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

Les observations à quantité nulle représentent environ :

```text
2.78 %
```

## 6.3 Revenue

```text
Total       : 67 966 700 AR
Moyenne     : 13 300.72 AR
Médiane     : 10 800 AR
Maximum     : 81 000 AR
P95         : 33 000 AR
P99         : 49 500 AR
```

## 6.4 Performance journalière

```text
Quantity moyenne : 76.92
Quantity médiane : 75
Minimum          : 36
Maximum          : 152

CA moyen         : 186 210.14 AR
CA médian        : 180 700 AR
CA minimum       : 92 400 AR
CA maximum       : 372 200 AR
```

## 6.5 Analyse temporelle

Jour le plus performant :

```text
Samedi
```

Jour le moins performant :

```text
Dimanche
```

Mois le plus performant :

```text
Décembre 2025
```

Mois le moins performant :

```text
Février 2026
```

Évolution de la demande :

```text
Premiers 30 jours : 71.17
Derniers 30 jours : 82.93

Évolution : +16.53 %
```

Sur 90 jours :

```text
Premiers 90 jours : 76.76
Derniers 90 jours : 83.08

Évolution : +8.24 %
```

## 6.6 Analyse produits

Produit avec le plus gros volume :

```text
Coca-Cola 33cl
Quantité : 3 995
```

Produit avec le plus gros chiffre d'affaires :

```text
Huile alimentaire 1L
CA : 9 190 500 AR
```

Produit le plus variable :

```text
Produit 001 CV
CV : 0.74
```

Produit le plus stable :

```text
Coca-Cola 33cl
CV : 0.40
```

## 6.7 Analyse des valeurs extrêmes

```text
Observations > P95 : 17 jours
Observations > P99 : 4 jours
```

Les jours extrêmes sont principalement concentrés sur :

```text
Vendredi
Samedi
```

Les anomalies cohérentes ont été conservées.

## 6.8 Corrélations

Corrélations au niveau des lignes :

```text
quantity ↔ revenue      : 0.511
quantity ↔ unit_price   : -0.280
unit_price ↔ revenue    : 0.510
```

Au niveau produit :

```text
unit_price ↔ revenue    : 0.781
unit_price ↔ quantity   : -0.431
quantity ↔ revenue      : 0.096
```

### Conclusions J4

```text
Target ML              : quantity
Historique temporel    : important
Calendrier             : important
LAG                    : pertinent
Rolling features       : pertinentes
Anomalies               : à conserver
CA                       : quantity × unit_price
```

Une différence de nommage entre certaines catégories a également été identifiée :

```text
Alimentaire
Produits alimentaires
```

Cette incohérence a été signalée sans modification automatique.

Commit J4 :

```text
fe779a6 feat: complete J4 EDA
```

---

# 7. J5 — Feature Engineering

**Statut : ✅ VALIDÉ**

J5.1 à J5.7 sont entièrement terminés et validés.

J5.8 — Commit + GitHub est la prochaine étape.

---

# 8. J5.1 — Définition des règles

**Statut : ✅ VALIDÉ**

Target ML :

```text
quantity
```

Features calendaires :

```text
day_of_week
day_of_month
month
week_of_year
is_weekend
```

Features LAG :

```text
lag_1
lag_7
lag_14
```

Features Rolling :

```text
rolling_mean_7
rolling_mean_14
rolling_mean_30
```

### Règles

Les LAG et Rolling sont calculés :

```text
par product_id
après tri par product_id + date
```

Aucune feature ne doit utiliser la demande future.

Aucune normalisation n'est réalisée pendant J5.

Aucun entraînement Machine Learning n'est réalisé pendant J5.

---

# 9. J5.2 — Variables calendaires

**Statut : ✅ VALIDÉ**

Script :

```text
src/data/create_calendar_features.py
```

Source :

```text
data/processed/sales_clean.csv
```

Output :

```text
data/processed/sales_calendar_features.csv
```

Features :

```text
day_of_week
day_of_month
month
week_of_year
is_weekend
```

### Validation

```text
5 110 lignes                         ✅
14 produits                          ✅
365 jours                            ✅
Dates valides                        ✅
5 features présentes                 ✅
Aucun NULL                           ✅
Valeurs calendaires valides          ✅
Cohérence weekend                    ✅
Nombre de lignes inchangé            ✅
quantity inchangée                   ✅
revenue inchangé                     ✅
Données sources intactes             ✅
```

---

# 10. J5.3 — Variables LAG

**Statut : ✅ VALIDÉ**

Script :

```text
src/data/create_lag_features.py
```

Source :

```text
data/processed/sales_calendar_features.csv
```

Output :

```text
data/processed/sales_lag_features.csv
```

Features :

```text
lag_1
lag_7
lag_14
```

Définition :

```text
lag_1  = quantity à J-1
lag_7  = quantity à J-7
lag_14 = quantity à J-14
```

Calcul effectué séparément par :

```text
product_id
```

### NULL structurels

```text
lag_1  : 14
lag_7  : 98
lag_14 : 196
```

### Validation

```text
5 110 lignes                         ✅
14 produits                          ✅
Tri product_id + date                ✅
Ordre chronologique                  ✅
LAG correctement calculés            ✅
NULL structurels conformes           ✅
Valeurs LAG >= 0                     ✅
Absence de data leakage              ✅
Données sources intactes             ✅
```

---

# 11. J5.4 — Variables Rolling

**Statut : ✅ VALIDÉ**

Script :

```text
src/data/create_rolling_features.py
```

Source :

```text
data/processed/sales_lag_features.csv
```

Output :

```text
data/processed/sales_rolling_features.csv
```

Features :

```text
rolling_mean_7
rolling_mean_14
rolling_mean_30
```

Méthode :

```text
shift(1) + rolling(...)
```

Ainsi, la valeur du jour courant n'est jamais utilisée pour calculer sa propre moyenne historique.

### NULL structurels

```text
rolling_mean_7  : 98
rolling_mean_14 : 196
rolling_mean_30 : 420
```

### Validation

```text
5 110 lignes                         ✅
14 produits                          ✅
Tri chronologique                    ✅
Rolling 7 correctement calculée      ✅
Rolling 14 correctement calculée     ✅
Rolling 30 correctement calculée     ✅
NULL structurels conformes           ✅
Valeurs Rolling >= 0                 ✅
Absence de data leakage              ✅
Données sources intactes             ✅
```

---

# 12. J5.5 — Dataset ML final

**Statut : ✅ VALIDÉ**

Script :

```text
src/data/create_ml_dataset.py
```

Source :

```text
data/processed/sales_rolling_features.csv
```

Output :

```text
data/processed/sales_ml_ready.csv
```

### Structure finale

Les 18 colonnes sont conservées :

```text
date
product_id
product_name
category
quantity
unit_price
revenue

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

### Target

```text
quantity
```

### Filtrage

Seules les lignes ne disposant pas de l'historique nécessaire aux features ML sont supprimées.

La Rolling 30 nécessite 30 jours historiques.

Donc :

```text
30 premiers jours × 14 produits = 420 lignes retirées
```

### Résultats

```text
Dataset source       : 5 110 lignes
Dataset ML           : 4 690 lignes
Lignes retirées      : 420
Produits             : 14
Jours / produit      : 335
Colonnes             : 18
```

Période du dataset ML :

```text
2025-10-01 → 2026-08-31
```

---

# 13. J5.6 — Quality Control

**Statut : ✅ VALIDÉ À 100 %**

Script :

```text
src/data/quality_check_ml_dataset.py
```

Dataset contrôlé :

```text
data/processed/sales_ml_ready.csv
```

### Contrôles

```text
Structure                         ✅
Types                             ✅
NULL                              ✅ Aucun
Doublons complets                 ✅ Aucun
Clés date + product_id            ✅ Uniques
14 produits                       ✅
335 jours / produit               ✅
Continuité temporelle             ✅
quantity >= 0                     ✅
quantity entière                  ✅
unit_price > 0                    ✅
revenue = quantity × unit_price   ✅
Features calendaires              ✅
LAG                               ✅
Rolling                           ✅
Data leakage                      ✅ Aucun
Valeurs extrêmes                  ✅ Analysées
Distribution produits             ✅ Équilibrée
Intégrité finale                  ✅
```

### Distribution de `quantity`

```text
count    4690.000000
mean        5.531130
std         3.716378
min         0.000000
25%         3.000000
50%         5.000000
75%         8.000000
max        26.000000
```

Valeurs extrêmes :

```text
P95 quantity : 13
P99 quantity : 17

Observations > P95 : 183
Observations > P99 : 40
```

Les valeurs extrêmes sont conservées.

---

# 14. J5.7 — Validation finale Feature Engineering

**Statut : ✅ VALIDÉ À 100 %**

Script :

```text
src/data/validate_feature_engineering.py
```

### Pipeline validé

```text
sales_clean.csv
        ↓
5 110 lignes
        ↓
sales_calendar_features.csv
        ↓
5 110 lignes
        ↓
sales_lag_features.csv
        ↓
5 110 lignes
        ↓
sales_rolling_features.csv
        ↓
5 110 lignes
        ↓
sales_ml_ready.csv
        ↓
4 690 lignes
```

### Validation globale

```text
Tous les datasets disponibles            ✅
Structure du pipeline                    ✅
14 produits                              ✅
Historique 365 jours                     ✅
Dataset ML 335 jours / produit           ✅
Intégrité des données sources             ✅
Features calendaires                      ✅
Features LAG                              ✅
Features Rolling                          ✅
NULL structurels conformes                ✅
Réduction du dataset correcte             ✅
Target quantity conservée                 ✅
Absence de data leakage                   ✅
18 colonnes finales                       ✅
4 690 observations ML                     ✅
```

### Résultat final J5.7

```text
Target               : quantity
Produits             : 14
Historique           : 365 jours
Dataset ML           : 335 jours / produit
Observations ML      : 4 690
Colonnes finales     : 18
Data leakage         : Aucun
NULL ML              : Aucun
```

```text
============================================================
J5.7 — VALIDATION FINALE : OK
============================================================
```

---

# 15. Dataset ML final

Fichier principal :

```text
data/processed/sales_ml_ready.csv
```

Dimensions :

```text
4 690 lignes
18 colonnes
14 produits
335 jours par produit
```

Période :

```text
2025-10-01 → 2026-08-31
```

### Features ML

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

### Target

```text
quantity
```

---

# 16. Fichiers créés pendant J5

### Scripts

```text
src/data/create_calendar_features.py
src/data/create_lag_features.py
src/data/create_rolling_features.py
src/data/create_ml_dataset.py
src/data/quality_check_ml_dataset.py
src/data/validate_feature_engineering.py
```

### Datasets intermédiaires

```text
data/processed/sales_calendar_features.csv
data/processed/sales_lag_features.csv
data/processed/sales_rolling_features.csv
data/processed/sales_ml_ready.csv
```

---

# 17. Décisions techniques importantes

## Dataset synthétique

Le dataset de forecasting est synthétique.

Il a été généré afin de disposer d'un historique suffisamment riche pour construire et tester le pipeline de Machine Learning.

Il ne doit pas être présenté comme un historique réel de transactions clients.

---

## Target

La variable cible du modèle est :

```text
quantity
```

Le chiffre d'affaires pourra ensuite être estimé avec :

```text
forecast_quantity × unit_price
```

---

## Data leakage

Les variables historiques sont construites uniquement à partir des observations précédentes.

Les LAG utilisent :

```text
J-1
J-7
J-14
```

Les Rolling utilisent uniquement les jours précédents :

```text
J-1 jusqu'aux fenêtres historiques nécessaires
```

Aucune information future n'est utilisée.

---

## Anomalies

Les valeurs extrêmes identifiées pendant J4 sont conservées.

Elles ne sont pas supprimées automatiquement, car elles peuvent représenter des comportements commerciaux légitimes.

---

## Normalisation

Aucune normalisation n'est réalisée pendant J5.

La préparation spécifique aux modèles sera traitée pendant J6 si nécessaire.

---

# 18. Git — état du projet

Commits principaux :

```text
J2  : ff629be chore: complete J2 dataset validation
J3  : 924bcb7 feat: complete J3 data cleaning
J4.5: 65caa49 feat: complete J4.5 anomaly and relationship analysis
J4  : fe779a6 feat: complete J4 EDA
```

### J5

```text
J5.1 → J5.7 : ✅ VALIDÉS
J5.8          : ⏳ À FAIRE
```

Commit prévu :

```text
feat: complete J5 feature engineering
```

---

# 19. Prochaine étape — J5.8

## J5.8 — Commit + GitHub

Objectifs :

```text
1. Vérifier git status
2. Vérifier les modifications
3. Vérifier les fichiers J5
4. Ajouter uniquement les fichiers nécessaires
5. Créer le commit J5
6. Push vers GitHub
7. Vérifier le repository
8. Vérifier que le working tree est propre
```

Le fichier privé :

```text
kshop_export.sql
```

ne doit jamais être ajouté au repository.

Les secrets, mots de passe, credentials et chaînes de connexion ne doivent pas être commités.

---

# 20. État actuel du projet

```text
============================================================
AI SALES FORECASTING
============================================================

J1  — INITIALISATION          ✅
J2  — DATASET                 ✅
J3  — DATA CLEANING           ✅
J4  — EDA                     ✅
J5  — FEATURE ENGINEERING     ✅
J6  — MACHINE LEARNING        ⏳
J7  — FORECAST                ⏳
J8  — STOCK RECOMMENDATION    ⏳
J9  — STREAMLIT DASHBOARD     ⏳
J10 — PRODUCTION              ⏳

============================================================
CURRENT CHECKPOINT : J5.7
============================================================

Feature Engineering : VALIDÉ À 100 %
Dataset ML          : 4 690 lignes
Produits             : 14
Features             : 11
Colonnes finales     : 18
Target               : quantity
Data leakage         : Aucun
NULL ML              : Aucun

PROCHAINE ÉTAPE :
J5.8 — COMMIT + GITHUB
============================================================
```
