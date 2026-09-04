# AI Sales Forecasting — Project Checkpoint

> Dernière mise à jour : 2026-09-04
> Checkpoint : **J6.9**
> Statut global : **J6 — MACHINE LEARNING TERMINÉ**
> Prochaine étape : **J7 — FORECASTING J+1 À J+7**

---

# 1. Projet

**Nom :** AI Sales Forecasting

**Objectif :**

Construire un système de prévision des ventes permettant de :

* analyser l'historique des ventes ;
* identifier les tendances et saisonnalités ;
* construire des variables temporelles et historiques ;
* entraîner plusieurs modèles de Machine Learning ;
* sélectionner le meilleur modèle ;
* prévoir la demande future J+1 à J+7 ;
* transformer les prévisions en recommandations de stock ;
* présenter les résultats dans un dashboard Streamlit.

**Architecture cible :**

```text
KShop / PostgreSQL
        ↓
     Pandas
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

# 2. Nature des données

Le dataset utilisé pour le projet est **synthétique**.

Il a été généré à partir de la structure métier/catalogue de KShop afin de disposer d'un historique suffisamment long pour construire un projet de forecasting crédible.

Il ne représente pas des transactions clients réelles.

Cette information doit être clairement mentionnée dans le README et dans le portfolio.

---

# 3. Dataset

## Dataset principal

```text
data/raw/kshop_sales_synthetic.csv
```

Période :

```text
2025-09-01 → 2026-08-31
```

Caractéristiques :

```text
365 jours
14 produits
5 110 lignes
1 ligne = 1 produit / 1 jour
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

Target Machine Learning :

```text
quantity
```

---

# 4. Structure du projet

```text
ai-sales-forecasting/
│
├── config/
│
├── dashboard/
│
├── data/
│   ├── processed/
│   │   ├── eda/
│   │   ├── ml_ready/
│   │   ├── ml_split/
│   │   ├── sales_clean.csv
│   │   ├── sales_calendar_features.csv
│   │   ├── sales_lag_features.csv
│   │   ├── sales_rolling_features.csv
│   │   └── sales_ml_ready.csv
│   │
│   └── raw/
│       └── kshop_sales_synthetic.csv
│
├── models/
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   ├── hist_gradient_boosting.joblib
│   ├── final_model.joblib
│   └── final_model_metadata.json
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── forecasting/
│   ├── models/
│   └── utils/
│
├── tests/
│
├── .env.example
├── .gitignore
├── PROJECT_CHECKPOINT.md
├── README.md
├── requirements.txt
└── src/main.py
```

---

# 5. J1 — INITIALISATION

**Statut : ✅ TERMINÉ**

Projet initialisé.

Environnement Python/Conda créé :

```text
ai-sales-forecasting
```

Python :

```text
3.12.14
```

Stack principale :

```text
NumPy
Pandas
Scikit-learn
Matplotlib
Plotly
Streamlit
SQLAlchemy
PostgreSQL / psycopg2
python-dotenv
Joblib
OpenPyXL
Jupyter
Pytest
```

`src/main.py` validé.

---

# 6. J2 — DATASET

**Statut : ✅ TERMINÉ**

Dataset synthétique généré et validé.

Résultats :

```text
5 110 lignes
14 produits
365 jours
2025-09-01 → 2026-08-31
```

Contrôles effectués :

```text
[PASS] Colonnes requises
[PASS] Dates
[PASS] Quantités
[PASS] Prix
[PASS] Revenue
[PASS] Absence de valeurs incohérentes
[PASS] Cohérence revenue = quantity × unit_price
```

---

# 7. J3 — DATA CLEANING

**Statut : ✅ TERMINÉ**

Source :

```text
data/raw/kshop_sales_synthetic.csv
```

Output :

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

Règles appliquées :

* validation des dates ;
* validation de `product_id` ;
* validation de `quantity` ;
* validation de `unit_price` ;
* recalcul du revenue ;
* suppression des doublons `date + product_id` ;
* nettoyage des espaces ;
* traitement des NULL critiques.

Les valeurs extrêmes cohérentes ont été conservées pour analyse.

---

# 8. J4 — EXPLORATORY DATA ANALYSIS

**Statut : ✅ TERMINÉ**

## 8.1 Profil général

```text
5 110 lignes
14 produits
365 jours
7 colonnes
```

## 8.2 Quantity

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

## 8.3 Revenue

```text
Total       : 67 966 700 AR
Moyenne     : 13 300.72 AR
Médiane     : 10 800 AR
Maximum     : 81 000 AR
P95         : 33 000 AR
P99         : 49 500 AR
```

## 8.4 Analyse temporelle

Moyenne quotidienne :

```text
Quantity : 76.92
Revenue  : 186 210.14 AR
```

Tendance :

```text
Premiers 30 jours : 71.17 unités/jour
Derniers 30 jours : 82.93 unités/jour

Évolution : +16.53 %
```

Les 90 derniers jours présentent également une progression par rapport aux 90 premiers jours.

## 8.5 Analyse des produits

Produit avec le plus gros volume :

```text
Coca-Cola 33cl
3995 unités
```

Produit avec le plus gros chiffre d'affaires :

```text
Huile alimentaire 1L
9 190 500 AR
```

Produit le plus variable :

```text
Produit 001 CV
CV = 0.74
```

Produit le plus stable :

```text
Coca-Cola 33cl
CV = 0.40
```

## 8.6 Anomalies

```text
17 jours > P95
4 jours > P99
```

Les journées extrêmes sont principalement concentrées sur :

```text
Vendredi
Samedi
```

Les anomalies ont été conservées pour ne pas supprimer artificiellement des comportements potentiellement utiles au forecasting.

## 8.7 Conclusion EDA

Facteurs importants identifiés :

* historique de la demande ;
* jour de la semaine ;
* mois ;
* produit ;
* saisonnalité ;
* tendances temporelles ;
* lags ;
* moyennes mobiles.

Target retenue :

```text
quantity
```

Le chiffre d'affaires sera ensuite calculé à partir des quantités prévues :

```text
forecast_revenue = forecast_quantity × unit_price
```

---

# 9. J5 — FEATURE ENGINEERING

**Statut : ✅ TERMINÉ**

Dataset ML :

```text
data/processed/sales_ml_ready.csv
```

Résultat :

```text
4 690 lignes
18 colonnes
14 produits
2025-10-01 → 2026-08-31
```

## Features calendaires

```text
day_of_week
day_of_month
month
week_of_year
is_weekend
```

## Features Lag

```text
lag_1
lag_7
lag_14
```

## Features Rolling

```text
rolling_mean_7
rolling_mean_14
rolling_mean_30
```

Les lags et rolling features sont calculés séparément pour chaque `product_id`.

Tri chronologique :

```text
product_id
date
```

Contrôles :

```text
[PASS] Feature engineering
[PASS] Absence de data leakage
[PASS] Structure finale
[PASS] Cohérence temporelle
[PASS] Dataset ML Ready
```

---

# 10. J6 — MACHINE LEARNING

**Statut : ✅ TERMINÉ**

---

## J6.1 — Temporal Split

**Statut : ✅ VALIDÉ**

Dataset :

```text
data/processed/sales_ml_ready.csv
```

Split chronologique :

### Train

```text
2025-10-01 → 2026-06-30
3 822 lignes
```

### Validation

```text
2026-07-01 → 2026-07-31
434 lignes
```

### Test

```text
2026-08-01 → 2026-08-31
434 lignes
```

Contrôles :

```text
[PASS] Aucun chevauchement temporel
[PASS] Train < Validation < Test
[PASS] 14 produits dans chaque split
[PASS] Target valide
[PASS] Aucun doublon date + product_id
```

---

# 11. J6.2 — Préparation X / y

**Statut : ✅ VALIDÉ**

Script :

```text
src/models/prepare_ml_data.py
```

Features catégorielles :

```text
product_id
```

Features numériques :

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

Target :

```text
quantity
```

Résultats :

```text
X_train      : (3822, 12)
y_train      : (3822,)

X_validation : (434, 12)
y_validation : (434,)

X_test       : (434, 12)
y_test       : (434,)
```

---

# 12. J6.3 — Baseline

**Statut : ✅ VALIDÉ**

Baseline :

```text
lag_7
```

Évaluation sur Validation :

```text
MAE  : 2.7442
RMSE : 3.6604
R²   : 0.0394
```

Le Test n'a pas été utilisé.

---

# 13. J6.4 — Entraînement des modèles

**Statut : ✅ VALIDÉ**

Modèles entraînés :

```text
1. Random Forest Regressor
2. Gradient Boosting Regressor
3. HistGradientBoosting Regressor
```

Les modèles ont été entraînés uniquement sur le Train.

Modèles sauvegardés :

```text
models/random_forest.joblib
models/gradient_boosting.joblib
models/hist_gradient_boosting.joblib
```

---

# 14. J6.5 — Évaluation Validation

**Statut : ✅ VALIDÉ**

Résultats :

| Modèle                |        MAE |       RMSE |         R² |
| --------------------- | ---------: | ---------: | ---------: |
| **Gradient Boosting** | **2.0337** | **2.6249** | **0.5060** |
| Random Forest         |     2.0640 |     2.6599 |     0.4928 |
| HistGradientBoosting  |     2.1287 |     2.7862 |     0.4434 |
| Baseline lag_7        |     2.7442 |     3.6604 |     0.0394 |

Gradient Boosting :

```text
Gain MAE  : 25.89 %
Gain RMSE : 28.29 %
```

Gradient Boosting est meilleur sur :

```text
MAE
RMSE
R²
```

---

# 15. J6.6 — Comparaison approfondie

**Statut : ⏭️ NON EXÉCUTÉ SÉPARÉMENT**

J6.6 n'a pas fait l'objet d'une exécution indépendante.

La comparaison nécessaire a néanmoins été réalisée dans J6.5 et J6.7 à partir des métriques Validation.

Aucune exécution fictive de J6.6 ne doit être déclarée comme réalisée.

---

# 16. J6.7 — Sélection du meilleur modèle

**Statut : ✅ VALIDÉ**

Critères :

```text
1. MAE ASC
2. RMSE ASC
3. R² DESC
```

Classement :

```text
1. gradient_boosting
2. random_forest
3. hist_gradient_boosting
4. baseline_lag_7
```

Modèle sélectionné :

```text
Gradient Boosting
```

Performances :

```text
MAE  : 2.0337
RMSE : 2.6249
R²   : 0.5060
```

Fichier :

```text
data/processed/ml_ready/best_model_selection.csv
```

Validation :

```text
[PASS] Gradient Boosting sélectionné
[PASS] Meilleur MAE
[PASS] Meilleur RMSE
[PASS] Meilleur R²
[PASS] Test NON utilisé
```

---

# 17. J6.8 — Sauvegarde du modèle final

**Statut : ✅ VALIDÉ**

Modèle final :

```text
models/final_model.joblib
```

Métadonnées :

```text
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

Métadonnées conservées :

* modèle sélectionné ;
* target ;
* features ;
* datasets Train/Validation ;
* métriques Validation ;
* critères de sélection ;
* confirmation que le Test n'a pas été utilisé.

Validation :

```text
[PASS] Modèle final non vide
[PASS] Métadonnées valides
[PASS] Target = quantity
[PASS] 12 features enregistrées
[PASS] Test NON utilisé
```

---

# 18. J6.9 — Évaluation finale sur le Test

**Statut : ✅ VALIDÉ**

Dataset Test :

```text
2026-08-01 → 2026-08-31
434 lignes
14 produits
```

Le Test a été utilisé pour la première fois à cette étape.

## Modèle final

```text
Gradient Boosting
```

Résultats Test :

```text
MAE  : 2.0928
RMSE : 2.7149
R²   : 0.5474
```

## Baseline Test

```text
MAE  : 2.9793
RMSE : 3.9691
R²   : 0.0327
```

## Amélioration finale

```text
Gain MAE  : 29.75 %
Gain RMSE : 31.60 %
```

Le modèle final généralise correctement sur le Test.

Le R² Test :

```text
0.5474
```

est supérieur au R² Validation :

```text
0.5060
```

Aucun signe évident de surapprentissage sévère n'est observé à partir de cette comparaison.

Prédictions :

```text
Moyenne : 5.9745
Minimum : 1.4912
Maximum : 16.2720
```

Fichiers créés :

```text
data/processed/ml_ready/final_model_test_predictions.csv

data/processed/ml_ready/final_model_test_evaluation.csv
```

Validation :

```text
[PASS] 434 prédictions générées
[PASS] Aucune prédiction NULL
[PASS] Période Test correcte
[PASS] Test évalué uniquement à J6.9
[PASS] Évaluation finale terminée
```

---

# 19. Résultat final du Machine Learning

Le modèle de référence du projet est :

```text
Gradient Boosting Regressor
```

Performance finale sur le Test :

```text
MAE  = 2.0928
RMSE = 2.7149
R²   = 0.5474
```

Comparaison avec la baseline :

```text
Baseline MAE      = 2.9793
Gradient Boosting = 2.0928

Amélioration      = 29.75 %
```

Conclusion :

> Le Gradient Boosting est retenu comme modèle de forecasting de la demande pour la suite du projet.

---

# 20. Artefacts Machine Learning

```text
models/
├── random_forest.joblib
├── gradient_boosting.joblib
├── hist_gradient_boosting.joblib
├── final_model.joblib
└── final_model_metadata.json
```

```text
data/processed/ml_ready/
├── X_train.csv
├── y_train.csv
├── X_validation.csv
├── y_validation.csv
├── X_test.csv
├── y_test.csv
├── baseline_validation.csv
├── training_results.csv
├── model_evaluation_validation.csv
├── best_model_selection.csv
├── final_model_test_predictions.csv
└── final_model_test_evaluation.csv
```

---

# 21. Git

Commits précédents validés :

```text
07fd765 chore: initialize AI sales forecasting project

ff629be chore: complete J2 dataset validation

924bcb7 feat: complete J3 data cleaning

65caa49 feat: complete J4.5 anomaly and relationship analysis

fe779a6 feat: complete J4 EDA

8acac3c feat: complete J5 feature engineering
```

Dernier état Git connu avant J6 :

```text
main
origin/main
working tree clean
```

J6 doit maintenant être préparé pour un commit dédié après vérification des nouveaux artefacts.

Le fichier privé :

```text
kshop_export.sql
```

reste exclu du dépôt via :

```text
*.sql
```

Aucune donnée sensible issue de la base KShop ne doit être commitée.

---

# 22. État global du projet

```text
J1  Initialisation                  ✅
J2  Dataset                         ✅
J3  Data Cleaning                   ✅
J4  EDA                             ✅
J5  Feature Engineering             ✅
J6  Machine Learning                ✅
J7  Forecasting                     ⏳
J8  Stock Recommendation            ⏳
J9  Streamlit Dashboard             ⏳
J10 Production / Deployment         ⏳
```

---

# 23. Prochaine étape — J7

## J7 — FORECASTING J+1 À J+7

Objectif :

Utiliser le modèle final :

```text
models/final_model.joblib
```

pour produire des prévisions de demande :

```text
J+1
J+2
J+3
J+4
J+5
J+6
J+7
```

pour chacun des 14 produits.

Le forecasting devra prendre en compte le caractère **multi-step** du problème et éviter toute fuite de données futures.

Sortie cible :

```text
data/processed/forecast/
```

avec notamment :

```text
forecast_j7.csv
```

Structure prévue :

```text
date
product_id
product_name
forecast_horizon
predicted_quantity
unit_price
predicted_revenue
```

La prévision de chiffre d'affaires sera dérivée de :

```text
predicted_revenue
=
predicted_quantity × unit_price
```

---

# 24. Règle de progression

Le projet suit une validation étape par étape.

Une étape n'est considérée comme terminée que lorsque :

```text
1. Le code est créé
2. Le script s'exécute correctement
3. Les contrôles passent
4. Les fichiers attendus sont créés
5. Les résultats sont analysés
6. Le checkpoint est mis à jour
7. Le commit Git est effectué
```

**Checkpoint actuel : J6.9 — Machine Learning terminé.**

**Prochaine étape : J7 — Forecasting J+1 à J+7.**
