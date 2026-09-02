# AI Sales Forecasting — Project Checkpoint

> Dernière mise à jour : 2026-09-03
> Checkpoint : J2.6
> Statut global : **J2 — DATASET VALIDÉ**

---

## 1. Projet

**Nom :** AI Sales Forecasting

**Objectif :**

Construire une solution de prévision des ventes permettant de :

1. récupérer les données de ventes KShop ;
2. nettoyer et préparer les données ;
3. analyser les tendances ;
4. créer des variables prédictives ;
5. entraîner un modèle Machine Learning ;
6. prévoir la demande J+1 à J+7 ;
7. produire des recommandations de stock ;
8. présenter les résultats dans un dashboard Streamlit ;
9. préparer le projet pour la production.

---

## 2. Architecture cible

```text
KShop / PostgreSQL
        │
        ▼
Extraction des ventes
        │
        ▼
Data Cleaning
        │
        ▼
EDA / Analyse
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

## 3. Stack technique

| Technologie     | Version / choix                      |
| --------------- | ------------------------------------ |
| Python          | 3.12.14                              |
| Conda           | environnement `ai-sales-forecasting` |
| NumPy           | 2.5.2                                |
| Pandas          | 3.0.5                                |
| Scikit-learn    | 1.9.0                                |
| Matplotlib      | 3.11.0                               |
| Plotly          | 6.9.0                                |
| Streamlit       | 1.63.0                               |
| SQLAlchemy      | 2.0.51                               |
| psycopg2        | 2.9.11                               |
| Joblib          | 1.5.3                                |
| OpenPyXL        | 3.1.5                                |
| Pytest          | 9.0.3                                |
| Jupyter         | Installé                             |
| Database        | PostgreSQL / Neon                    |
| Version control | Git / GitHub                         |

---

## 4. Structure actuelle

```text
ai-sales-forecasting/
│
├── config/
│
├── dashboard/
│
├── data/
│   ├── processed/
│   └── raw/
│       └── kshop_sales_synthetic.csv
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── data/
│   │   └── generate_synthetic_sales.py
│   ├── features/
│   ├── forecasting/
│   ├── models/
│   └── utils/
│
├── tests/
│
├── .env.example
├── .gitignore
├── README.md
├── PROJECT_CHECKPOINT.md
├── requirements.txt
└── src/
    └── main.py
```

---

## 5. Git / GitHub

**Repository :**

`rrabeari/ai-sales-forecasting`

**Branche principale :**

`main`

**Premier commit :**

`07fd765 chore: initialize AI sales forecasting project`

Le repository GitHub est configuré et le premier push a été effectué.

**État Git initial :**

```text
On branch main

Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

---

# 6. J1 — INITIALISATION

**Statut : VALIDÉ ✅**

### Réalisé

* structure du projet créée ;
* environnement Conda créé ;
* Python vérifié ;
* dépendances principales installées ;
* `requirements.txt` créé ;
* `.gitignore` créé ;
* `.env.example` créé ;
* Git initialisé ;
* branche `main` configurée ;
* repository GitHub créé ;
* premier commit effectué ;
* premier push effectué ;
* `src/main.py` testé avec succès.

**Message obtenu :**

```text
AI Sales Forecasting - Project initialized successfully.
```

---

# 7. J2 — DATASET

**Statut : VALIDÉ ✅**

## J2.1 — Définition des règles

**Statut : VALIDÉ ✅**

* Période : `2025-09-01 → 2026-08-31`
* Durée : 365 jours
* Nombre de produits : 14
* Granularité : 1 produit × 1 jour
* Nombre maximal de lignes : 5 110
* Target ML : `quantity`
* Random seed : 42

### Facteurs de demande

* jour de la semaine ;
* mois ;
* popularité produit ;
* tendance ;
* variation contrôlée ;
* jours sans demande.

---

## J2.2 — Générateur synthétique

**Statut : VALIDÉ ✅**

**Fichier :**

```text
src/data/generate_synthetic_sales.py
```

Le générateur produit un dataset synthétique basé sur la structure métier de KShop.

Le caractère synthétique des données est explicitement conservé afin de ne pas présenter des transactions fictives comme des données réelles.

---

## J2.3 — Génération du dataset

**Statut : VALIDÉ ✅**

**Fichier généré :**

```text
data/raw/kshop_sales_synthetic.csv
```

### Résultat

* Lignes : 5 110
* Produits : 14
* Période : `2025-09-01 → 2026-08-31`
* Colonnes : 7

---

## J2.4 — Contrôle qualité initial

**Statut : VALIDÉ ✅**

### Contrôles réalisés

* aucune valeur NULL ;
* aucun doublon ;
* quantités ≥ 0 ;
* prix > 0 ;
* revenus ≥ 0 ;
* cohérence `revenue = quantity × unit_price` ;
* 14 produits ;
* 6 catégories ;
* dates conformes.

---

## J2.5 — Analyse statistique

**Statut : VALIDÉ ✅**

### Résultats principaux

* Quantité totale vendue : **28 076**
* Chiffre d'affaires total : **67 966 700 AR**
* Vente moyenne par jour : **76,92 unités**
* CA moyen par jour : **186 210,14 AR**
* Produit le plus vendu : **Coca-Cola 33cl — 3 995 unités**
* Catégorie avec le plus grand volume : **Boissons — 11 788 unités**
* Jour de la semaine avec la plus forte demande : **Samedi — 5 120 unités**
* Mois avec la plus forte demande : **Décembre 2025 — 2 927 unités**

Les tendances observées sont cohérentes avec les facteurs de demande intégrés au générateur.

---

## J2.6 — Validation finale

**Statut : PASS ✅**

### Contrôles finaux

| Contrôle                          | Résultat |
| --------------------------------- | -------: |
| Fichier présent                   |        ✅ |
| 5 110 lignes                      |        ✅ |
| 7 colonnes                        |        ✅ |
| 14 produits                       |        ✅ |
| 365 jours                         |        ✅ |
| Dates correctes                   |        ✅ |
| Doublons `date + product_id`      |    **0** |
| Valeurs NULL                      |    **0** |
| Quantités négatives               |    **0** |
| Prix invalides                    |    **0** |
| Revenus négatifs                  |    **0** |
| Prix unique par produit           |        ✅ |
| Catégorie unique par produit      |        ✅ |
| `revenue = quantity × unit_price` |        ✅ |
| Valeurs extrêmes contrôlées       |        ✅ |

### Valeurs extrêmes

Le 99e percentile de la quantité vendue est de **17 unités**.

42 observations se situent au-dessus de ce seuil.

Ces observations ont été contrôlées et ne présentent pas d'anomalie évidente. Elles sont conservées afin de représenter les variations naturelles de la demande et pourront être prises en compte lors du Machine Learning.

---

# 8. Résultat final J2

## J2 — DATASET : VALIDÉ ✅

Le dataset synthétique est considéré comme **prêt pour la phase de préparation des données et de Machine Learning**.

### Dataset de référence

```text
Fichier :
data/raw/kshop_sales_synthetic.csv

Lignes :
5 110

Produits :
14

Période :
2025-09-01 → 2026-08-31

Target ML :
quantity

Random seed :
42

Chiffre d'affaires total :
67 966 700 AR
```

---

# 9. PROCHAINE ÉTAPE

# J3 — DATA CLEANING

### Objectifs

1. Charger le dataset ;
2. vérifier les types ;
3. traiter les valeurs manquantes ;
4. vérifier les doublons ;
5. vérifier les valeurs aberrantes ;
6. vérifier la cohérence métier ;
7. standardiser les données ;
8. produire le dataset nettoyé ;
9. effectuer les tests de validation ;
10. créer un nouveau checkpoint.

---

## 10. Règle de travail du projet

> **Définition → Implémentation → Test → Analyse → Validation → Checkpoint → Étape suivante**

---

## 11. État actuel

| Étape                      | Statut       |
| -------------------------- | ------------ |
| J1 — Initialisation        | ✅ VALIDÉ     |
| J2.1 — Règles dataset      | ✅ VALIDÉ     |
| J2.2 — Générateur          | ✅ VALIDÉ     |
| J2.3 — Génération          | ✅ VALIDÉ     |
| J2.4 — Contrôle qualité    | ✅ VALIDÉ     |
| J2.5 — Analyse statistique | ✅ VALIDÉ     |
| J2.6 — Validation finale   | ✅ PASS       |
| **J2 — Dataset**           | **✅ VALIDÉ** |
| J3 — Data Cleaning         | ⏳ À DÉMARRER |

---

**Dernière validation : J2.6 — PASS**

**Prochaine action : J3.1 — Définition du Data Cleaning**
