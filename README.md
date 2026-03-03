# Pipeline MLOps – Prédiction du Risque de Crédit

Projet complet de **Machine Learning orienté production**, mettant en place une pipeline MLOps reproductible pour la prédiction du risque de crédit.

Ce projet démontre :

*  Versioning des données avec DVC
*  Entraînement et évaluation automatisés
*  Contrôle qualité des métriques (quality gate)
*  Pipeline ML reproductible
*  API déployable
*  Containerisation avec Docker
*  Intégration continue via GitHub Actions

---

##  Objectif du Projet

Construire un modèle de **prédiction du risque de crédit** à partir de données clients et historiques de paiement, en respectant des standards proches de la production.

La pipeline inclut :

1. Prétraitement des données
2. Séparation train / test
3. Entraînement du modèle
4. Évaluation des performances
5. Validation automatique des métriques
6. Exposition du modèle via API
7. Containerisation Docker
8. Validation automatique via CI

---

##  Architecture du Projet

```text
mlops-credit-risk-pipeline/
│
├── data/
│   ├── raw/                # Données brutes (suivies par DVC)
│   ├── processed/          # Données prétraitées
│   └── split/              # Jeux train / test
│
├── models/
│   ├── model.pkl           # Modèle entraîné
│   └── metrics.json        # Métriques d’évaluation
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── api.py              # Application FastAPI
│
├── scripts/
│   └── check_metrics.py
│
├── dvc.yaml                # Définition de la pipeline DVC
├── Dockerfile              # Image Docker
└── .github/workflows/      # Workflows CI
```

---

## Pipeline Reproductible

Le projet utilise **DVC** pour :

* Versionner les données
* Reproduire les expériences
* Garantir la traçabilité

Exécuter toute la pipeline :

```bash
dvc pull
dvc repro
```

Étapes définies :

* `preprocess`
* `train`
* `evaluate`

Chaque étape produit des artefacts versionnés.

---

##  Entraînement du Modèle

L’étape d’entraînement :

* Sépare les données avec stratification
* Applique un `StandardScaler`
* Entraîne le modèle
* Sauvegarde :

  * `model.pkl`
  * `X_test.csv`
  * `y_test.csv`

---

##  Évaluation & Quality Gate

Après l’entraînement :

* Génération des prédictions
* Calcul des métriques
* Sauvegarde dans `metrics.json`

Un script `check_metrics.py` :

* Vérifie que les performances dépassent un seuil défini
* Stoppe la pipeline si les performances sont insuffisantes

Cela simule un **contrôle qualité en production**.

---

##  API du Modèle

Le modèle est exposé via une API développée avec **FastAPI**.

Endpoints disponibles :

* `POST /predict`

Lancement local :

```bash
uvicorn src.api:app --reload
```

Documentation interactive :

```
http://localhost:8000/docs
```

---

## Containerisation

L’API est containerisée avec **Docker**.

Construire l’image :

```bash
docker build -t credit-api .
```

Lancer le container :

```bash
docker run -p 8000:8000 credit-api
```

Accès :

```
http://localhost:8080/docs
```

---

## Intégration Continue (CI)

Deux workflows GitHub Actions sont mis en place :

### 🔹 Pull Request (validation)

* Exécution de la pipeline DVC
* Vérification des métriques
* Build Docker

Si une étape échoue → le merge est bloqué.

---

### 🔹 Push sur `main`

* Tests unitaires (pytest)
* Lint (flake8)
* Vérification qualité du code

---

## Tests

Lancer les tests :

```bash
pytest
```

---

## Concepts MLOps Implémentés

* Versioning des données
* Pipeline ML reproductible
* Validation automatique des performances
* Quality gate
* CI/CD appliqué au ML
* API containerisée
* Architecture orientée production

---

##  Améliorations Futures

* Optimisation d’hyperparamètres (Optuna)
* Suivi d’expérimentations
* Model registry
* Monitoring des prédictions
* Déploiement cloud
* A/B testing

---

## Pourquoi ce projet ?

Ce projet illustre une approche **pragmatique et professionnelle du Machine Learning**, au-delà d’un simple notebook :

* Industrialisation d’un modèle
* Automatisation des contrôles
* Reproductibilité
* Déploiement prêt pour la production

---
## ⚠️ Remarque concernant DVC et le stockage distant

Actuellement, **DVC n’est pas configuré avec un stockage distant (remote cloud)**.

Cela signifie que :

* `dvc pull` ne fonctionne pas dans l’environnement CI
* Les données versionnées ne sont pas accessibles automatiquement depuis GitHub Actions
* Les Pull Requests ne peuvent pas valider entièrement la pipeline avec les données distantes

Cependant :

* La structure DVC est entièrement en place
* Les fichiers `dvc.yaml` et `.dvc` sont correctement configurés
* Il suffirait d’ajouter un remote (ex: S3, Azure Blob, GCS, etc.) pour activer un fonctionnement complet en environnement cloud

Le projet est donc prêt pour une intégration cloud.

---

 Projet réalisé dans une démarche d’apprentissage approfondi des pratiques MLOps.

---
