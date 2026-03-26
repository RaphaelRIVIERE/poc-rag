# POC RAG — Assistant de recommandation d'événements culturels

Proof of Concept d'un chatbot intelligent basé sur un système RAG (Retrieval-Augmented Generation) pour la plateforme **Puls-Events**.

## Structure du projet

```
poc-rag/
├── api/              # API REST FastAPI
├── scripts/          # Scripts de collecte, nettoyage et indexation
│   ├── fetch_events.py   # Récupération des événements Open Agenda
│   ├── clean_events.py   # Nettoyage et structuration des données
│   └── build_index.py    # Vectorisation et construction de l'index FAISS
├── tests/            # Tests unitaires
│   ├── test_preprocessing.py
│   └── test_build_index.py
├── data/             # Données brutes et nettoyées (non versionnées)
├── index/            # Index FAISS sauvegardé (non versionné)
├── docs/             # Rapport technique et présentation
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

### 1. Cloner le dépôt

```bash
git clone <repo-url>
cd poc-rag
```

### 2. Créer l'environnement virtuel

```bash
python3 -m venv env
source env/bin/activate   # Linux/Mac
# env\Scripts\activate    # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env et renseigner vos clés API
```

| Variable | Description |
|---|---|
| `MISTRAL_API_KEY` | Clé API Mistral AI |
| `OPENAGENDA_API_KEY` | Clé API OpenAgenda |

### 5. Vérifier l'installation

```bash
python scripts/test_imports.py
```

---

## Utilisation

### Récupérer les événements

```bash
python scripts/fetch_events.py
```

Récupère jusqu'à 1000 événements en Île-de-France depuis l'API Open Agenda et les sauvegarde dans `data/raw_events.json`.

### Nettoyer les données

```bash
python scripts/clean_events.py
```

Nettoie et structure les données brutes, produit `data/clean_events.json`.

### Construire l'index vectoriel

```bash
python scripts/build_index.py
```

Génère les embeddings via Mistral AI et sauvegarde l'index FAISS dans `index/faiss_index/`.

### Lancer les tests

```bash
pytest tests/
```

### Mesurer la couverture des tests

```bash
pytest tests/ --cov=scripts --cov-report=term-missing
```

Affiche la couverture ligne par ligne pour chaque script. Pour générer un rapport HTML :

```bash
pytest tests/ --cov=scripts --cov-report=html
# Ouvrir htmlcov/index.html dans un navigateur
```

> Les fichiers `.coverage` et `htmlcov/` sont exclus du dépôt (`.gitignore`).
