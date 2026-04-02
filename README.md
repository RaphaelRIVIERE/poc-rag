# POC RAG — Assistant de recommandation d'événements culturels

Proof of Concept d'un chatbot intelligent basé sur un système RAG (Retrieval-Augmented Generation) pour la plateforme **Puls-Events**.

## Structure du projet

```
poc-rag/
├── api/
│   ├── main.py           # Point d'entrée FastAPI
│   ├── routes.py         # Endpoints (/ask, /health, /rebuild)
│   ├── schemas.py        # Modèles Pydantic (entrées/sorties)
│   └── security.py       # Authentification par clé API (X-API-Key)
├── scripts/
│   ├── fetch_events.py   # Récupération des événements Open Agenda
│   ├── clean_events.py   # Nettoyage et structuration des données
│   ├── build_index.py    # Vectorisation et construction de l'index FAISS
│   └── rag_chain.py      # Pipeline RAG (FAISS + Mistral)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_build_index.py
│   ├── test_rag_chain.py
│   ├── api_test.py       # Tests fonctionnels des endpoints
│   ├── evaluate_rag.py   # Évaluation automatique avec Ragas
│   └── annotated_qa.json # Jeu de test annoté (questions/réponses)
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
| `API_KEY` | Hash SHA-256 de la clé protégeant l'endpoint `/rebuild` |

Pour générer le hash de votre clé :

```bash
python3 -c "import hashlib; print(hashlib.sha256(b'votre-cle-secrete').hexdigest())"
```

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

### Lancer l'API

```bash
uvicorn api.main:app --reload
```

L'API est accessible sur `http://localhost:8000`. La documentation interactive Swagger est disponible sur `http://localhost:8000/docs`.

### Utiliser l'API

```bash
# Vérifier que l'API est opérationnelle
curl http://localhost:8000/health

# Poser une question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels événements jazz à Paris cette semaine ?"}'

# Reconstruire l'index (nécessite le header X-API-Key)
curl -X POST http://localhost:8000/rebuild \
  -H "X-API-Key: votre-cle-secrete"
```

### Endpoints

| Route | Méthode | Auth | Description |
|---|---|---|---|
| `/health` | GET | Non | Vérifie que l'API est opérationnelle |
| `/ask` | POST | Non | Pose une question, retourne une réponse augmentée |
| `/rebuild` | POST | Oui (`X-API-Key`) | Reconstruit l'index vectoriel FAISS |
| `/docs` | GET | Non | Documentation Swagger interactive |

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

### Évaluation des réponses

```bash
python tests/evaluate_rag.py
```

Lance l'évaluation automatique du système RAG avec Ragas sur le jeu de test annoté (`tests/annotated_qa.json`).
