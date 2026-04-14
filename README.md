# POC RAG — Assistant de recommandation d'événements culturels

![CI](https://github.com/RaphaelRIVIERE/poc-rag/actions/workflows/ci.yml/badge.svg)

Proof of Concept d'un chatbot intelligent basé sur un système RAG (Retrieval-Augmented Generation) pour la plateforme **Puls-Events**.

> Le rapport technique complet est disponible dans [docs/rapport_technique.md](docs/rapport_technique.md).

## Démarrage rapide

```bash
cp .env.example .env   # renseigner MISTRAL_API_KEY et API_KEY
docker compose up
```

Lance le pipeline complet (récupération → nettoyage → indexation) puis démarre l'API sur `http://localhost:8000`.


## Prérequis

- Python 3.10+
- Docker et Docker Compose
- Une clé API [Mistral AI](https://console.mistral.ai/)


## Installation (mode développement)

```bash
git clone <repo-url>
cd poc-rag
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Variables d'environnement

| Variable | Description |
|---|---|
| `MISTRAL_API_KEY` | Clé API Mistral AI |
| `API_KEY` | Hash SHA-256 de la clé protégeant `/rebuild` |

Pour générer le hash :

```bash
python3 -c "import hashlib; print(hashlib.sha256(b'votre-cle-secrete').hexdigest())"
```


## Utilisation

### Pipeline de données

```bash
python scripts/fetch_events.py   # récupère les événements Open Agenda
python scripts/clean_events.py   # nettoie et structure les données
python scripts/build_index.py    # construit l'index FAISS
```

Ou en une commande :

```bash
make data
```

### Lancer l'API

```bash
uvicorn api.main:app --reload
```

### Endpoints

| Route | Méthode | Auth | Description |
|---|---|---|---|
| `/health` | GET | Non | Vérifie que l'API est opérationnelle |
| `/ask` | POST | Non | Pose une question, retourne une réponse augmentée |
| `/rebuild` | POST | Oui (`X-API-Key`) | Reconstruit l'index vectoriel FAISS |
| `/metadata` | GET | Non | Informations sur la base indexée (nb événements, départements…) |
| `/docs` | GET | Non | Documentation Swagger interactive |

### Exemples

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels événements jazz à Paris cette semaine ?"}'
```

## Tests

```bash
pytest tests/                                          # tests unitaires et fonctionnels
pytest tests/ --cov=scripts --cov=api                  # avec couverture
python tests/evaluate_rag.py                           # évaluation Ragas
```


## Structure du projet

```
poc-rag/
├── api/
│   ├── main.py          # Point d'entrée FastAPI
│   ├── routes.py        # Définition des endpoints (/ask, /rebuild, /health, /metadata)
│   ├── schemas.py       # Modèles Pydantic (requêtes/réponses)
│   └── security.py      # Vérification clé API (X-API-Key)
│
├── scripts/
│   ├── fetch_events.py  # Récupération des événements via API OpenDataSoft
│   ├── clean_events.py  # Nettoyage et normalisation des données brutes
│   ├── build_index.py   # Chunking, embeddings Mistral, construction index FAISS
│   ├── rag_chain.py     # Pipeline RAG : retrieval FAISS + génération Mistral
│   └── show_eval.py     # Affichage des résultats d'évaluation Ragas
│
├── tests/
│   ├── annotated_qa.json       # 12 questions/réponses annotées manuellement
│   ├── evaluate_rag.py         # Évaluation automatique Ragas
│   ├── test_fetch_events.py    # Tests unitaires fetch_events
│   ├── test_preprocessing.py   # Tests unitaires clean_events
│   ├── test_build_index.py     # Tests unitaires build_index
│   ├── test_rag_chain.py       # Tests unitaires rag_chain
│   └── api_test.py             # Tests fonctionnels de l'API
│
├── docs/
│   └── demo.postman_collection.json  # Collection Postman
│
├── data/                # Données brutes et nettoyées (non versionné — .gitignore)
├── index/               # Index FAISS persisté (non versionné — .gitignore)
├── results/             # Résultats d'évaluation JSON
│
├── .github/
│   └── workflows/
│       └── ci.yml       # Pipeline CI (tests unitaires + évaluation Ragas)
│
├── docker-compose.yml   # Lancement du pipeline complet (build index + API)
├── Dockerfile           # Image Docker pour l'API
├── Makefile             # Commandes raccourcies (build, run, test...)
├── requirements.txt     # Dépendances Python
├── conftest.py          # Configuration pytest
├── .env.example         # Template variables d'environnement
└── README.md            # Documentation de démarrage rapide
```
