# POC RAG — Assistant de recommandation d'événements culturels

Proof of Concept d'un chatbot intelligent basé sur un système RAG (Retrieval-Augmented Generation) pour la plateforme **Puls-Events**.

## Structure du projet

```
poc-rag/
├── scripts/        # Scripts de collecte, nettoyage et indexation
├── api/            # API REST FastAPI (à venir)
├── tests/          # Tests unitaires et fonctionnels (à venir)
├── data/           # Données (non versionnées)
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
