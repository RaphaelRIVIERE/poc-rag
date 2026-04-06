"""
main.py — Point d'entrée de l'API FastAPI Puls-Events.

Lancement :
    uvicorn api.main:app --reload

Documentation interactive :
    http://localhost:8000/docs
"""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API de recommandation d'événements culturels basée sur un système RAG.\n\n"
        "Elle combine une recherche vectorielle **FAISS** et le modèle de langage **Mistral** "
        "pour répondre à des questions en langage naturel sur des événements culturels."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "Système", "description": "Endpoints de supervision et d'administration."},
        {"name": "RAG", "description": "Endpoints du pipeline de question-réponse."},
    ],
)

app.include_router(router)
