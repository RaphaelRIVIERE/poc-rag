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
    description="API de recommandation d'événements culturels basée sur un système RAG (FAISS + Mistral).",
    version="0.1.0",
)

app.include_router(router)
