from fastapi import APIRouter, HTTPException

from api.schemas import AskRequest, AskResponse, RebuildResponse
from scripts.rag_chain import ask
from scripts.build_index import (
    load_events,
    events_to_documents,
    split_documents,
    build_faiss_index,
    save_index,
    DATA_FILE,
    INDEX_DIR,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest):
    """Pose une question au système RAG et retourne la réponse générée."""
    try:
        answer = ask(body.question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Index FAISS introuvable. Lancez d'abord POST /rebuild.",
        )
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            raise HTTPException(
                status_code=429,
                detail="Limite de requêtes Mistral atteinte. Veuillez réessayer dans quelques secondes.",
            )
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")
    return AskResponse(answer=answer)


@router.post("/rebuild", response_model=RebuildResponse)
def rebuild_index():
    """Reconstruit l'index vectoriel FAISS à partir des données nettoyées."""
    if not DATA_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Fichier de données introuvable : {DATA_FILE}. Lancez d'abord scripts/clean_events.py.",
        )
    try:
        events = load_events(DATA_FILE)
        docs = events_to_documents(events)
        chunks = split_documents(docs)
        index = build_faiss_index(chunks)
        save_index(index, INDEX_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la reconstruction : {str(e)}")
    return RebuildResponse(
        message="Index FAISS reconstruit avec succès.",
        chunks_indexed=len(chunks),
    )
