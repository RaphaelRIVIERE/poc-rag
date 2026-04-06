from fastapi import APIRouter, Depends, HTTPException

from api.security import verify_api_key

from api.schemas import AskRequest, AskResponse, RebuildResponse, MetadataResponse
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

@router.get(
    "/health",
    tags=["Système"],
    summary="Vérifier l'état de l'API",
    description="Retourne `ok` si l'API est opérationnelle. Utile pour les health checks (load balancer, monitoring).",
    responses={200: {"content": {"application/json": {"example": {"status": "ok"}}}}},
)
def health():
    return {"status": "ok"}


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    tags=["Système"],
    summary="Consulter les métadonnées de l'index",
    description=(
        "Retourne des informations sur la base d'événements actuellement indexée : "
        "nombre total d'événements, liste des départements et arrondissements couverts, "
        "et date de la dernière reconstruction de l'index FAISS."
    ),
)
def get_metadata():
    """Retourne des informations sur la base d'événements indexée."""
    import json
    from datetime import datetime

    if not DATA_FILE.exists():
        raise HTTPException(status_code=503, detail="Fichier de données introuvable.")

    events = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    departments = sorted({e.get("location_dept", "") for e in events if e.get("location_dept")})
    districts = sorted({e.get("location_district", "") for e in events if e.get("location_district")})

    begin_dates = [e["firstdate_begin"] for e in events if e.get("firstdate_begin")]
    end_dates = [e["lastdate_end"] for e in events if e.get("lastdate_end")]
    first_event_date = min(begin_dates) if begin_dates else None
    last_event_date = max(end_dates) if end_dates else None

    last_rebuilt = None
    if INDEX_DIR.exists():
        last_rebuilt = datetime.fromtimestamp(INDEX_DIR.stat().st_mtime).isoformat(timespec="seconds")

    return MetadataResponse(
        total_events=len(events),
        last_rebuilt=last_rebuilt,
        first_event_date=first_event_date,
        last_event_date=last_event_date,
        departments=departments,
        districts=districts
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["RAG"],
    summary="Poser une question au système RAG",
    description=(
        "Envoie une question en langage naturel au pipeline RAG. "
        "Le système recherche les événements les plus pertinents dans l'index FAISS, "
        "puis génère une réponse augmentée via Mistral.\n\n"
        "**Exemples de questions :**\n"
        "- *Quels concerts ont lieu à Paris ce week-end ?*\n"
        "- *Y a-t-il des expositions gratuites à Lyon en mai ?*"
    ),
    responses={
        503: {"description": "Index FAISS introuvable — lancer d'abord POST /rebuild."},
        429: {"description": "Limite de requêtes Mistral atteinte."},
    },
)
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


@router.post(
    "/rebuild",
    response_model=RebuildResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Système"],
    summary="Reconstruire l'index FAISS",
    description=(
        "Recharge les événements depuis le fichier de données nettoyées, "
        "génère les embeddings Mistral et reconstruit l'index FAISS.\n\n"
        "> **Authentification requise** : fournir la clé API dans le header `X-API-Key`.\n\n"
        "Prérequis : le fichier de données doit exister (lancer d'abord `scripts/clean_events.py`)."
    ),
    responses={
        503: {"description": "Fichier de données introuvable."},
    },
)
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
