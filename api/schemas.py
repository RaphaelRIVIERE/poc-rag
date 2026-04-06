from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        description="Question en langage naturel sur les événements culturels.",
        examples=["Quels concerts ont lieu à Paris ce week-end ?"],
    )


class AskResponse(BaseModel):
    answer: str = Field(..., description="Réponse générée par le modèle Mistral, augmentée par les documents retrouvés.")


class RebuildResponse(BaseModel):
    message: str = Field(..., description="Confirmation de la reconstruction.")
    chunks_indexed: int = Field(..., description="Nombre de chunks de texte indexés dans FAISS.")


class MetadataResponse(BaseModel):
    total_events: int = Field(..., description="Nombre total d'événements dans le fichier de données.")
    last_rebuilt: str | None = Field(None, description="Date ISO 8601 de la dernière reconstruction de l'index FAISS. `null` si l'index n'existe pas encore.")
    first_event_date: str | None = Field(None, description="Date de début du premier événement indexé (format ISO 8601).")
    last_event_date: str | None = Field(None, description="Date de fin du dernier événement indexé (format ISO 8601).")
    departments: list[str] = Field(..., description="Liste des départements couverts (triée alphabétiquement).")
    districts: list[str] = Field(..., description="Liste des arrondissements couverts (triée alphabétiquement).")
