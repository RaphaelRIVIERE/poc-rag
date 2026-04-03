from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class RebuildResponse(BaseModel):
    message: str
    chunks_indexed: int


class MetadataResponse(BaseModel):
    total_events: int
    departments: list[str]
    last_rebuilt: str | None
