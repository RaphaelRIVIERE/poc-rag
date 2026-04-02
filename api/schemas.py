from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class RebuildResponse(BaseModel):
    message: str
    chunks_indexed: int
