import hashlib
import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def make_verify_api_key(expected_hash):
    """Fabrique une dépendance FastAPI vérifiant le header X-API-Key contre un hash SHA-256."""
    def verify(x_api_key: str | None = Security(_api_key_scheme)) -> None:
        if not expected_hash:
            raise HTTPException(status_code=500, detail="API_KEY non configurée sur le serveur.")
        if not x_api_key or not hmac.compare_digest(
            hashlib.sha256(x_api_key.encode()).hexdigest(), expected_hash
        ):
            raise HTTPException(status_code=401, detail="Clé API manquante ou invalide.")
    return verify


verify_api_key = make_verify_api_key(os.getenv("API_KEY"))
