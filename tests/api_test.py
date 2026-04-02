import hashlib
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from api.main import app
from api.security import make_verify_api_key, verify_api_key

client = TestClient(app)

VALID_KEY = "test-api-key"
VALID_KEY_HASH = hashlib.sha256(VALID_KEY.encode()).hexdigest()


@pytest.fixture
def valid_auth():
    app.dependency_overrides[verify_api_key] = make_verify_api_key(VALID_KEY_HASH)
    yield
    app.dependency_overrides.clear()

# Sécurité — api/security.py
def test_verify_api_key_non_configuree_retourne_500():
    """verify() doit retourner 500 si API_KEY n'est pas configurée côté serveur."""
    from api.security import make_verify_api_key
    from fastapi import HTTPException

    verify = make_verify_api_key(None)
    with pytest.raises(HTTPException) as exc_info:
        verify("une-cle-quelconque")
    assert exc_info.value.status_code == 500

# GET /health
def test_health_retourne_ok():
    """GET /health doit retourner {"status": "ok"} avec un code 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# POST /ask — cas nominaux

def test_ask_retourne_une_reponse():
    """POST /ask doit retourner une réponse générée quand ask() est mockée."""
    with patch("api.routes.ask", return_value="Il y a un concert de jazz à Paris le 15 avril."):
        response = client.post("/ask", json={"question": "Quels concerts à Paris ?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Il y a un concert de jazz à Paris le 15 avril."}


# POST /ask — gestion des erreurs

def test_ask_question_vide_retourne_422():
    """POST /ask doit retourner 422 si la question est vide."""
    with patch("api.routes.ask", side_effect=ValueError("La question ne peut pas être vide.")):
        response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_champ_manquant_retourne_422():
    """POST /ask doit retourner 422 si le champ 'question' est absent."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_index_absent_retourne_503():
    """POST /ask doit retourner 503 si l'index FAISS est introuvable."""
    with patch("api.routes.ask", side_effect=FileNotFoundError("index not found")):
        response = client.post("/ask", json={"question": "Quels événements ?"})
    assert response.status_code == 503
    assert "rebuild" in response.json()["detail"].lower()


def test_ask_rate_limit_retourne_429():
    """POST /ask doit retourner 429 si Mistral retourne une erreur de rate limit."""
    with patch("api.routes.ask", side_effect=Exception("Error response 429 rate_limit exceeded")):
        response = client.post("/ask", json={"question": "Quels événements ?"})
    assert response.status_code == 429
    assert "réessayer" in response.json()["detail"].lower()


def test_ask_erreur_serveur_retourne_500():
    """POST /ask doit retourner 500 en cas d'erreur inattendue."""
    with patch("api.routes.ask", side_effect=RuntimeError("Erreur inconnue")):
        response = client.post("/ask", json={"question": "Quels événements ?"})
    assert response.status_code == 500


# POST /rebuild — cas nominaux

def test_rebuild_succes(valid_auth):
    """POST /rebuild doit retourner 200 et le nombre de chunks indexés."""
    mock_index = MagicMock()
    with patch("api.routes.DATA_FILE") as mock_data_file, \
         patch("api.routes.load_events", return_value=[]) as _, \
         patch("api.routes.events_to_documents", return_value=[]) as _, \
         patch("api.routes.split_documents", return_value=["chunk1", "chunk2", "chunk3"]) as _, \
         patch("api.routes.build_faiss_index", return_value=mock_index) as _, \
         patch("api.routes.save_index") as _:
        mock_data_file.exists.return_value = True
        response = client.post("/rebuild", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_indexed"] == 3
    assert "succès" in data["message"]


# POST /rebuild — authentification

def test_rebuild_sans_header_retourne_401(valid_auth):
    """POST /rebuild doit retourner 401 si le header X-API-Key est absent."""
    response = client.post("/rebuild")
    assert response.status_code == 401


def test_rebuild_cle_invalide_retourne_401(valid_auth):
    """POST /rebuild doit retourner 401 si la clé API est incorrecte."""
    response = client.post("/rebuild", headers={"X-API-Key": "mauvaise-cle"})
    assert response.status_code == 401


# POST /rebuild — gestion des erreurs

def test_rebuild_erreur_reconstruction_retourne_500(valid_auth):
    """POST /rebuild doit retourner 500 si la reconstruction de l'index échoue."""
    mock_index = MagicMock()
    with patch("api.routes.DATA_FILE") as mock_data_file, \
         patch("api.routes.load_events", return_value=[]), \
         patch("api.routes.events_to_documents", return_value=[]), \
         patch("api.routes.split_documents", return_value=[]), \
         patch("api.routes.build_faiss_index", side_effect=RuntimeError("Erreur FAISS")):
        mock_data_file.exists.return_value = True
        response = client.post("/rebuild", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 500
    assert "reconstruction" in response.json()["detail"].lower()


def test_rebuild_fichier_absent_retourne_503(valid_auth):
    """POST /rebuild doit retourner 503 si le fichier de données est absent."""
    with patch("api.routes.DATA_FILE") as mock_data_file:
        mock_data_file.exists.return_value = False
        response = client.post("/rebuild", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 503
    assert "clean_events" in response.json()["detail"].lower()
