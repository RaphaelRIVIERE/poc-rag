"""Tests fonctionnels de l'API FastAPI — /ask, /rebuild, /health, /metadata."""

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


def test_verify_api_key_non_configuree_retourne_500():
    from api.security import make_verify_api_key
    from fastapi import HTTPException

    verify = make_verify_api_key(None)
    with pytest.raises(HTTPException) as exc_info:
        verify("une-cle-quelconque")
    assert exc_info.value.status_code == 500


def test_health_retourne_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_retourne_une_reponse():
    with patch("api.routes.ask", return_value="Il y a un concert de jazz à Paris le 15 avril."):
        response = client.post("/ask", json={"question": "Quels concerts à Paris ?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Il y a un concert de jazz à Paris le 15 avril."}


def test_ask_question_vide_retourne_422():
    # La validation se fait dans rag_chain.ask(), pas besoin de mocker
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_champ_manquant_retourne_422():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_index_absent_retourne_503():
    """Retourne 503 si l'index FAISS n'a pas encore été construit."""
    with patch("api.routes.ask", side_effect=FileNotFoundError("index not found")):
        response = client.post("/ask", json={"question": "Quels événements ?"})
    assert response.status_code == 503
    assert "rebuild" in response.json()["detail"].lower()


def test_ask_rate_limit_retourne_429():
    """Retourne 429 si Mistral dépasse le rate limit.
    Le message "429 rate_limit exceeded" est le format renvoyé par l'API Mistral —
    la route le détecte pour distinguer ce cas d'une erreur serveur générique."""
    with patch("api.routes.ask", side_effect=Exception("Error response 429 rate_limit exceeded")):
        response = client.post("/ask", json={"question": "Quels événements ?"})
    assert response.status_code == 429
    assert "réessayer" in response.json()["detail"].lower()


def test_ask_erreur_serveur_retourne_500():
    with patch("api.routes.ask", side_effect=RuntimeError("Erreur inconnue")):
        response = client.post("/ask", json={"question": "Quels événements ?"})
    assert response.status_code == 500


def test_metadata_fichier_absent_retourne_503():
    with patch("api.routes.DATA_FILE") as mock_data_file:
        mock_data_file.exists.return_value = False
        response = client.get("/metadata")
    assert response.status_code == 503


def test_metadata_sans_index_retourne_last_rebuilt_none():
    mock_events = [{"location_dept": "Paris"}]
    with patch("api.routes.DATA_FILE") as mock_data_file, \
         patch("api.routes.INDEX_DIR") as mock_index_dir, \
         patch("json.loads", return_value=mock_events):
        mock_data_file.exists.return_value = True
        mock_data_file.read_text.return_value = "[]"
        mock_index_dir.exists.return_value = False
        response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["last_rebuilt"] is None


def test_rebuild_succes(valid_auth):
    """Vérifie le flux complet de /rebuild : chargement → conversion → chunking → indexation → sauvegarde.
    Chaque étape est mockée indépendamment pour isoler la logique de la route."""
    with patch("api.routes.DATA_FILE") as mock_data_file, \
         patch("api.routes.load_events", return_value=[]), \
         patch("api.routes.events_to_documents", return_value=[]), \
         patch("api.routes.split_documents", return_value=["chunk1", "chunk2", "chunk3"]), \
         patch("api.routes.build_faiss_index", return_value=MagicMock()), \
         patch("api.routes.save_index"):
        mock_data_file.exists.return_value = True
        response = client.post("/rebuild", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_indexed"] == 3
    assert "succès" in data["message"]


def test_rebuild_sans_header_retourne_401(valid_auth):
    response = client.post("/rebuild")
    assert response.status_code == 401


def test_rebuild_cle_invalide_retourne_401(valid_auth):
    response = client.post("/rebuild", headers={"X-API-Key": "mauvaise-cle"})
    assert response.status_code == 401


def test_rebuild_erreur_reconstruction_retourne_500(valid_auth):
    """Retourne 500 si la construction de l'index FAISS échoue."""
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
    with patch("api.routes.DATA_FILE") as mock_data_file:
        mock_data_file.exists.return_value = False
        response = client.post("/rebuild", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 503
    assert "clean_events" in response.json()["detail"].lower()
