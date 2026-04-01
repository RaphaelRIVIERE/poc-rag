from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


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

def test_rebuild_succes():
    """POST /rebuild doit retourner 200 et le nombre de chunks indexés."""
    mock_index = MagicMock()
    with patch("api.routes.DATA_FILE") as mock_data_file, \
         patch("api.routes.load_events", return_value=[]) as _, \
         patch("api.routes.events_to_documents", return_value=[]) as _, \
         patch("api.routes.split_documents", return_value=["chunk1", "chunk2", "chunk3"]) as _, \
         patch("api.routes.build_faiss_index", return_value=mock_index) as _, \
         patch("api.routes.save_index") as _:
        mock_data_file.exists.return_value = True
        response = client.post("/rebuild")
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_indexed"] == 3
    assert "succès" in data["message"]


# POST /rebuild — gestion des erreurs

def test_rebuild_fichier_absent_retourne_503():
    """POST /rebuild doit retourner 503 si le fichier de données est absent."""
    with patch("api.routes.DATA_FILE") as mock_data_file:
        mock_data_file.exists.return_value = False
        response = client.post("/rebuild")
    assert response.status_code == 503
    assert "clean_events" in response.json()["detail"].lower()
