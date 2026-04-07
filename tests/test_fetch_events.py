"""Tests pour scripts/fetch_events.py — pagination et gestion d'erreurs HTTP."""

import pytest
from unittest.mock import patch, MagicMock

from scripts.fetch_events import fetch_events


def make_mock_response(results, status_code=200, raise_error=False):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {"results": results}
    if raise_error:
        mock_resp.raise_for_status.side_effect = Exception(f"HTTP Error {status_code}")
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


def test_fetch_events_retourne_les_evenements():
    mock_batch = [{"uid": "event-001", "title_fr": "Concert"}, {"uid": "event-002", "title_fr": "Expo"}]

    # Première page : 2 événements, deuxième page : vide (fin de pagination)
    with patch("scripts.fetch_events.requests.get") as mock_get:
        mock_get.side_effect = [
            make_mock_response(mock_batch),
            make_mock_response([]),
        ]
        result = fetch_events()

    assert len(result) == 2
    assert result[0]["uid"] == "event-001"
    assert result[1]["uid"] == "event-002"


def test_fetch_events_sarrette_si_batch_vide():
    with patch("scripts.fetch_events.requests.get") as mock_get:
        mock_get.return_value = make_mock_response([])
        result = fetch_events()

    assert result == []
    mock_get.assert_called_once()  # un seul appel, pas de pagination inutile


def test_fetch_events_leve_erreur_si_http_echec():
    with patch("scripts.fetch_events.requests.get") as mock_get:
        mock_get.return_value = make_mock_response([], status_code=500, raise_error=True)
        with pytest.raises(Exception):
            fetch_events()
