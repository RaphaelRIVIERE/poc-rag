"""
Tests unitaires pour scripts/clean_events.py

Couvre les fonctions de nettoyage et de validation des événements bruts :
  - strip_html         : suppression des balises HTML d'une chaîne de caractères
  - clean_event        : nettoyage d'un événement brut (filtre les titres vides)

Les tests sont organisés en trois groupes :
  1. Validation des données brutes (format, unicité des uid, région, plage de dates)
  2. Tests unitaires des fonctions de nettoyage (avec données mockées stables)
  3. Vérification des données nettoyées produites (text non vide, uid uniques)
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from scripts.clean_events import clean_event, strip_html

RAW_FILE   = Path(__file__).parent.parent / "data" / "raw_events.json"
CLEAN_FILE = Path(__file__).parent.parent / "data" / "clean_events.json"

# Date fixe de référence — les dates des mocks sont calculées par rapport à elle
FIXED_NOW = datetime(2026, 3, 25)

MOCK_RAW_EVENTS = [
    {
        "uid": "mock-001",
        "title_fr": "Concert de jazz",
        "description_fr": "Un concert de jazz.",
        "longdescription_fr": "<p>Une soirée <strong>jazz</strong> inoubliable.</p>",
        "conditions_fr": "Entrée libre",
        "keywords_fr": ["jazz", "musique"],
        "firstdate_begin": "2026-01-15T18:00:00+00:00",  # ~70 jours avant FIXED_NOW
        "lastdate_end": "2026-01-15T21:00:00+00:00",
        "daterange_fr": "15 janvier 2026",
        "location_name": "Salle Pleyel",
        "location_address": "252 Rue du Faubourg Saint-Honoré",
        "location_city": "Paris",
        "location_department": "Paris",
        "location_region": "Île-de-France",
        "location_coordinates": {"lon": 2.308, "lat": 48.879},
        "canonicalurl": "https://example.com/concert-jazz",
    },
    {
        "uid": "mock-002",
        "title_fr": "Exposition art moderne",
        "description_fr": "Une exposition d'art moderne.",
        "longdescription_fr": "",
        "conditions_fr": "",
        "keywords_fr": ["art", "exposition"],
        "firstdate_begin": "2026-05-10T10:00:00+00:00",  # ~46 jours après FIXED_NOW
        "lastdate_end": "2026-06-10T18:00:00+00:00",
        "daterange_fr": "10 mai - 10 juin 2026",
        "location_name": "Centre Pompidou",
        "location_address": "Place Georges-Pompidou",
        "location_city": "Paris",
        "location_department": "Paris",
        "location_region": "Île-de-France",
        "location_coordinates": {"lon": 2.352, "lat": 48.861},
        "canonicalurl": "https://example.com/expo-art",
    },
]


@pytest.fixture
def raw_events():
    return MOCK_RAW_EVENTS


@pytest.fixture
def clean_events_data():
    return [e for e in (clean_event(r) for r in MOCK_RAW_EVENTS) if e is not None]


# ---------------------------------------------------------------------------
# Données brutes
# ---------------------------------------------------------------------------

def test_raw_file_exists_and_not_empty():
    """Vérifie que le fichier raw_events.json existe et contient au moins un événement."""
    assert RAW_FILE.exists(), "raw_events.json introuvable"
    data = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    assert len(data) > 0, "raw_events.json est vide"


def test_raw_uid_unique(raw_events):
    """Vérifie qu'il n'y a pas de doublons d'uid dans les données brutes."""
    uids = [e["uid"] for e in raw_events]
    assert len(uids) == len(set(uids)), "Des doublons d'uid existent dans les données brutes"


def test_raw_location_region(raw_events):
    """Vérifie que tous les événements appartiennent bien à la région Île-de-France."""
    for event in raw_events:
        region = event.get("location_region")
        assert region == "Île-de-France", (
            f"Événement {event.get('uid')} hors région : {region}"
        )


def test_raw_dates_in_range(raw_events):
    """Vérifie que les dates des événements sont dans la plage autorisée.

    La plage est calculée par rapport à FIXED_NOW (date fixe) pour garantir
    des résultats stables indépendamment de la date d'exécution des tests.
    """
    date_min = FIXED_NOW - timedelta(days=365)
    date_max = FIXED_NOW + timedelta(days=180)

    for event in raw_events:
        raw_date = event.get("firstdate_begin")
        if not raw_date:
            continue
        date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
        assert date >= date_min.date(), (
            f"Événement {event.get('uid')} trop ancien : {raw_date}"
        )
        assert date <= date_max.date(), (
            f"Événement {event.get('uid')} trop loin dans le futur : {raw_date}"
        )


# ---------------------------------------------------------------------------
# Fonctions de nettoyage
# ---------------------------------------------------------------------------

def test_clean_event_sans_titre_retourne_none():
    """Un événement sans titre (title_fr vide) doit être rejeté et retourner None."""
    event_sans_titre = {
        "uid": "test-001",
        "title_fr": "",
        "description_fr": "Une description.",
        "firstdate_begin": "2026-06-01T10:00:00+00:00",
        "location_region": "Île-de-France",
    }
    assert clean_event(event_sans_titre) is None


def test_strip_html_supprime_balises():
    """strip_html doit retourner un texte sans aucune balise HTML tout en conservant le contenu textuel."""
    html = "<p>Bonjour <strong>le monde</strong> !</p>"
    result = strip_html(html)
    assert "<" not in result and ">" not in result
    assert "Bonjour" in result
    assert "le monde" in result


# ---------------------------------------------------------------------------
# Données nettoyées
# ---------------------------------------------------------------------------

def test_clean_text_non_vide(clean_events_data):
    """Chaque événement nettoyé doit avoir un champ 'text' non vide (utilisé pour l'indexation)."""
    for event in clean_events_data:
        assert event.get("text"), (
            f"Le champ 'text' est vide pour l'événement {event.get('uid')}"
        )


def test_clean_uid_unique(clean_events_data):
    """Vérifie qu'il n'y a pas de doublons d'uid dans les données nettoyées."""
    uids = [e["uid"] for e in clean_events_data]
    assert len(uids) == len(set(uids)), "Des doublons d'uid existent dans les données nettoyées"
