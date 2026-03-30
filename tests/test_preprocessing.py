"""
Tests unitaires pour scripts/clean_events.py

Couvre les fonctions de nettoyage et de validation des événements bruts :
  - strip_html         : suppression des balises HTML d'une chaîne de caractères
  - parse_label_fr     : extraction du label français depuis un champ JSON stringifié
  - _build_address     : construction d'une adresse sans doublons
  - clean_event        : nettoyage d'un événement brut (filtre les titres vides)
  - clean_events       : nettoyage par lot avec déduplication

Les tests sont organisés en trois groupes :
  1. Tests unitaires des fonctions de nettoyage (avec données mockées stables)
  2. Vérification des données nettoyées produites (text non vide, uid uniques)
  3. Tests de la fonction batch clean_events (déduplication, filtrage)
"""

import json
import pytest

from scripts.clean_events import clean_event, clean_events, strip_html, parse_label_fr, _build_address

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
    {
        "uid": "mock-003",
        "title_fr": "Atelier créatif enfants",
        "description_fr": "Atelier de peinture pour les enfants.",
        "longdescription_fr": "<p>Un atelier <em>créatif</em> pour petits artistes.</p>",
        "conditions_fr": "Sur inscription",
        "keywords_fr": ["atelier", "enfants"],
        "firstdate_begin": "2026-04-05T14:00:00+00:00",  # ~11 jours après FIXED_NOW
        "lastdate_end": "2026-04-05T17:00:00+00:00",
        "daterange_fr": "5 avril 2026",
        "location_name": "Maison des Arts",
        "location_address": "12 rue de la Paix",
        "location_postalcode": "75002",
        "location_city": "Paris",
        "location_district": "2e arrondissement",
        "location_department": "Paris",
        "location_region": "Île-de-France",
        "location_coordinates": {"lon": 2.331, "lat": 48.869},
        "canonicalurl": "https://example.com/atelier-enfants",
        "age_min": 6,
        "age_max": 12,
        "accessibility_label_fr": ["PMR"],
        "attendancemode": json.dumps({"label": {"fr": "En présentiel"}}),
        "status": json.dumps({"label": {"fr": "Programmé"}}),
    },
]


@pytest.fixture
def clean_events_data():
    return [e for e in (clean_event(r) for r in MOCK_RAW_EVENTS) if e is not None]


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


# ---------------------------------------------------------------------------
# parse_label_fr
# ---------------------------------------------------------------------------

def test_parse_label_fr_valide():
    """Extrait correctement le label français depuis un JSON stringifié valide."""
    val = json.dumps({"label": {"fr": "En présentiel"}})
    assert parse_label_fr(val) == "En présentiel"


def test_parse_label_fr_vide():
    """Retourne une chaîne vide si la valeur est None ou vide."""
    assert parse_label_fr(None) == ""
    assert parse_label_fr("") == ""


def test_parse_label_fr_json_invalide():
    """Retourne une chaîne vide si le JSON est malformé."""
    assert parse_label_fr("pas_du_json") == ""


# ---------------------------------------------------------------------------
# _build_address
# ---------------------------------------------------------------------------

def test_build_address_complet():
    """Construit une adresse complète avec code postal et ville."""
    result = _build_address("10 rue Victor Hugo", "69001", "Lyon")
    assert "69001" in result
    assert "Lyon" in result


def test_build_address_sans_doublon_ville():
    """Ne duplique pas la ville si elle est déjà présente dans l'adresse."""
    result = _build_address("10 rue de Paris", "75001", "Paris")
    assert result.count("Paris") == 1


# ---------------------------------------------------------------------------
# Comportements modifiés de clean_event
# ---------------------------------------------------------------------------

def test_clean_event_description_fallback():
    """Si description_fr est vide, le champ 'description' du résultat doit valoir le titre."""
    event = {**MOCK_RAW_EVENTS[0], "description_fr": ""}
    result = clean_event(event)
    assert result is not None
    assert result["description"] == result["title"]


# ---------------------------------------------------------------------------
# Fonction batch clean_events
# ---------------------------------------------------------------------------

def test_clean_events_deduplique():
    """clean_events doit ignorer les doublons d'uid et ne conserver que la première occurrence."""
    doublon = [MOCK_RAW_EVENTS[0], MOCK_RAW_EVENTS[0]]
    result = clean_events(doublon)
    assert len(result) == 1


def test_clean_events_filtre_sans_titre():
    """clean_events doit exclure les événements sans titre."""
    events = [{"uid": "x", "title_fr": "", "description_fr": "desc"}]
    assert clean_events(events) == []


