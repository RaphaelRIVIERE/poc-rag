"""Tests pour scripts/clean_events.py — nettoyage et normalisation des données OpenAgenda."""

import json
import pytest

from scripts.clean_events import clean_event, clean_events, strip_html, parse_label_fr, _build_address, normalize_dept, normalize_district

MOCK_RAW_EVENTS = [
    {
        "uid": "mock-001",
        "title_fr": "Concert de jazz",
        "description_fr": "Un concert de jazz.",
        "longdescription_fr": "<p>Une soirée <strong>jazz</strong> inoubliable.</p>",
        "conditions_fr": "Entrée libre",
        "keywords_fr": ["jazz", "musique"],
        "firstdate_begin": "2026-01-15T18:00:00+00:00",
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
        "firstdate_begin": "2026-05-10T10:00:00+00:00",
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
        "firstdate_begin": "2026-04-05T14:00:00+00:00",
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


def test_clean_event_sans_titre_retourne_none():
    event_sans_titre = {
        "uid": "test-001",
        "title_fr": "",
        "description_fr": "Une description.",
        "firstdate_begin": "2026-06-01T10:00:00+00:00",
        "location_region": "Île-de-France",
    }
    assert clean_event(event_sans_titre) is None


def test_strip_html_supprime_balises():
    html = "<p>Bonjour <strong>le monde</strong> !</p>"
    result = strip_html(html)
    assert "<" not in result and ">" not in result
    assert "Bonjour" in result and "le monde" in result


def test_clean_text_non_vide(clean_events_data):
    for event in clean_events_data:
        assert event.get("text"), f"Le champ 'text' est vide pour {event.get('uid')}"


def test_clean_uid_unique(clean_events_data):
    uids = [e["uid"] for e in clean_events_data]
    assert len(uids) == len(set(uids))


def test_parse_label_fr():
    val = json.dumps({"label": {"fr": "En présentiel"}})
    assert parse_label_fr(val) == "En présentiel"
    assert parse_label_fr(None) == ""
    assert parse_label_fr("") == ""
    assert parse_label_fr("pas_du_json") == ""


def test_build_address():
    result = _build_address("10 rue Victor Hugo", "69001", "Lyon")
    assert "69001" in result and "Lyon" in result

    # ne duplique pas la ville si déjà présente dans l'adresse
    result = _build_address("10 rue de Paris", "75001", "Paris")
    assert result.count("Paris") == 1


def test_clean_event_normalise_city():
    cases = [
        ("PARIS", "Paris"),
        ("centre-ville", "Centre-Ville"),
        ("  aix-en-provence  ", "Aix-En-Provence"),
    ]
    for input_city, expected in cases:
        result = clean_event({**MOCK_RAW_EVENTS[0], "location_city": input_city})
        assert result is not None
        assert result["location_city"] == expected


def test_clean_event_description_fallback():
    event = {**MOCK_RAW_EVENTS[0], "description_fr": ""}
    result = clean_event(event)
    assert result is not None
    assert result["description"] == result["title"]


def test_clean_events_deduplique_et_filtre():
    doublon = [MOCK_RAW_EVENTS[0], MOCK_RAW_EVENTS[0]]
    assert len(clean_events(doublon)) == 1

    sans_titre = [{"uid": "x", "title_fr": "", "description_fr": "desc"}]
    assert clean_events(sans_titre) == []


def test_normalize_dept():
    assert normalize_dept("Seine-St-Denis") == "Seine-Saint-Denis"
    assert normalize_dept("Seine-St.-Denis") == "Seine-Saint-Denis"
    assert normalize_dept("paris") == "Paris"
    assert normalize_dept("HAUTS-DE-SEINE") == "Hauts-De-Seine"
    assert normalize_dept("Bretagne") == "Bretagne"


def test_clean_event_normalise_dept():
    raw = {**MOCK_RAW_EVENTS[0], "location_department": "Seine-St-Denis"}
    result = clean_event(raw)
    assert result is not None
    assert result["location_dept"] == "Seine-Saint-Denis"


def test_normalize_district():
    # Suppression du préfixe "Quartier"
    assert normalize_district("Quartier Saint-Lambert", "") == "Saint-Lambert"
    assert normalize_district("Quartier de la Sorbonne", "") == "Sorbonne"
    assert normalize_district("Quartier du Marais", "") == "Marais"
    assert normalize_district("Quartier des Batignolles", "") == "Batignolles"

    # Paris → arrondissement si code postal connu
    assert normalize_district("Paris", "75012") == "Paris 12e Arrondissement"
    assert normalize_district("Paris", "75001") == "Paris 1er Arrondissement"
    assert normalize_district("Paris", "") == "Paris"

    # Normalisation Centre-Ville
    assert normalize_district("Centre Ville", "") == "Centre-Ville"
    assert normalize_district("Centre-ville", "") == "Centre-Ville"


