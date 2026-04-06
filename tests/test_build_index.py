"""
Tests unitaires pour scripts/build_index.py

Couvre les 4 fonctions principales du pipeline d'indexation :
  - load_events        : chargement d'un fichier JSON d'événements
  - events_to_documents: conversion en Documents LangChain (filtre les textes vides)
  - split_documents    : découpage en chunks pour l'indexation
  - build_faiss_index  : construction de l'index vectoriel (Mistral mocké, pas d'appel API)
  - save_index         : sauvegarde de l'index sur disque

Les embeddings Mistral sont mockés pour que les tests s'exécutent sans clé API.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.build_index import (
    events_to_documents,
    load_events,
    save_index,
    split_documents,
)

MOCK_EVENTS = [
    {
        "uid": "test-001",
        "title": "Concert de jazz",
        "text": "Titre : Concert de jazz | Description : Un concert incroyable | Dates : 15 janvier 2026 | Lieu : Salle Pleyel | Ville : Paris",
        "firstdate_begin": "2026-01-15",
        "lastdate_end": "2026-01-15",
        "location_name": "Salle Pleyel",
        "location_city": "Paris",
        "location_district": "",
        "location_postalcode": "75008",
        "location_dept": "Paris",
        "location_region": "Île-de-France",
        "conditions": "",
        "age_min": None,
        "age_max": None,
        "url": "https://example.com/concert-jazz",
    },
    {
        "uid": "test-002",
        "title": "Exposition art moderne",
        "text": "Titre : Exposition art moderne | Description : Une expo fascinante | Dates : 10 mai - 10 juin 2026 | Lieu : Centre Pompidou | Ville : Paris",
        "firstdate_begin": "2026-05-10",
        "lastdate_end": "2026-06-10",
        "location_name": "Centre Pompidou",
        "location_city": "Paris",
        "location_district": "",
        "location_postalcode": "75004",
        "location_dept": "Paris",
        "location_region": "Île-de-France",
        "conditions": "",
        "age_min": None,
        "age_max": None,
        "url": "https://example.com/expo-art",
    },
    {
        "uid": "test-003",
        "title": "Evenement sans texte",
        "text": "",  # doit être ignoré
        "firstdate_begin": "",
        "lastdate_end": "",
        "location_name": "",
        "location_city": "",
        "location_district": "",
        "location_postalcode": "",
        "location_dept": "",
        "location_region": "",
        "conditions": "",
        "age_min": None,
        "age_max": None,
        "url": "",
    },
]



# ---------------------------------------------------------------------------
# Chargement des événements
# ---------------------------------------------------------------------------

def test_load_events_charge_correctement():
    """Vérifie que load_events lit l'intégralité des événements d'un fichier JSON."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(MOCK_EVENTS, f, ensure_ascii=False)
        tmp_path = Path(f.name)

    events = load_events(tmp_path)
    assert len(events) == len(MOCK_EVENTS)
    tmp_path.unlink()



# ---------------------------------------------------------------------------
# Conversion des événements en documents LangChain
# ---------------------------------------------------------------------------

def test_events_to_documents_cree_bons_documents():
    """Sur 3 événements dont 1 sans texte, seuls 2 documents doivent être créés."""
    docs = events_to_documents(MOCK_EVENTS)
    assert len(docs) == 2


def test_events_to_documents_contenu_correct():
    """Le contenu du document doit inclure le titre de l'événement."""
    docs = events_to_documents(MOCK_EVENTS)
    assert "Concert de jazz" in docs[0].page_content


def test_events_to_documents_metadata_presentes():
    """Tous les champs de métadonnées attendus doivent être présents dans chaque document."""
    docs = events_to_documents(MOCK_EVENTS)
    meta = docs[0].metadata
    for key in ("uid", "title", "firstdate_begin", "lastdate_end", "location_name", "location_city", "location_district", "location_postalcode", "location_dept", "location_region", "url"):
        assert key in meta, f"Métadonnée manquante : {key}"


def test_events_to_documents_ignore_texte_vide():
    """Un événement avec un champ text vide ne doit pas produire de document."""
    events_vides = [{"uid": "x", "text": "", "title": "Vide"}]
    docs = events_to_documents(events_vides)
    assert len(docs) == 0



# ---------------------------------------------------------------------------
# Découpage en chunks
# ---------------------------------------------------------------------------

def test_split_documents_produit_des_chunks():
    """Le découpage doit produire au moins autant de chunks que de documents en entrée."""
    docs = events_to_documents(MOCK_EVENTS)
    chunks = split_documents(docs)
    assert len(chunks) >= len(docs)


def test_split_documents_chunks_non_vides():
    """Aucun chunk ne doit avoir un contenu vide après découpage."""
    docs = events_to_documents(MOCK_EVENTS)
    chunks = split_documents(docs)
    for chunk in chunks:
        assert chunk.page_content.strip() != ""


def test_split_documents_respecte_taille_max():
    """Chaque chunk doit respecter la taille maximale configurée (chunk_size=500 + marge)."""
    docs = events_to_documents(MOCK_EVENTS)
    chunks = split_documents(docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= 600  # chunk_size=500 + marge séparateur


# ---------------------------------------------------------------------------
# Construction de l'index vectoriel FAISS
# ---------------------------------------------------------------------------

def test_build_faiss_index_avec_embeddings_mock():
    """Vérifie que build_faiss_index appelle FAISS.from_documents avec les bons arguments.
    Les embeddings Mistral sont mockés pour éviter tout appel API réel."""
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS

    chunks = [
        Document(page_content="Concert de jazz à Paris", metadata={"uid": "test-001"}),
        Document(page_content="Exposition art moderne au Centre Pompidou", metadata={"uid": "test-002"}),
    ]

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    with patch("scripts.build_index.MistralAIEmbeddings", return_value=mock_embeddings), \
         patch("os.getenv", return_value="fake-api-key"), \
         patch.object(FAISS, "from_documents") as mock_faiss:

        mock_index = MagicMock(spec=FAISS)
        mock_faiss.return_value = mock_index

        from scripts.build_index import build_faiss_index
        index = build_faiss_index(chunks)

        mock_faiss.assert_called_once_with(chunks, mock_embeddings)
        assert index is mock_index


def test_build_faiss_index_leve_erreur_sans_cle():
    """build_faiss_index doit lever une ValueError si MISTRAL_API_KEY est absente."""
    from langchain_core.documents import Document

    chunks = [Document(page_content="test", metadata={})]

    with patch("os.getenv", return_value=None):
        from scripts.build_index import build_faiss_index
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            build_faiss_index(chunks)


# ---------------------------------------------------------------------------
# Sauvegarde de l'index sur disque
# ---------------------------------------------------------------------------

def test_save_index_cree_les_fichiers():
    """Vérifie que save_index appelle save_local sur l'index avec le bon chemin."""
    from langchain_community.vectorstores import FAISS

    mock_index = MagicMock(spec=FAISS)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "faiss_index"
        save_index(mock_index, output_path)
        mock_index.save_local.assert_called_once_with(str(output_path))
