"""Tests pour scripts/build_index.py — chargement JSON, conversion en Documents, chunking et indexation FAISS."""

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
        "text": "Concert de jazz à Paris le 15 janvier 2026",
        "firstdate_begin": "2026-01-15",
        "lastdate_end": "2026-01-15",
        "location_name": "Salle Pleyel",
        "location_city": "Paris",
        "location_district": "",
        "location_postalcode": "75008",
        "location_dept": "Paris",
        "location_region": "Île-de-France",
        "url": "https://example.com/concert-jazz",
    },
    {
        "uid": "test-002",
        "title": "Exposition art moderne",
        "text": "Exposition art moderne au Centre Pompidou",
        "firstdate_begin": "2026-05-10",
        "lastdate_end": "2026-06-10",
        "location_name": "Centre Pompidou",
        "location_city": "Paris",
        "location_district": "",
        "location_postalcode": "75004",
        "location_dept": "Paris",
        "location_region": "Île-de-France",
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
        "url": "",
    },
]


def test_load_events_charge_correctement(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps(MOCK_EVENTS), encoding="utf-8")
    events = load_events(path)
    assert len(events) == len(MOCK_EVENTS)


def test_events_to_documents_ignore_texte_vide():
    docs = events_to_documents(MOCK_EVENTS)
    assert len(docs) == 2


def test_events_to_documents_contenu_et_metadata():
    docs = events_to_documents(MOCK_EVENTS)
    assert "Concert de jazz" in docs[0].page_content
    meta = docs[0].metadata
    for key in ("uid", "title", "firstdate_begin", "lastdate_end", "location_name",
                "location_city", "location_district", "location_postalcode",
                "location_dept", "location_region", "url"):
        assert key in meta, f"Métadonnée manquante : {key}"


def test_split_documents_produit_des_chunks_non_vides():
    docs = events_to_documents(MOCK_EVENTS)
    chunks = split_documents(docs)
    assert len(chunks) >= len(docs)
    for chunk in chunks:
        assert chunk.page_content.strip() != ""
        assert len(chunk.page_content) <= 600  # chunk_size=500 + marge séparateur


def test_build_faiss_index_avec_embeddings_mock():
    """Vérifie que FAISS.from_documents est appelé avec les bons arguments.
    Les embeddings sont mockés pour éviter tout appel à l'API Mistral."""
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
    from langchain_core.documents import Document

    chunks = [Document(page_content="test", metadata={})]

    with patch("os.getenv", return_value=None):
        from scripts.build_index import build_faiss_index
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            build_faiss_index(chunks)


def test_save_index_appelle_save_local():
    from langchain_community.vectorstores import FAISS

    mock_index = MagicMock(spec=FAISS)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "faiss_index"
        save_index(mock_index, output_path)
        mock_index.save_local.assert_called_once_with(str(output_path))
