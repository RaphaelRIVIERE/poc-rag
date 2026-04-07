"""Tests pour scripts/rag_chain.py — validation des entrées, chaîne RAG et lazy loading."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_question_invalide_leve_erreur(question):
    from scripts.rag_chain import ask
    with pytest.raises(ValueError, match="vide"):
        ask(question)


def test_ask_retourne_une_chaine():
    import scripts.rag_chain as rag_module

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Il y a un concert de jazz à Paris le 15 avril."

    with patch.object(rag_module, "_get_chain", return_value=mock_chain):
        result = rag_module.ask("Quels concerts à Paris ?")

    assert isinstance(result, str)
    assert len(result) > 0


def test_ask_appelle_invoke_avec_la_question():
    import scripts.rag_chain as rag_module

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Réponse de test."

    question = "Quels événements jazz cette semaine ?"
    with patch.object(rag_module, "_get_chain", return_value=mock_chain):
        rag_module.ask(question)

    mock_chain.invoke.assert_called_once_with(question)


def test_build_chain_retourne_une_chaine():
    # vérifie que le retriever est configuré avec k=5
    import scripts.rag_chain as rag_module

    mock_index = MagicMock()
    mock_index.as_retriever.return_value = MagicMock()

    with patch("scripts.rag_chain.ChatMistralAI"):
        chain = rag_module.build_chain(mock_index)

    assert chain is not None
    mock_index.as_retriever.assert_called_once_with(search_kwargs={"k": 5})


def test_get_chain_charge_index_au_premier_appel():
    """Vérifie le lazy loading : l'index est chargé une seule fois."""
    import scripts.rag_chain as rag_module

    mock_index = MagicMock()
    mock_chain = MagicMock()

    rag_module._chain = None
    rag_module._index = None

    with patch.object(rag_module, "load_index", return_value=mock_index) as mock_load, \
         patch.object(rag_module, "build_chain", return_value=mock_chain) as mock_build:
        result = rag_module._get_chain()

    assert result is mock_chain
    mock_load.assert_called_once()
    mock_build.assert_called_once_with(mock_index)


def test_load_index_leve_erreur_sans_cle():
    from scripts.rag_chain import load_index
    with patch("os.getenv", return_value=None):
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            load_index()
