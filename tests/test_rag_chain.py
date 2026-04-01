"""
Tests unitaires pour scripts/rag_chain.py

Couvre :
  - ask() : lève ValueError si la question est vide
  - ask() : retourne une chaîne non vide (chaîne mockée)
  - load_index() : lève ValueError si MISTRAL_API_KEY est absente
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# ask() — validation de l'entrée
# ---------------------------------------------------------------------------

def test_ask_question_vide_leve_erreur():
    """ask() doit lever ValueError si la question est vide."""
    from scripts.rag_chain import ask
    with pytest.raises(ValueError, match="vide"):
        ask("")


def test_ask_question_espaces_leve_erreur():
    """ask() doit lever ValueError si la question ne contient que des espaces."""
    from scripts.rag_chain import ask
    with pytest.raises(ValueError, match="vide"):
        ask("   ")


# ---------------------------------------------------------------------------
# ask() — réponse générée (chaîne mockée)
# ---------------------------------------------------------------------------

def test_ask_retourne_une_chaine():
    """ask() doit retourner une chaîne non vide quand la chaîne RAG est mockée."""
    import scripts.rag_chain as rag_module

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Il y a un concert de jazz à Paris le 15 avril."

    with patch.object(rag_module, "_get_chain", return_value=mock_chain):
        result = rag_module.ask("Quels concerts à Paris ?")

    assert isinstance(result, str)
    assert len(result) > 0


def test_ask_appelle_invoke_avec_la_question():
    """ask() doit transmettre la question telle quelle à la chaîne."""
    import scripts.rag_chain as rag_module

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Réponse de test."

    question = "Quels événements jazz cette semaine ?"
    with patch.object(rag_module, "_get_chain", return_value=mock_chain):
        rag_module.ask(question)

    mock_chain.invoke.assert_called_once_with(question)


# ---------------------------------------------------------------------------
# load_index() — clé API manquante
# ---------------------------------------------------------------------------

def test_load_index_leve_erreur_sans_cle():
    """load_index() doit lever ValueError si MISTRAL_API_KEY est absente."""
    from scripts.rag_chain import load_index
    with patch("scripts.build_index.os.getenv", return_value=None):
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            load_index(provider="mistral")
