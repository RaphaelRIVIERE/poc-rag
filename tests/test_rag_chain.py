"""
Tests unitaires pour scripts/rag_chain.py

Couvre :
  - ask() : lève ValueError si la question est vide
  - ask() : retourne une chaîne non vide (chaîne mockée)
  - load_index() : lève ValueError si MISTRAL_API_KEY est absente
  - build_chain() : retourne une chaîne LangChain à partir d'un index mocké
  - _get_chain() : charge l'index et construit la chaîne au premier appel
"""

import pytest
from unittest.mock import MagicMock, patch

# ask() — validation de l'entrée
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

# ask() — réponse générée (chaîne mockée)
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

# build_chain() — construction de la chaîne RAG
def test_build_chain_retourne_une_chaine():
    """build_chain() doit retourner une chaîne LangChain à partir d'un index mocké."""
    import scripts.rag_chain as rag_module

    mock_index = MagicMock()
    mock_index.as_retriever.return_value = MagicMock()

    with patch("scripts.rag_chain.ChatMistralAI"):
        chain = rag_module.build_chain(mock_index)

    assert chain is not None
    mock_index.as_retriever.assert_called_once_with(search_kwargs={"k": 5})


def test_build_chain_utilise_le_bon_retriever():
    """build_chain() doit configurer le retriever avec k=5."""
    import scripts.rag_chain as rag_module

    mock_index = MagicMock()
    mock_retriever = MagicMock()
    mock_index.as_retriever.return_value = mock_retriever

    with patch("scripts.rag_chain.ChatMistralAI"):
        rag_module.build_chain(mock_index)

    mock_index.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

# _get_chain() — chargement paresseux de l'index
def test_get_chain_charge_index_au_premier_appel():
    """_get_chain() doit appeler load_index et build_chain quand _chain est None."""
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

# load_index() — clé API manquante
def test_load_index_leve_erreur_sans_cle():
    """load_index() doit lever ValueError si MISTRAL_API_KEY est absente."""
    from scripts.rag_chain import load_index
    with patch("os.getenv", return_value=None):
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            load_index()
