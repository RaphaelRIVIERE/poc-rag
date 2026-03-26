"""
rag_chain.py — Chaîne RAG : recherche FAISS + génération Mistral.

Utilisation :
    from scripts.rag_chain import ask
    réponse = ask("Quels concerts sont prévus à Paris en avril ?")
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index" / "faiss_index"

PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Tu es un assistant spécialisé dans les événements culturels.
Réponds à la question en t'appuyant uniquement sur les événements fournis ci-dessous.
Si aucun événement ne correspond, dis-le clairement.

Événements pertinents :
{context}

Question : {question}

Réponse :"""
)


def load_index() -> FAISS:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY manquante — vérifiez votre fichier .env")

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=SecretStr(api_key),
    )
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_chain(index: FAISS):
    retriever = index.as_retriever(search_kwargs={"k": 5})

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=SecretStr(os.getenv("MISTRAL_API_KEY", "")),
        temperature=0.2,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return chain


# Index et chaîne chargés une seule fois au démarrage
_index = None
_chain = None


def _get_chain():
    global _index, _chain
    if _chain is None:
        _index = load_index()
        _chain = build_chain(_index)
    return _chain


def ask(question: str) -> str:
    """Pose une question au système RAG et retourne la réponse générée."""
    if not question.strip():
        raise ValueError("La question ne peut pas être vide.")
    return _get_chain().invoke(question)


if __name__ == "__main__":
    print("Chargement de l'index...")
    test_question = "Quels événements culturels sont prévus à Paris ?"
    print(f"Question : {test_question}\n")
    réponse = ask(test_question)
    print(f"Réponse :\n{réponse}")
