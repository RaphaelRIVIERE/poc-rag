"""
rag_chain.py — Chaîne RAG : recherche FAISS + génération Mistral.

Utilisation :
    from scripts.rag_chain import ask
    réponse = ask("Quels concerts sont prévus à Paris en avril ?")
"""

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from scripts.build_index import get_embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index" / "faiss_index"
RETRIEVER_K = 5

def _build_prompt_template() -> PromptTemplate:
    return PromptTemplate.from_template(
        """Tu es un assistant spécialisé dans les événements culturels en Île-de-France.
Ce système couvre uniquement les événements culturels en Île-de-France.
La date d'aujourd'hui est le {today}.
Si la question porte sur une région hors Île-de-France, indique clairement que la base est limitée à l'Île-de-France et qu'aucun événement hors de cette région n'est disponible.
Réponds à la question en t'appuyant uniquement sur les événements fournis ci-dessous.
Si aucun événement ne correspond, dis-le clairement sans proposer de sources externes.

Événements pertinents :
{context}

Question : {question}

Réponse :"""
)


def load_index() -> FAISS:
    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def build_chain(index: FAISS, k: int = RETRIEVER_K):
    retriever = index.as_retriever(search_kwargs={"k": k})

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=SecretStr(os.getenv("MISTRAL_API_KEY", "")),
        temperature=0.2,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "today": RunnableLambda(lambda _: date.today().strftime("%A %d %B %Y")),
        }
        | _build_prompt_template()
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
