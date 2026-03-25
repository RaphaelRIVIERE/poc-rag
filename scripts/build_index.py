"""
build_index.py — Vectorisation des événements et construction de l'index FAISS.

Étapes :
  1. Charger data/clean_events.json
  2. Découper les textes en chunks (RecursiveCharacterTextSplitter)
  3. Générer les embeddings via Mistral AI
  4. Construire et sauvegarder l'index FAISS dans index/faiss_index
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

DATA_FILE  = Path(__file__).parent.parent / "data" / "clean_events.json"
INDEX_DIR  = Path(__file__).parent.parent / "index" / "faiss_index"


def load_events(path: Path) -> list[dict]:
    events = json.loads(path.read_text(encoding="utf-8"))
    print(f"{len(events)} evenements charges depuis {path}")
    return events


def events_to_documents(events: list[dict]) -> list:
    """
    Convertit chaque événement en Document LangChain.
    Le champ 'text' devient le contenu, les autres champs deviennent les métadonnées.
    """
    from langchain_core.documents import Document

    docs = []
    for event in events:
        text = event.get("text", "").strip()
        if not text:
            continue
        metadata = {
            "uid":           event.get("uid", ""),
            "title":         event.get("title", ""),
            "date_begin":    event.get("date_begin", ""),
            "date_end":      event.get("date_end", ""),
            "daterange":     event.get("daterange", ""),
            "location_name": event.get("location_name", ""),
            "location_city": event.get("location_city", ""),
            "url":           event.get("url", ""),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    print(f"{len(docs)} documents crees")
    return docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[" | ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} chunks apres decoupage")
    return chunks


def build_faiss_index(chunks: list) -> FAISS:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY manquante — verifiez votre fichier .env")

    print("Generation des embeddings via Mistral AI...")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=SecretStr(api_key),
    )
    index = FAISS.from_documents(chunks, embeddings)
    print("Index FAISS construit.")
    return index


def save_index(index: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(path))
    print(f"Index sauvegarde dans {path}")


if __name__ == "__main__":
    t_start = time.time()

    events = load_events(DATA_FILE)
    docs   = events_to_documents(events)
    chunks = split_documents(docs)

    t_embed = time.time()
    index = build_faiss_index(chunks)
    print(f"  Embeddings generes en {time.time() - t_embed:.1f}s")

    save_index(index, INDEX_DIR)

    print(f"Termine en {time.time() - t_start:.1f}s.")
