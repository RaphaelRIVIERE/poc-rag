"""
build_index.py — Vectorisation des événements et construction de l'index FAISS.

Étapes :
  1. Charger data/clean_events.json
  2. Découper les textes en chunks (RecursiveCharacterTextSplitter)
  3. Générer les embeddings (HuggingFace ou Mistral)
  4. Construire et sauvegarder l'index FAISS dans index/faiss_index_<provider>

Utilisation :
    python scripts/build_index.py                        # HuggingFace par défaut
    python scripts/build_index.py --embeddings mistral   # Mistral AI
"""

import json
import os
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_FILE = Path(__file__).parent.parent / "data" / "clean_events.json"
INDEX_BASE = Path(__file__).parent.parent / "index"

HF_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_index_dir(provider: str) -> Path:
    return INDEX_BASE / f"faiss_index_{provider}"


def get_embeddings(provider: str = "huggingface"):
    if provider == "mistral":
        from pydantic import SecretStr
        from langchain_mistralai import MistralAIEmbeddings
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY manquante — vérifiez votre fichier .env")
        return MistralAIEmbeddings(model="mistral-embed", api_key=SecretStr(api_key))
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=HF_MODEL)


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


def split_documents(docs: list, chunk_size:int=500, chunk_overlap:int=50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[" | ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} chunks apres decoupage")
    return chunks


def build_faiss_index(chunks: list, provider: str = "mistral") -> FAISS:
    print(f"Generation des embeddings via {provider}...")
    index = FAISS.from_documents(chunks, get_embeddings(provider))
    print("Index FAISS construit.")
    return index


def save_index(index: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(path))
    print(f"Index sauvegarde dans {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings", choices=["huggingface", "mistral"], default="mistral",
        help="Modèle d'embeddings à utiliser (défaut: mistral)"
    )
    args = parser.parse_args()

    t_start = time.time()

    events = load_events(DATA_FILE)
    docs   = events_to_documents(events)
    chunks = split_documents(docs)

    t_embed = time.time()
    index = build_faiss_index(chunks, args.embeddings)
    print(f"  Embeddings generes en {time.time() - t_embed:.1f}s")

    save_index(index, get_index_dir(args.embeddings))

    print(f"Termine en {time.time() - t_start:.1f}s.")
