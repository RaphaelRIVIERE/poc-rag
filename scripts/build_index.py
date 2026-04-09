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

CHUNK_SIZE    = 700
CHUNK_OVERLAP = 50


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
            "uid":             event.get("uid", ""),
            "title":           event.get("title", ""),
            "firstdate_begin": event.get("firstdate_begin", ""),
            "lastdate_end":    event.get("lastdate_end", ""),
            "location_name":   event.get("location_name", ""),
            "location_city":       event.get("location_city", ""),
            "location_district":   event.get("location_district", ""),
            "location_postalcode": event.get("location_postalcode", ""),
            "location_dept":       event.get("location_dept", ""),
            "location_region": event.get("location_region", ""),
            "conditions":      event.get("conditions", ""),
            "age_min":         event.get("age_min"),
            "age_max":         event.get("age_max"),
            "url":             event.get("url", ""),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    print(f"{len(docs)} documents crees")
    return docs


def split_documents(docs: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[" | ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} chunks apres decoupage")
    return chunks


def get_embeddings() -> MistralAIEmbeddings:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY manquante — verifiez votre fichier .env")
    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=SecretStr(api_key),
    )


def build_faiss_index(chunks: list) -> FAISS:
    print("Generation des embeddings via Mistral AI...")
    index = FAISS.from_documents(chunks, get_embeddings())
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
