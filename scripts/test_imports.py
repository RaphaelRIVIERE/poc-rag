"""Script de vérification des imports principaux du projet RAG."""

import sys


def check_import(module_path, label):
    try:
        exec(f"import {module_path.split('.')[0]}")
        # Try the full import path
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            mod = __import__(parts[0], fromlist=[parts[1]])
            getattr(mod, parts[1])
        print(f"  [OK] {label}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {label} — {e}")
        return False
    except AttributeError as e:
        print(f"  [FAIL] {label} — {e}")
        return False


def main():
    print("Vérification des imports du projet RAG Puls-Events\n")

    checks = [
        ("faiss", "faiss"),
        ("langchain_community.vectorstores.FAISS", "langchain FAISS vectorstore"),
        ("langchain_community.embeddings.HuggingFaceEmbeddings", "langchain HuggingFace Embeddings"),
        ("mistralai", "mistralai"),
        ("langchain_mistralai", "langchain-mistralai"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pandas", "Pandas"),
        ("requests", "Requests"),
        ("dotenv", "python-dotenv"),
        ("sentence_transformers", "sentence-transformers"),
    ]

    results = [check_import(mod, label) for mod, label in checks]

    print(f"\n{sum(results)}/{len(results)} imports réussis.")
    if not all(results):
        print("Lancez : pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("Environnement prêt.")


if __name__ == "__main__":
    main()
