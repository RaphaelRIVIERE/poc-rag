"""
evaluate_rag.py — Évaluation automatique du système RAG avec Ragas.

Métriques calculées :
  - answer_relevancy   : la réponse répond-elle bien à la question ?
  - faithfulness       : la réponse est-elle fidèle aux documents récupérés ?
  - context_precision  : le contexte récupéré contient-il peu de bruit ?
  - context_recall     : toutes les infos nécessaires ont-elles été récupérées ?

Utilisation :
    python tests/evaluate_rag.py
    python tests/evaluate_rag.py --output results/eval_results.json
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from pydantic import SecretStr
from langchain_mistralai import ChatMistralAI
from scripts.build_index import get_embeddings

ANNOTATIONS_PATH = Path(__file__).parent / "annotated_qa.json"


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}min {s:02d}s" if m else f"{s}s"


def load_annotations(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_ragas_llm_and_embeddings(provider: str = "huggingface"):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY manquante — vérifiez votre fichier .env")

    llm = LangchainLLMWrapper(
        ChatMistralAI(model="mistral-small-latest", api_key=SecretStr(api_key), temperature=0.0)
    )
    embeddings = LangchainEmbeddingsWrapper(get_embeddings(provider))
    return llm, embeddings


def collect_answers(annotations: list[dict]) -> list[dict]:
    """Génère les réponses RAG et récupère les contextes pour chaque question."""
    from scripts.rag_chain import load_index, build_chain

    index = load_index()
    chain = build_chain(index)
    retriever = index.as_retriever(search_kwargs={"k": 5})

    rows = []
    for i, item in enumerate(annotations, 1):
        print(f"  [{i}/{len(annotations)}] {item['question'][:60]}...")
        docs = retriever.invoke(item["question"])
        contexts = [doc.page_content for doc in docs]
        answer = chain.invoke(item["question"])
        rows.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["expected_answer"],
        })
    return rows


def print_summary(result) -> None:
    print("\n" + "=" * 60)
    print("RÉSULTATS D'ÉVALUATION RAG (Ragas)")
    print("=" * 60)
    df = result.to_pandas()
    for metric in ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]:
        if metric in df.columns:
            avg = df[metric].mean()
            print(f"  {metric:<25} : {avg:.3f}")
    print("=" * 60)
    print("\nDétail par question :")
    for i, row in enumerate(df.itertuples(), 1):
        scores = " | ".join(
            f"{m[:10]}={getattr(row, m):.2f}"
            for m in ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
            if m in df.columns
        )
        print(f"  Q{i:02d} → {scores}")
    print("=" * 60)


def save_results(result, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()
    report = {
        "evaluated_at": datetime.now().isoformat(),
        "results": df.to_dict(orient="records"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats sauvegardés dans : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Évaluation automatique du système RAG")
    parser.add_argument("--output", type=str, default=None,
                        help="Chemin de sauvegarde des résultats JSON")
    args = parser.parse_args()

    print("Chargement des annotations...")
    annotations = load_annotations(ANNOTATIONS_PATH)
    print(f"{len(annotations)} questions chargées.\n")

    t_start = time.time()

    print("Génération des réponses RAG...")
    t0 = time.time()
    rows = collect_answers(annotations)
    t_generation = time.time() - t0
    print(f"  Terminé en {fmt_duration(t_generation)}")

    print("\nÉvaluation avec Ragas...")
    t0 = time.time()
    llm, embeddings = build_ragas_llm_and_embeddings()
    dataset = Dataset.from_list(rows)
    run_config = RunConfig(timeout=180, max_workers=1, max_retries=5)
    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )

    t_evaluation = time.time() - t0
    print(f"  Terminé en {fmt_duration(t_evaluation)}")

    print_summary(result)
    print(f"\nTemps total : {fmt_duration(time.time() - t_start)}")

    if args.output:
        save_results(result, Path(args.output))


if __name__ == "__main__":
    main()
