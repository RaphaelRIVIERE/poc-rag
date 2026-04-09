import sys
import json
import argparse
from pathlib import Path


METRICS = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]


def load_report(path: Path) -> dict:
    if path.is_dir():
        files = sorted(path.glob("eval_*.json"))
        if not files:
            print(f"Aucun fichier eval_*.json trouvé dans {path}")
            sys.exit(1)
        path = files[-1]
        print(f"Fichier chargé : {path}\n")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def display(report: dict) -> None:
    print(f"Évalué le : {report['evaluated_at']}")
    rows = report["results"]

    print("\n" + "=" * 60)
    print("RÉSULTATS D'ÉVALUATION RAG (Ragas)")
    print("=" * 60)
    for metric in METRICS:
        values = [r[metric] for r in rows if r.get(metric) is not None and r[metric] == r[metric]]
        if values:
            avg = sum(values) / len(values)
            note = f"  ({len(rows) - len(values)} nan ignoré)" if len(values) < len(rows) else ""
            print(f"  {metric:<25} : {avg:.3f}{note}")
    print("=" * 60)

    print("\nDétail par question :")
    for i, row in enumerate(rows, 1):
        scores = " | ".join(
            f"{m[:10]}={row[m]:.2f}" for m in METRICS if row.get(m) is not None
        )
        print(f"  Q{i:02d} → {scores}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Affiche les résultats d'évaluation RAG")
    parser.add_argument("path", type=str, help="Fichier JSON ou dossier (dernier fichier utilisé)")
    args = parser.parse_args()

    report = load_report(Path(args.path))
    display(report)


if __name__ == "__main__":
    main()
