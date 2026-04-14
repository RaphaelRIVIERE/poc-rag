import sys
import json
import argparse
from pathlib import Path


METRICS = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
THRESHOLDS = {"answer_relevancy": 0.70, "faithfulness": 0.65, "context_precision": 0.45, "context_recall": 0.70}


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


def print_metrics_summary(averages: dict[str, float], notes: dict[str, str] | None = None) -> None:
    print("\n" + "=" * 60)
    print("RÉSULTATS D'ÉVALUATION RAG (Ragas)")
    print("=" * 60)
    for metric, avg in averages.items():
        threshold = THRESHOLDS.get(metric)
        status = "✓" if threshold is None or avg >= threshold else "✗"
        threshold_str = f"  (seuil : {threshold:.2f})" if threshold is not None else ""
        note = (notes or {}).get(metric, "")
        print(f"  {status} {metric:<25} : {avg:.3f}{threshold_str}{note}")
    print("=" * 60)


def display(report: dict) -> None:
    print(f"Évalué le : {report['evaluated_at']}")
    rows = report["results"]

    averages = {}
    notes = {}
    for metric in METRICS:
        values = [r[metric] for r in rows if r.get(metric) is not None and r[metric] == r[metric]]
        if values:
            averages[metric] = sum(values) / len(values)
            if len(values) < len(rows):
                notes[metric] = f"  ({len(rows) - len(values)} nan ignoré)"

    print_metrics_summary(averages, notes)

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
