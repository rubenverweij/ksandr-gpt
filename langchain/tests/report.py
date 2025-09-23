import json
import os
import argparse
from statistics import mean


def analyze_report(path, fname):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    num_correct = sum(1 for r in data if r["resultaat"] == "correct")
    num_incorrect = sum(1 for r in data if r["resultaat"] == "incorrect")
    num_unknown = sum(1 for r in data if r["resultaat"] == "unknown")

    scores = [r["score"] for r in data if r.get("score") is not None]

    stats = {
        "file": fname,
        "date": fname.replace("evaluation_results_", "").replace(".json", ""),
        "total": total,
        "correct": num_correct,
        "incorrect": num_incorrect,
        "unknown": num_unknown,
        "avg_score": mean(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyseer evaluatie rapporten")
    parser.add_argument(
        "--dir",
        type=str,
        default="/home/ubuntu/onprem_data/tests/",
        help="Map waar de evaluatie_resultaten_*.json bestanden staan",
    )
    args = parser.parse_args()

    all_stats = []
    for fname in sorted(os.listdir(args.dir)):
        if fname.startswith("evaluation_results_") and fname.endswith(".json"):
            path = os.path.join(args.dir, fname)
            stats = analyze_report(path, fname)
            all_stats.append(stats)

            print(f"📅 Rapport {stats['date']}:")
            print(f"   Totaal: {stats['total']}")
            print(
                f"   Correct: {stats['correct']}, Incorrect: {stats['incorrect']}, Unknown: {stats['unknown']}"
            )
            if stats["avg_score"] is not None:
                print(
                    f"   Score gem.: {stats['avg_score']:.2f}/10 "
                    f"(min: {stats['min_score']:.1f}, max: {stats['max_score']:.1f})"
                )
            else:
                print("   Geen scores gevonden")
            print("-" * 50)

    if not all_stats:
        print("⚠️ Geen evaluatie rapporten gevonden.")
    else:
        # Optioneel: totale samenvatting over alle rapporten
        all_total = sum(s["total"] for s in all_stats)
        all_correct = sum(s["correct"] for s in all_stats)
        all_incorrect = sum(s["incorrect"] for s in all_stats)
        all_unknown = sum(s["unknown"] for s in all_stats)
        all_scores = [r for s in all_stats for r in ([]) if s["avg_score"]]

        print("📊 Samenvatting over alle rapporten:")
        print(f"   Totale vragen: {all_total}")
        print(
            f"   Correct: {all_correct}, Incorrect: {all_incorrect}, Unknown: {all_unknown}"
        )
