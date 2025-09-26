import json
import os
import argparse
from statistics import mean


def analyze_report(path, fname):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    num_referentie = sum(1 for r in data if r.get("beste_antwoord") == "referentie")
    num_nieuw = sum(1 for r in data if r.get("beste_antwoord") == "nieuw")

    # Collect scores
    reranker_scores = [
        r.get("score_reranker") for r in data if r.get("score_reranker") is not None
    ]
    reranker_ref_scores = [
        r.get("score_reranker_ref")
        for r in data
        if r.get("score_reranker_ref") is not None
    ]

    cosine_scores = [
        r.get("score_consine_similarity")
        for r in data
        if r.get("score_consine_similarity") is not None
    ]
    cosine_ref_scores = [
        r.get("score_consine_similarity_ref")
        for r in data
        if r.get("score_consine_similarity_ref") is not None
    ]

    stats = {
        "file": fname,
        "date": fname.replace("evaluation_results_", "").replace(".json", ""),
        "total": total,
        "referentie": num_referentie,
        "nieuw": num_nieuw,
        "pct_referentie": (num_referentie / total) * 100 if total else 0,
        "pct_nieuw": (num_nieuw / total) * 100 if total else 0,
        "avg_reranker": mean(reranker_scores) if reranker_scores else None,
        "avg_reranker_ref": mean(reranker_ref_scores) if reranker_ref_scores else None,
        "avg_cosine": mean(cosine_scores) if cosine_scores else None,
        "avg_cosine_ref": mean(cosine_ref_scores) if cosine_ref_scores else None,
        "min_cosine": min(cosine_scores) if cosine_scores else None,
        "max_cosine": max(cosine_scores) if cosine_scores else None,
    }

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyseer evaluatie rapporten")
    parser.add_argument(
        "--dir",
        type=str,
        default="/home/ubuntu/onprem_data/tests/reports",
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
            print(f"   Totale vragen: {stats['total']}")
            print(
                f"   Referentie beter: {stats['referentie']} ({stats['pct_referentie']:.1f}%)"
            )
            print(f"   Nieuw beter:      {stats['nieuw']} ({stats['pct_nieuw']:.1f}%)")

            if stats["avg_reranker"] is not None:
                print(
                    f"   🔹 Reranker score gem.: {stats['avg_reranker']:.2f} (ref: {stats['avg_reranker_ref']:.2f})"
                )

            if stats["avg_cosine"] is not None:
                print(
                    f"   🔸 Cosine score gem.: {stats['avg_cosine']:.2f} (ref: {stats['avg_cosine_ref']:.2f})"
                )
                print(
                    f"      Cosine min: {stats['min_cosine']:.2f}, max: {stats['max_cosine']:.2f}"
                )

            print("-" * 50)

    # Totals across all reports
    if all_stats:
        total_qs = sum(s["total"] for s in all_stats)
        total_ref = sum(s["referentie"] for s in all_stats)
        total_nieuw = sum(s["nieuw"] for s in all_stats)

        print("\n📊 Samenvatting over alle rapporten:")
        print(f"   Totale vragen: {total_qs}")
        print(f"   Referentie beter: {total_ref} ({(total_ref / total_qs) * 100:.1f}%)")
        print(
            f"   Nieuw beter:      {total_nieuw} ({(total_nieuw / total_qs) * 100:.1f}%)"
        )

        # Optional: average scores across reports
        all_cosine = []
        all_cosine_ref = []
        all_reranker = []
        all_reranker_ref = []

        for s in all_stats:
            if s["avg_cosine"] is not None:
                all_cosine.append(s["avg_cosine"])
            if s["avg_cosine_ref"] is not None:
                all_cosine_ref.append(s["avg_cosine_ref"])
            if s["avg_reranker"] is not None:
                all_reranker.append(s["avg_reranker"])
            if s["avg_reranker_ref"] is not None:
                all_reranker_ref.append(s["avg_reranker_ref"])

        if all_reranker:
            print(
                f"   🔹 Gem. reranker score: {mean(all_reranker):.2f} (ref: {mean(all_reranker_ref):.2f})"
            )
        if all_cosine:
            print(
                f"   🔸 Gem. cosine score: {mean(all_cosine):.2f} (ref: {mean(all_cosine_ref):.2f})"
            )
    else:
        print("⚠️ Geen evaluatie rapporten gevonden.")
