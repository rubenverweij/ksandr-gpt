import json
import os
import argparse

import statistics
from collections import defaultdict

SCORE_KEYS = ["robbert-2022", "mini-lm-l6"]


# === HELPERS ===
def compare_scores(ref, new):
    return {key: new[key] - ref[key] for key in SCORE_KEYS}


def summarize_deltas(all_deltas):
    summary = {}
    for key in SCORE_KEYS:
        values = all_deltas[key]
        if values:
            summary[key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        else:
            summary[key] = {"mean": 0.0, "std": 0.0}
    return summary


def process_question(data):
    vraag = data.get("vraag")
    ref_accepted = data.get("referentie_acceptabel", "Onbekend")
    threshold = data.get("score_threshold", None)

    scores_ref = data["scores_ref"]
    scores_new = data["scores_new"]
    deltas = compare_scores(scores_ref, scores_new)

    results = {
        "vraag": vraag,
        "referentie_acceptabel": ref_accepted,
        "score_threshold": threshold,
        "scores_ref": scores_ref,
        "scores_new": scores_new,
        "delta": deltas,
    }

    if threshold:
        results["threshold_passed"] = {
            key: scores_new[key] >= threshold for key in SCORE_KEYS
        }

    return results


def print_question_stats(q, index):
    print(f"\n🔹 Vraag {index + 1}: {q['vraag']}")
    print(f"   Referentie acceptabel: {q['referentie_acceptabel']}")
    if "score_threshold" in q:
        print(f"   Drempelwaarde: {q['score_threshold']}")

    for key in SCORE_KEYS:
        ref = q["scores_ref"][key]
        new = q["scores_new"][key]
        delta = q["delta"][key]
        pass_status = ""
        if "threshold_passed" in q:
            passed = q["threshold_passed"][key]
            pass_status = "✅ Passed" if passed else "❌ Failed"

        print(f"   {key:15}: {new:.4f} (ref: {ref:.4f}) → Δ {delta:+.4f} {pass_status}")


# === MAIN SCRIPT ===
def main(directory):
    all_questions = []
    aggregate_ref_scores = defaultdict(list)
    aggregate_new_scores = defaultdict(list)
    aggregate_improvements = defaultdict(int)
    threshold_pass_counts = defaultdict(int)
    ref_acceptable_count = 0
    not_ref_acceptable_count = 0
    ref_acceptable_count = 0
    not_ref_acceptable_count = 0

    aggregate_deltas = defaultdict(list)

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Each file may be a single question or a list
        questions = data if isinstance(data, list) else [data]

        for q_data in questions:
            q_result = process_question(q_data)
            if q_result["referentie_acceptabel"].lower() == "ja":
                ref_acceptable_count += 1
            else:
                not_ref_acceptable_count += 1

            for key in SCORE_KEYS:
                delta = q_result["delta"][key]
                ref = q_result["scores_ref"][key]
                new = q_result["scores_new"][key]

                aggregate_deltas[key].append(delta)
                aggregate_ref_scores[key].append(ref)
                aggregate_new_scores[key].append(new)

                if new > ref:
                    aggregate_improvements[key] += 1

                if "threshold_passed" in q_result and q_result["threshold_passed"][key]:
                    threshold_pass_counts[key] += 1

            all_questions.append(q_result)

            for key in SCORE_KEYS:
                aggregate_deltas[key].append(q_result["delta"][key])

    # === PRINT OUTPUT ===
    print("\n=================== 📊 PER-QUESTION RESULTS ===================")
    for i, q in enumerate(all_questions):
        print_question_stats(q, i)

    print("\n=================== 📈 EXTENDED AGGREGATE SUMMARY ===================")
    for key in SCORE_KEYS:
        deltas = aggregate_deltas[key]
        ref_scores = aggregate_ref_scores[key]
        new_scores = aggregate_new_scores[key]
        improvements = aggregate_improvements[key]
        passes = threshold_pass_counts[key]

        mean_delta = statistics.mean(deltas)
        std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        mean_ref = statistics.mean(ref_scores)
        mean_new = statistics.mean(new_scores)

        print(f"\n🔹 {key}")
        print(f"   Avg Ref Score     : {mean_ref:.4f}")
        print(f"   Avg New Score     : {mean_new:.4f}")
        print(f"   Δ Mean            : {mean_delta:+.4f}")
        print(f"   Δ Std Dev         : {std_delta:.4f}")
        print(f"   # Improvements    : {improvements}/{len(deltas)}")
        print(f"   # Passed Threshold: {passes}/{len(deltas)}")

    print("\n=================== 📊 GLOBAL STATS ===================")
    print(f"✅ Referentie acceptabel (Ja):   {ref_acceptable_count}")
    print(f"❌ Referentie niet acceptabel:   {not_ref_acceptable_count}")
    print(f"📄 Total questions processed:    {len(all_questions)}")
    print(f"📁 Directory: {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyseer evaluatie rapporten")
    parser.add_argument(
        "--dir",
        type=str,
        default="/home/ubuntu/onprem_data/tests/reports",
        help="Map waar de evaluatie_resultaten_*.json bestanden staan",
    )
    args = parser.parse_args()
    main(args.dir)
