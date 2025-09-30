import os
import json
import statistics
from pathlib import Path

# === CONFIGURATION ===
BASE = "/home/ubuntu/onprem_data/tests"
REPORTS = os.path.join(BASE, "reports")
OUTPUT_DIR = os.path.join(BASE, "results")
SCORE_KEYS = ["robbert-2022", "mini-lm-l6", "tfidf", "cross_encoder"]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def compare_scores(ref, new):
    return {key: new[key] - ref[key] for key in SCORE_KEYS}


def summarize_scores(score_dicts):
    return {
        key: {
            "mean_ref": statistics.mean([s["ref"][key] for s in score_dicts]),
            "mean_new": statistics.mean([s["new"][key] for s in score_dicts]),
            "mean_delta": statistics.mean([s["delta"][key] for s in score_dicts]),
            "std_delta": (
                statistics.stdev([s["delta"][key] for s in score_dicts])
                if len(score_dicts) > 1
                else 0
            ),
            "improved": sum(1 for s in score_dicts if s["delta"][key] > 0),
            "threshold_passed": sum(
                1 for s in score_dicts if s["threshold_passed"].get(key, False)
            ),
            "count": len(score_dicts),
        }
        for key in SCORE_KEYS
    }


def process_question(q):
    ref_scores = q["scores_ref"]
    new_scores = q["scores_new"]
    deltas = compare_scores(ref_scores, new_scores)
    threshold = q.get("score_threshold", None)
    threshold_passed = {
        key: new_scores[key] >= threshold if threshold else False for key in SCORE_KEYS
    }

    return {
        "vraag": q.get("vraag", ""),
        "ref_accepted": q.get("referentie_acceptabel", "Onbekend"),
        "ref": ref_scores,
        "new": new_scores,
        "delta": deltas,
        "threshold": threshold,
        "threshold_passed": threshold_passed,
    }


def generate_file_report(filename, questions_data):
    lines = [f"📄 Evaluation Report: {filename}\n"]
    score_dicts = []

    for i, q in enumerate(questions_data, 1):
        result = process_question(q)
        score_dicts.append(result)

        lines.append(f"\n🔸 Vraag {i}: {result['vraag']}")
        lines.append(f"   Referentie acceptabel: {result['ref_accepted']}")
        if result["threshold"] is not None:
            lines.append(f"   Drempelwaarde: {result['threshold']}")

        for key in SCORE_KEYS:
            ref = result["ref"][key]
            new = result["new"][key]
            delta = result["delta"][key]
            passed = result["threshold_passed"].get(key, False)
            status = (
                "✅ Passed"
                if passed
                else "❌ Failed"
                if result["threshold"] is not None
                else ""
            )
            lines.append(
                f"   {key:15}: {new:.4f} (ref: {ref:.4f}) → Δ {delta:+.4f} {status}"
            )

    # Per-file summary
    summary = summarize_scores(score_dicts)
    lines.append("\n📊 Summary per scoring model:")
    for key, stats in summary.items():
        lines.append(
            f"\n🔹 {key}\n"
            f"   Avg Ref Score     : {stats['mean_ref']:.4f}\n"
            f"   Avg New Score     : {stats['mean_new']:.4f}\n"
            f"   Δ Mean            : {stats['mean_delta']:+.4f}\n"
            f"   Δ Std Dev         : {stats['std_delta']:.4f}\n"
            f"   # Improvements    : {stats['improved']}/{stats['count']}\n"
            f"   # Passed Threshold: {stats['threshold_passed']}/{stats['count']}"
        )

    # Write to individual file
    out_path = os.path.join(OUTPUT_DIR, f"file_report_{Path(filename).stem}.txt")
    with open(out_path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(lines))

    return score_dicts, sum(1 for s in score_dicts if s["ref_accepted"].lower() == "ja")


def main():
    all_scores = []
    total_questions = 0
    total_acceptable = 0

    for filename in os.listdir(REPORTS):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(REPORTS, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data if isinstance(data, list) else [data]
        score_data, file_acceptable_count = generate_file_report(filename, questions)
        all_scores.extend(score_data)
        total_acceptable += file_acceptable_count
        total_questions += len(questions)

    # Generate total summary report
    lines = ["📊 Global Evaluation Summary\n"]
    summary = summarize_scores(all_scores)

    for key, stats in summary.items():
        lines.append(
            f"\n🔹 {key}\n"
            f"   Avg Ref Score     : {stats['mean_ref']:.4f}\n"
            f"   Avg New Score     : {stats['mean_new']:.4f}\n"
            f"   Δ Mean            : {stats['mean_delta']:+.4f}\n"
            f"   Δ Std Dev         : {stats['std_delta']:.4f}\n"
            f"   # Improvements    : {stats['improved']}/{stats['count']}\n"
            f"   # Passed Threshold: {stats['threshold_passed']}/{stats['count']}"
        )

    lines.append("\n================= TOTALS =================")
    lines.append(f"✅ Referentie acceptabel (Ja): {total_acceptable}")
    lines.append(f"❌ Referentie niet acceptabel: {total_questions - total_acceptable}")
    lines.append(f"📄 Total questions processed: {total_questions}")
    lines.append(f"📁 Directory: {REPORTS}")

    with open(
        os.path.join(OUTPUT_DIR, "summary_report.txt"), "w", encoding="utf-8"
    ) as out:
        out.write("\n".join(lines))

    print(f"\n✅ Reports written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
