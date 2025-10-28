import os
import json
import statistics
from pathlib import Path

# === CONFIGURATION ===
BASE = "/home/ubuntu/onprem_data/tests"
REPORTS = os.path.join(BASE, "reports")
OUTPUT_DIR = os.path.join(BASE, "results")
SCORE_KEYS = ["robbert-2022", "mini-lm-l6"]


def compare_scores(ref, new):
    return {key: new[key] - ref[key] for key in SCORE_KEYS}


EXCLUDED_ANSWERS = {
    "Op basis van de informatie die ik tot mijn beschikking heb, weet ik het antwoord helaas niet.",
    "Het antwoord is niet duidelijk uit de context.",
    "Ik weet het antwoord niet.",
}


def process_question(q):
    antwoord = q.get("antwoord", "").strip()
    ref_scores = q.get("scores_ref", {})
    new_scores = q.get("scores_new", {})
    threshold = q.get("score_threshold", None)

    if antwoord in EXCLUDED_ANSWERS:
        # Invalidate scores
        ref_scores = {key: None for key in SCORE_KEYS}
        new_scores = {key: None for key in SCORE_KEYS}
        deltas = {key: None for key in SCORE_KEYS}
        threshold_passed = {key: False for key in SCORE_KEYS}
    else:
        deltas = compare_scores(ref_scores, new_scores)
        threshold_passed = {
            key: new_scores[key] >= threshold if threshold is not None else False
            for key in SCORE_KEYS
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


def summarize_scores(score_dicts):
    summary = {}

    for key in SCORE_KEYS:
        ref_values = [s["ref"][key] for s in score_dicts if s["ref"][key] is not None]
        new_values = [s["new"][key] for s in score_dicts if s["new"][key] is not None]
        delta_values = [
            s["delta"][key] for s in score_dicts if s["delta"][key] is not None
        ]
        threshold_passed = [
            s["threshold_passed"][key] for s in score_dicts if s["new"][key] is not None
        ]

        count = len(new_values)
        improved = sum(
            1
            for s in score_dicts
            if s["delta"][key] is not None and s["delta"][key] > 0
        )

        summary[key] = {
            "mean_ref": statistics.mean(ref_values) if ref_values else 0,
            "mean_new": statistics.mean(new_values) if new_values else 0,
            "mean_delta": statistics.mean(delta_values) if delta_values else 0,
            "std_delta": statistics.stdev(delta_values) if len(delta_values) > 1 else 0,
            "improved": improved,
            "threshold_passed": sum(threshold_passed),
            "count": count,
        }

    return summary


def write_text_file_report(filename, score_data, summary, model_info):
    lines = [f"📄 Evaluation Report: {filename}\n"]
    for i, q in enumerate(score_data, 1):
        lines.append(f"\n🔸 Vraag {i}: {q['vraag']}")
        lines.append(f"   Referentie acceptabel: {q['ref_accepted']}")
        if q["threshold"] is not None:
            lines.append(f"   Drempelwaarde: {q['threshold']}")
        for key in SCORE_KEYS:
            ref = q["ref"][key]
            new = q["new"][key]
            delta = q["delta"][key]
            passed = q["threshold_passed"].get(key, False)
            status = "✅ Passed" if passed else "❌ Failed" if q["threshold"] else ""
            if ref is None or new is None or delta is None:
                lines.append(f"   {key:15}: N/A (excluded from evaluation)")
            else:
                lines.append(
                    f"   {key:15}: {new:.4f} (ref: {ref:.4f}) → Δ {delta:+.4f} {status}"
                )

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
    lines.append(f"\n Model info: {model_info}")
    path = os.path.join(OUTPUT_DIR, f"file_report_{Path(filename).stem}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_summary_report(
    per_file_summaries, global_summary, total_acceptable, total_questions
):
    lines = []

    lines.append("📊 Global Evaluation Summary")
    for key, stats in global_summary.items():
        lines.append(
            f"\n🔹 {key}\n"
            f"   Avg Ref Score     : {stats['mean_ref']:.4f}\n"
            f"   Avg New Score     : {stats['mean_new']:.4f}\n"
            f"   Δ Mean            : {stats['mean_delta']:+.4f}\n"
            f"   Δ Std Dev         : {stats['std_delta']:.4f}\n"
            f"   # Improvements    : {stats['improved']}/{stats['count']}\n"
            f"   # Passed Threshold: {stats['threshold_passed']}/{stats['count']}"
        )

    lines.append("\n================= 📄 PER-FILE SUMMARIES =================")
    for filename, summary_data, count, model_info in per_file_summaries:
        lines.append(
            f"\n📄 Summary for: {filename} with model_info {model_info.get('IMAGE_NAME', 'unknown')}"
        )
        for key in SCORE_KEYS:
            stats = summary_data[key]
            lines.append(
                f"\n🔹 {key}\n"
                f"   Avg Ref Score     : {stats['mean_ref']:.4f}\n"
                f"   Avg New Score     : {stats['mean_new']:.4f}\n"
                f"   Δ Mean            : {stats['mean_delta']:+.4f}\n"
                f"   Δ Std Dev         : {stats['std_delta']:.4f}\n"
                f"   # Improvements    : {stats['improved']}/{count}\n"
                f"   # Passed Threshold: {stats['threshold_passed']}/{count}"
            )

    lines.append("\n================= TOTALS =================")
    lines.append(f"✅ Referentie acceptabel (Ja): {total_acceptable}")
    lines.append(f"❌ Referentie niet acceptabel: {total_questions - total_acceptable}")
    lines.append(f"📄 Total questions processed: {total_questions}")
    lines.append(f"📁 Directory: {REPORTS}")

    path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    total_questions = 0
    total_acceptable = 0
    all_scores = []
    report_json = []
    per_file_summaries = []
    files = sorted([f for f in os.listdir(REPORTS) if f.endswith(".json")])
    for filename in files:
        path = os.path.join(REPORTS, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data[1:] if isinstance(data, list) else [data]
        score_data = [process_question(q) for q in questions]
        summary = summarize_scores(score_data)
        model_info = data[0].get("model_info", {})
        report = {
            "file": filename,
            "model": model_info,
            "scores": summary,
        }
        report_json.append(report)
        write_text_file_report(
            filename,
            score_data,
            summary,
            model_info=model_info,
        )

        all_scores.extend(score_data)
        total_questions += len(score_data)
        total_acceptable += sum(
            1 for s in score_data if s["ref_accepted"].lower() == "ja"
        )
        per_file_summaries.append((filename, summary, len(score_data), model_info))

    with open("/home/ubuntu/onprem_data/tests/results/visual_report.json", "w") as f:
        json.dump(data, f, indent=3)

    global_summary = summarize_scores(all_scores)
    write_summary_report(
        per_file_summaries, global_summary, total_acceptable, total_questions
    )

    print(f"✅ Reports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
