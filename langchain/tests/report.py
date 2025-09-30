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


def write_text_file_report(filename, score_data, summary):
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

    path = os.path.join(OUTPUT_DIR, f"file_report_{Path(filename).stem}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_html_report(
    per_file_summaries, global_summary, total_acceptable, total_questions
):
    html = [
        '<html><head><meta charset="UTF-8"><style>',
        "body { font-family: Arial; }",
        "summary { font-weight: bold; }",
        "table { border-collapse: collapse; margin-bottom: 20px; }",
        "td, th { border: 1px solid #aaa; padding: 4px 8px; text-align: left; }",
        ".pass { color: green; } .fail { color: red; }",
        "</style></head><body>",
    ]

    html.append("<h1>📊 Evaluation Summary</h1>")

    html.append(f"<p>Total Questions: <b>{total_questions}</b><br>")
    html.append(f"Referentie Acceptabel: <b>{total_acceptable}</b><br>")
    html.append(
        f"Referentie Niet Acceptabel: <b>{total_questions - total_acceptable}</b></p>"
    )

    html.append("<h2>📁 Per Report Summary</h2>")

    for filename, summary, count in per_file_summaries:
        html.append(f"<details><summary>{filename}</summary>")
        html.append(
            "<table><tr><th>Model</th><th>Avg Ref</th><th>Avg New</th><th>Δ Mean</th>"
            "<th>Δ Std</th><th># Improvements</th><th># Passed Threshold</th></tr>"
        )
        for key in SCORE_KEYS:
            s = summary[key]
            html.append(
                f"<tr><td>{key}</td><td>{s['mean_ref']:.2f}</td><td>{s['mean_new']:.2f}</td>"
                f"<td>{s['mean_delta']:+.2f}</td><td>{s['std_delta']:.2f}</td>"
                f"<td>{s['improved']}/{count}</td><td>{s['threshold_passed']}/{count}</td></tr>"
            )
        html.append("</table></details>")

    html.append("<h2>🌍 Global Summary</h2>")
    html.append(
        "<table><tr><th>Model</th><th>Avg Ref</th><th>Avg New</th><th>Δ Mean</th>"
        "<th>Δ Std</th><th># Improvements</th><th># Passed Threshold</th></tr>"
    )
    for key in SCORE_KEYS:
        s = global_summary[key]
        html.append(
            f"<tr><td>{key}</td><td>{s['mean_ref']:.2f}</td><td>{s['mean_new']:.2f}</td>"
            f"<td>{s['mean_delta']:+.2f}</td><td>{s['std_delta']:.2f}</td>"
            f"<td>{s['improved']}/{s['count']}</td><td>{s['threshold_passed']}/{s['count']}</td></tr>"
        )
    html.append("</table></body></html>")

    path = os.path.join(OUTPUT_DIR, "summary_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def main():
    total_questions = 0
    total_acceptable = 0
    all_scores = []
    per_file_summaries = []

    for filename in os.listdir(REPORTS):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(REPORTS, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data if isinstance(data, list) else [data]
        score_data = [process_question(q) for q in questions]
        summary = summarize_scores(score_data)

        write_text_file_report(filename, score_data, summary)

        all_scores.extend(score_data)
        total_questions += len(score_data)
        total_acceptable += sum(
            1 for s in score_data if s["ref_accepted"].lower() == "ja"
        )
        per_file_summaries.append((filename, summary, len(score_data)))

    # Global Summary
    global_summary = summarize_scores(all_scores)

    # Generate HTML report
    write_html_report(
        per_file_summaries, global_summary, total_acceptable, total_questions
    )

    print(f"✅ Reports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
