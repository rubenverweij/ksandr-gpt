import csv
import requests
import argparse
from datetime import datetime
import json
from scoring import get_answer_quality, compare_answers_with_cross_encoder
import time

url_question = "http://localhost:8080/ask"
url_question_output = "http://localhost:8080/status/{request_id}"
url_evaluate = "http://localhost:8080/evaluate"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate test questions against model."
    )
    parser.add_argument(
        "--file", type=str, default="testvragen.csv", help="Path to CSV test file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/onprem_data/tests/reports",
        help="Path to output dir",
    )
    parser.add_argument(
        "--question_col",
        type=str,
        default="Vraag",
        help="Column name for questions",
    )
    parser.add_argument(
        "--expected_col",
        type=str,
        default="Antwoordrichting",
        help="Column name for expected answers",
    )
    parser.add_argument(
        "--actual_col",
        type=str,
        default="Resultaat",
        help="Column name for actual answers",
    )
    args = parser.parse_args()

    location_testdata = args.file
    results = []

    # Datumstempel voor outputfile
    today = datetime.today().strftime("%Y-%m-%d")
    output_file = f"{args.output_dir}/evaluation_results_{today}.json"

    # Eerst tellen
    with open(location_testdata, newline="", encoding="latin-1") as f:
        total = sum(1 for _ in f) - 1

    def get_status_response(request_id):
        return requests.get(f"http://localhost:8080/status/{request_id}")

    # Daarna verwerken
    with open(location_testdata, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")
        for idx, row in enumerate(reader, start=1):
            percent = (idx / total) * 100
            print(f"Processing test question {idx}/{total} ({percent:.1f}%)")

            question = row[args.question_col]
            expected = row[args.expected_col]
            actual = row[args.actual_col]
            acceptable = row["Antwoord_acceptabel"]

            payload = {"prompt": question}
            response = requests.post(url_question, json=payload)
            request_id = json.loads(response.text)["request_id"]

            print(f"Processing test question with id {request_id}")

            while True:
                response_str = get_status_response(request_id)
                response = json.loads(response_str.text)
                status = response.get("status")
                print(f"Current status: {status}")
                time.sleep(10)
                if status == "completed":
                    print("Processing completed.")
                    break  # Exit the loop]

            answer = response["response"].get("answer")
            print(f"Answer is {answer} versus {actual}")

            cosine_score_reference = get_answer_quality(
                answer_1=expected, answer_2=actual
            )
            cosine_score_now = get_answer_quality(answer_1=expected, answer_2=answer)
            best_answer, score_diff, scores = compare_answers_with_cross_encoder(
                query=question, answer_1=actual, answer_2=answer
            )

            print(
                f"The cosine score is now: {cosine_score_now}, the score was: {cosine_score_reference}, the best answer according to the reranker {best_answer} with score diff {score_diff}"
            )

            results.append(
                {
                    "vraag": question,
                    "verwacht_antwoord": expected,
                    "antwoord": answer,
                    "antwoord_referentie": actual,
                    "referentie_acceptabel": acceptable,
                    "score_cosine_similarity": round(float(cosine_score_now), 2),
                    "score_reranker": round(float(scores[1]), 2),
                    "score_cosine_similarity_ref": round(
                        float(cosine_score_reference), 2
                    ),
                    "score_reranker_ref": round(float(scores[0]), 2),
                    "beste_antwoord": best_answer,
                }
            )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
