import csv
import requests
import argparse
from datetime import datetime
import json

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

            payload = {"prompt": question}
            response = requests.post(url_question, json=payload)
            request_id = json.loads(response.text)["request_id"]

            print(f"Processing test question with id {request_id} - {response.text}")

            while True:
                response_str = get_status_response(request_id)
                response = json.loads(response_str.text)
                status = response.get("status")
                print(f"Current status: {status}")
                if status == "completed":
                    print("Processing completed.")
                    break  # Exit the loop]
            print(response["response"]["answer"])

            if idx == 1:
                break

    #         payload = {"expected": expected, "actual": actual}

    #         response = requests.post(url, json=payload)
    #         data = response.json()
    #         response_text = data["evaluation"].strip().lower()

    #         match = re.search(
    #             r"\b([0-9](?:\.\d+)?|10(?:\.0+)?)(?:/10)?\b", response_text
    #         )
    #         if match:
    #             score = float(match.group(1))
    #         else:
    #             score = None

    #         if response_text.startswith("correct"):
    #             result = "correct"
    #         elif response_text.startswith("incorrect"):
    #             result = "incorrect"
    #         else:
    #             result = "unknown"

    #         results.append(
    #             {
    #                 "actual": actual,
    #                 "expected": expected,
    #                 "resultaat": result,
    #                 "toelichting": response_text,
    #                 "score": score,
    #             }
    #         )

    # # Schrijf naar JSON bestand
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

    # total = len(results)
    # num_correct = sum(1 for r in results if r["resultaat"] == "correct")

    # print(f"✅ Results saved to {output_file}")
    # print(f"Totaal aantal vragen: {total}")
    # print(f"Aantal correct: {num_correct}")
    # print(f"Aantal incorrect: {total - num_correct}")
