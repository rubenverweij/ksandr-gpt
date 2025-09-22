import csv
import json
import requests

url = "http://localhost:8000/evaluate"

if __name__ == "__main__":
    results = []
    with open("langchain/tests/testvragen.csv", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for idx, row in enumerate(reader):
            print(f"Processing test question {idx}/{len(reader)}")
            expected = row["Antwoordrichting"]
            actual = row["Resultaat_19-9-2025"]
            payload = {"expected": expected, "actual": actual}

            response = requests.post(url, json=payload)
            response = response.json()["model_response"].strip().lower()
            if response.startswith("correct"):
                result = "correct"
            elif response.startswith("incorrect"):
                result = "incorrect"
            else:
                result = "unknown"
            results.append(
                {
                    "actual": actual,
                    "expected": expected,
                    "resultaat": result,
                    "toelichting": response,
                }
            )

    # Schrijf naar JSON bestand
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = len(results)
    num_correct = sum(1 for r in results if r["result"] == "correct")
    print(f"Totaal aantal vragen: {total}")
    print(f"Aantal correct: {num_correct}")
    print(f"Aantal incorrect: {total - num_correct}")
