import csv
import json
import requests

url = "http://localhost:8080/evaluate"

if __name__ == "__main__":
    location_testdata = "testvragen.csv"
    results = []

    with open(location_testdata, newline="", encoding="latin-1") as f:
        total = sum(1 for _ in f) - 1

    with open(location_testdata, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")
        for idx, row in enumerate(reader):
            percent = (idx / total) * 100
            print(f"Processing test question {idx}/{total} ({percent:.1f}%)")
            if idx == 10:
                break
            expected = row["Antwoordrichting"]
            actual = row["Resultaat_19-9-2025"]
            payload = {"expected": expected, "actual": actual}

            response = requests.post(url, json=payload)
            print(response.json())
            response = response.json()["evaluation"].strip().lower()
            if response.startswith("correct"):
                result = "correct"
            elif response.startswith("incorrect"):
                result = "incorrect"
            else:
                result = "unknown"  # fallback, mocht er iets misgaan

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
    num_correct = sum(1 for r in results if r["resultaat"] == "correct")
    print(f"Totaal aantal vragen: {total}")
    print(f"Aantal correct: {num_correct}")
    print(f"Aantal incorrect: {total - num_correct}")
