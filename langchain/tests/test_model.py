import csv
import requests
import argparse
from datetime import datetime
import json
from scoring import (
    get_answer_quality,
    compare_answers_with_cross_encoder,
    SENTENCE_TRANSFORMERS,
    tfidf_cosine_sim,
)
import time

url_question = "http://localhost:8080/ask"
url_question_output = "http://localhost:8080/status/{request_id}"
url_evaluate = "http://localhost:8080/evaluate"


def drop_keys(obj, keys_to_drop):
    """
    Recursively drop all keys in `keys_to_drop` from nested dicts/lists.
    """
    if isinstance(obj, dict):
        return {
            k: drop_keys(v, keys_to_drop)
            for k, v in obj.items()
            if k not in keys_to_drop
        }
    elif isinstance(obj, list):
        return [drop_keys(i, keys_to_drop) for i in obj]
    else:
        return obj


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
    model_metadata = requests.get("http://localhost:8080/metadata")
    results = [{"model_info": json.loads(model_metadata.text)}]

    # Datumstempel voor outputfile
    today = datetime.today().strftime("%Y-%m-%d_%H:%M")
    output_file = f"{args.output_dir}/evaluation_results_{today}.json"

    # Eerst tellen
    with open(location_testdata, newline="", encoding="latin-1") as f:
        total = sum(1 for _ in f) - 1

    def get_status_response(request_id):
        return requests.get(f"http://localhost:8080/status/{request_id}")

    # Daarna verwerken
    with open(location_testdata, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")
        for idx, row in enumerate(reader, start=1):
            if idx == 36:
                continue
            percent = (idx / total) * 100
            print(f"Processing test question {idx}/{total} ({percent:.1f}%)")

            question = row[args.question_col]
            expected = row[args.expected_col]
            actual = row[args.actual_col]
            acceptable = row["Antwoord_acceptabel"]

            payload = {
                "prompt": question,
                "permission": {
                    "aads": {
                        "cat-1": [
                            318,
                            655,
                            1555,
                            1556,
                            1557,
                            1558,
                            2059,
                            2061,
                            2963,
                            8825,
                            8827,
                            9026,
                            9027,
                            9028,
                            10535,
                            10536,
                            10540,
                            10542,
                            10545,
                            10546,
                            10547,
                            10548,
                            10551,
                            10552,
                            10553,
                            10554,
                            10555,
                            10556,
                            10557,
                        ]
                    },
                    "documents": [
                        10733,
                        10734,
                        10735,
                        10736,
                        10737,
                        10738,
                        10739,
                        10740,
                        10759,
                        10760,
                        10761,
                        10863,
                        10864,
                        10865,
                        10866,
                        10867,
                        10868,
                        10869,
                        10999,
                        11041,
                        11265,
                        11274,
                        11275,
                        11383,
                        11384,
                        11385,
                        11386,
                        11551,
                        12403,
                        12404,
                        12405,
                        12406,
                        12462,
                        12463,
                        12556,
                        13682,
                        13683,
                        13684,
                        13685,
                        13686,
                        13687,
                        13692,
                        13697,
                        13698,
                        14025,
                        14198,
                        14199,
                        14215,
                        14221,
                        14247,
                        14371,
                        14373,
                        14374,
                        14375,
                        14376,
                        14377,
                        14378,
                        14379,
                        14380,
                        14383,
                        14385,
                        14399,
                    ],
                    "groups": [
                        260,
                        277,
                        278,
                        280,
                        281,
                        826,
                        827,
                        828,
                        832,
                        1217,
                        1218,
                        1961,
                        1968,
                        2175,
                        2408,
                        9001,
                        9358,
                        9359,
                        10193,
                        10541,
                        10678,
                        10684,
                        10685,
                        10686,
                        10687,
                        10688,
                        10689,
                        10690,
                        10691,
                        10692,
                        10693,
                        10694,
                        10695,
                        10696,
                        10697,
                        10698,
                        10699,
                        10700,
                        10702,
                        10703,
                        10705,
                        10706,
                        10707,
                        10708,
                        10709,
                        10710,
                        10712,
                        10713,
                    ],
                    "ese": True,
                    "esg": True,
                    "rmd": [
                        14284172,
                        18613440,
                        18860943,
                        22990584,
                        27200298,
                        29964311,
                        35966728,
                        37781047,
                        44972089,
                        48539960,
                        49958430,
                        51044944,
                        62155242,
                        62276376,
                        63665801,
                        65056104,
                        65542778,
                        70474078,
                        74132964,
                        83602093,
                        89942749,
                        91846034,
                        92430633,
                        98606898,
                    ],
                    "dga": [
                        14284172,
                        18613440,
                        18860943,
                        22990584,
                        27200298,
                        29964311,
                        35966728,
                        37781047,
                        44972089,
                        48539960,
                        49958430,
                        51044944,
                        62155242,
                        62276376,
                        63665801,
                        65056104,
                        65542778,
                        70474078,
                        74132964,
                        83602093,
                        89942749,
                        91846034,
                        92430633,
                        98606898,
                    ],
                },
            }
            response = requests.post(url_question, json=payload)
            request_id = json.loads(response.text)["request_id"]

            print(f"Processing test question with id {request_id}")

            while True:
                response_str = get_status_response(request_id)
                response = json.loads(response_str.text)
                status = response.get("status")
                time.sleep(1)
                if status == "completed":
                    print("Processing completed.")
                    break  # Exit the loop]

            answer = response["response"].get("answer")
            results_ref = {}
            results_new = {}
            for name, transformer in SENTENCE_TRANSFORMERS.items():
                results_ref.update(
                    {
                        name: float(
                            get_answer_quality(
                                transformer, answer_1=expected, answer_2=actual
                            )
                        )
                    }
                )
                results_new.update(
                    {
                        name: get_answer_quality(
                            transformer, answer_1=expected, answer_2=answer
                        )
                    }
                )

            results_ref.update({"tfidf": tfidf_cosine_sim(expected, actual)})
            results_new.update({"tfidf": tfidf_cosine_sim(expected, answer)})
            _, score_diff, scores = compare_answers_with_cross_encoder(
                query=question, answer_1=actual, answer_2=answer
            )
            results_ref.update({"cross_encoder": float(scores[1])})
            results_new.update({"cross_encoder": float(scores[0])})

            if acceptable == "Ja":
                score_threshold = (
                    results_ref["robbert-2022"] + results_ref["mini-lm-l6"]
                ) / 2
                if score_threshold > 75:
                    score_threshold = 75
            else:
                score_threshold = 75

            response = drop_keys(
                obj=response,
                keys_to_drop=["question", "answer", "page_content"],
            )
            results.append(
                {
                    "vraag": question,
                    "verwacht_antwoord": expected,
                    "antwoord": answer,
                    "response": response,
                    "duration": response.get("time_duration"),
                    "score_threshold": score_threshold,
                    "scores_ref": results_ref,
                    "scores_new": results_new,
                }
            )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
