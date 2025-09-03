import os
import json
import argparse
from bs4 import BeautifulSoup


# Functie om HTML-tags uit de waarde te verwijderen
def clean_html(value):
    if isinstance(value, str):
        return BeautifulSoup(value, "html.parser").get_text(separator=" ")
    elif isinstance(value, dict):
        return {k: clean_html(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_html(v) for v in value]
    else:
        return value


# Functie om door een directory te lopen en alle JSON-bestanden te verwerken
def clean_json_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):  # Alleen JSON-bestanden
                file_path = os.path.join(root, file)
                print(f"Verwerken bestand: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cleaned_data = clean_html(data)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
                print(f"Bestand opgeschoond: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verwijder HTML-tags uit JSON-bestanden in een opgegeven directory."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Pad naar de directory waar de JSON-bestanden zich bevinden",
    )
    args = parser.parse_args()
    clean_json_in_directory(args.directory)


if __name__ == "__main__":
    main()
