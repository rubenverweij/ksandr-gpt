import os
import json
import argparse
from bs4 import BeautifulSoup


# Functie om HTML-tags uit de waarde te verwijderen en lege of null waarden te verwijderen
def clean_html(value):
    # Vervang None (null in JSON) met de opgegeven tekst
    if value is None:
        return None

    # Verwijder lege waarden zoals lege strings, lege lijsten, en lege dictionaries
    elif value == "" or value == [] or value == {}:
        return None

    # Als het een string is, haal de HTML-tags eruit
    elif isinstance(value, str):
        return BeautifulSoup(value, "html.parser").get_text(separator=" ")

    # Als het een dictionary is, pas dezelfde bewerking toe op elk item en verwijder lege items
    elif isinstance(value, dict):
        cleaned_dict = {k: clean_html(v) for k, v in value.items()}
        return {k: v for k, v in cleaned_dict.items() if v is not None}

    # Als het een lijst is, pas dezelfde bewerking toe op elk item
    elif isinstance(value, list):
        cleaned_list = [clean_html(v) for v in value]
        cleaned_list = [v for v in cleaned_list if v is not None]
        if not cleaned_list:
            return None
        return cleaned_list
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


# Hoofdfunctie om de commandoregelargumenten te verwerken
def main():
    parser = argparse.ArgumentParser(
        description="Verwijder HTML-tags en verwijder lege of 'null' waarden uit JSON-bestanden."
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
