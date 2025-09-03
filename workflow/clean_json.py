import os
import json
import argparse
import re
from bs4 import BeautifulSoup


# Functie om HTML-tags uit de waarde te verwijderen, lege waarden te verwijderen, onnodige spaties te verwijderen, en "tags" attributen te verwijderen
def clean_html(value):
    # Vervang None (null in JSON) met de opgegeven tekst
    if value is None:
        return None

    # Verwijder lege waarden zoals lege strings, lege lijsten, en lege dictionaries
    elif value == "" or value == [] or value == {}:
        return None

    # Als het een string is, haal de HTML-tags eruit en verwijder onnodige spaties
    elif isinstance(value, str):
        # Verwijder HTML-tags
        cleaned_value = BeautifulSoup(value, "html.parser").get_text(separator=" ")
        # Verwijder onnodige spaties (aan begin, eind en tussenin)
        cleaned_value = re.sub(r"\s+", " ", cleaned_value).strip()
        return cleaned_value

    # Als het een dictionary is, pas dezelfde bewerking toe op elk item en verwijder lege items
    elif isinstance(value, dict):
        # Verwijder "tags" sleutel als het aanwezig is
        if "tags" in value:
            del value["tags"]

        # Pas de bewerkingen toe op de resterende items
        cleaned_dict = {k: clean_html(v) for k, v in value.items()}
        # Verwijder items die None zijn (bijv. als ze leeg of null waren)
        return {k: v for k, v in cleaned_dict.items() if v is not None}

    # Als het een lijst is, pas dezelfde bewerking toe op elk item
    elif isinstance(value, list):
        cleaned_list = [clean_html(v) for v in value]
        # Verwijder items die None zijn en zorg ervoor dat lege lijsten ook worden verwijderd
        cleaned_list = [v for v in cleaned_list if v is not None]
        # Verwijder de lijst zelf als deze leeg is
        if not cleaned_list:
            return None
        return cleaned_list

    # Als het een ander type is (int, float, enz.), laat het dan zoals het is
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

                # Opschonen van het JSON-bestand
                cleaned_data = clean_html(data)

                # Opslaan van de opgeschoonde JSON in dezelfde directory
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
                print(f"Bestand opgeschoond: {file_path}")


# Hoofdfunctie om de commandoregelargumenten te verwerken
def main():
    # Argument parser aanmaken
    parser = argparse.ArgumentParser(
        description="Verwijder HTML-tags, onnodige spaties, verwijder lege of 'null' waarden en verwijder 'tags' attributen uit JSON-bestanden."
    )

    # Voeg een argument toe voor de directory
    parser.add_argument(
        "directory",
        type=str,
        help="Pad naar de directory waar de JSON-bestanden zich bevinden",
    )

    # Haal de argumenten op
    args = parser.parse_args()

    # Roep de functie aan met de opgegeven directory
    clean_json_in_directory(args.directory)


if __name__ == "__main__":
    main()
