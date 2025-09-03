import os
import json
import argparse
import re
from bs4 import BeautifulSoup


AADS = {
    "10535": "LK ELA12 schakelinstallatie",
    "10536": "ABB VD4 vaccuum vermogensschakelaar",
    "10540": "Eaton L-SEP installatie",
    "10542": "Siemens NXplusC schakelaar",
    "10545": "Siemens 8DJH schakelaar",
    "10546": "Eaton FMX schakelinstallatie",
    "1555": "Merlin Gerin RM6 schakelaar",
    "1556": "Hazemeijer CONEL schakelinstallatie",
    "1557": "Eaton 10 kV COQ schakelaar",
    "1558": "Eaton Capitole schakelaar",
    "2059": "Eaton Xiria schakelinstallatie",
    "2061": "Eaton Holec SVS schakelaar",
    "2963": "MS/LS distributie transformator",
    "318": "Eaton Magnefix MD/MF schakelinstallatie",
    "655": "ABB DR12 schakelaar",
    "8825": "ABB Safe schakelinstallatie",
    "8827": "kabelmoffen",
    "9026": "Eaton MMS schakelinstallatie",
    "9027": "ABB BBC DB10 schakelaar",
    "9028": "HS MS vermogens transformator",
}


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


# Functie om de sleutels van de eerste en tweede niveau te hernoemen op basis van het pad van het bestand
def rename_json_keys_based_on_file_path(json_data, file_path):
    # Haal het vijfde element uit het pad (bijv. "2061")
    directory_parts = file_path.split(os.sep)
    aad = directory_parts[5]
    if aad in AADS.keys():
        new_key_prefix = AADS[aad]  # Het 5e element is op index 4
        if "fail-types" in directory_parts:
            new_key_prefix = f"faaltype {AADS[aad]}"
    else:
        new_key_prefix = ""

    print(f"Nieuwe key {new_key_prefix} voor aad: {aad}")

    # Functie om sleutels te hernoemen op basis van de nieuwe prefix
    def rename_keys(data, level=1):
        if isinstance(data, dict):
            renamed_data = {}
            for key, value in data.items():
                # Bij het eerste en tweede niveau, hernoem de sleutel
                if "fail-types" in directory_parts:
                    if level <= 1:
                        if len(new_key_prefix) > 0:
                            new_key = f"{key} {new_key_prefix}"  # Gebruik de prefix voor hernoemen
                        else:
                            new_key = key
                else:
                    if level <= 2:
                        if len(new_key_prefix) > 0:
                            new_key = f"{key} {new_key_prefix}"  # Gebruik de prefix voor hernoemen
                        else:
                            new_key = key
                    else:
                        new_key = key  # Andere niveaus behouden de originele sleutel
                renamed_data[new_key] = rename_keys(value, level + 1)
            return renamed_data
        elif isinstance(data, list):
            return [rename_keys(item, level) for item in data]
        else:
            return data

    # Pas de hernoeming toe
    return rename_keys(json_data)


# Functie om door een directory te lopen en alle JSON-bestanden te verwerken
def clean_json_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):  # Alleen JSON-bestanden
                file_path = os.path.join(root, file)
                print(f"Verwerken bestand: {file_path}")

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Het hernoemen van de keys op basis van het pad van het bestand
                renamed_data = rename_json_keys_based_on_file_path(data, file_path)

                # Opschonen van het JSON-bestand
                cleaned_data = clean_html(renamed_data)

                # Opslaan van de opgeschoonde JSON in dezelfde directory
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
                print(f"Bestand opgeschoond: {file_path}")


# Hoofdfunctie om de commandoregelargumenten te verwerken
def main():
    # Argument parser aanmaken
    parser = argparse.ArgumentParser(
        description="Verwijder HTML-tags, onnodige spaties, verwijder lege of 'null' waarden, verwijder 'tags' attributen en hernoem de sleutels van JSON-bestanden."
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
