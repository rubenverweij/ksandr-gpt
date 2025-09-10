import os
import json
import argparse
import re
import string
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


# Functie om door een directory te lopen en alle JSON-bestanden te verwerken
def clean_json_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):  # Alleen JSON-bestanden
                file_path = os.path.join(root, file)
                print(f"Verwerken bestand: {file_path}")

                # Controleren of het bestand leeg is
                if os.stat(file_path).st_size == 0:
                    print(f"Bestand overgeslagen (leeg): {file_path}")
                    continue  # Sla lege bestanden over

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Fout bij het lezen van JSON uit {file_path}: {e}")
                    continue
                except Exception as e:
                    print(f"Onverwachte fout bij {file_path}: {e}")
                    continue

                # Het hernoemen van de sleutels op basis van het pad van het bestand
                renamed_data = rename_json_keys_based_on_file_path(data, file_path)

                # Opschonen van het JSON-bestand
                cleaned_data = clean_html(renamed_data)

                # Opsplitsen van de JSON op niveau 2 als er geen "fail-types" is
                split_data = split_json_by_level_2(cleaned_data, file_path)

                # Als de JSON gesplitst is, slaan we meerdere bestanden op
                if split_data:
                    # Controleer of split_data een lijst van tuples is
                    if isinstance(split_data, list) and all(
                        isinstance(item, tuple) and len(item) == 2
                        for item in split_data
                    ):
                        for key, split_json in split_data:
                            if key is not None:  # We slaan alleen niet-None keys op
                                # Opslaan van de gesplitste JSON in een nieuw bestand
                                cleaned_filename = key.translate(
                                    str.maketrans(
                                        {
                                            " ": "_",
                                            **{char: "" for char in string.punctuation},
                                        }
                                    )
                                )
                                split_file_name = (
                                    f"{file.split('.')[0]}_{cleaned_filename}.json"
                                )
                                split_file_path = os.path.join(root, split_file_name)
                                with open(split_file_path, "w", encoding="utf-8") as f:
                                    json.dump(
                                        split_json, f, ensure_ascii=False, indent=2
                                    )
                                print(
                                    f"Bestand opgesplitst en opgeslagen: {split_file_path}"
                                )
                    else:
                        print(
                            f"Fout: split_data heeft niet de verwachte structuur voor {file_path}"
                        )
                else:
                    # Als er geen splitsing was, slaan we het bestand op in dezelfde locatie
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
                    print(f"Bestand opgeschoond: {file_path}")

                with open(file_path, "r", encoding="utf-8") as f:
                    cleaned_data = json.load(f)
                json_string = json_to_single_occurrence_string(cleaned_data)
                file_path_without_extension, _ = os.path.splitext(file_path)
                with open(f"{file_path_without_extension}.txt", "w") as file:
                    file.write(json_string)

    # Faalvormen samenvoegen
    combine_json_files_for_aads(directory)


# Functie om HTML-tags uit de waarde te verwijderen, lege waarden te verwijderen, onnodige spaties te verwijderen, en "tags" attributen te verwijderen
def clean_html(value):
    if value is None or value == "" or value == [] or value == {}:
        return None

    if isinstance(value, str):
        # Verwijder HTML-tags
        cleaned_value = BeautifulSoup(value, "html.parser").get_text(separator=" ")
        # Verwijder onnodige spaties (aan begin, eind en tussenin)
        return re.sub(r"\s+", " ", cleaned_value).strip()

    if isinstance(value, dict):
        # Verwijder "tags" sleutel als het aanwezig is
        value.pop("tags", None)
        return {k: clean_html(v) for k, v in value.items() if clean_html(v) is not None}

    if isinstance(value, list):
        cleaned_list = [clean_html(v) for v in value if clean_html(v) is not None]
        return cleaned_list if cleaned_list else None

    return value  # Voor andere types zoals int, float, etc.


def rename_json_keys_based_on_file_path(json_data, file_path):
    directory_parts = file_path.split(os.sep)

    if len(directory_parts) > 5:
        aad = directory_parts[5]
    else:
        aad = ""
        print(
            f"Waarschuwing: Ongeldige padstructuur, 'aad' niet gevonden in {file_path}"
        )

    new_key_prefix = AADS.get(aad, "")
    if "fail-types" in directory_parts:
        new_key_prefix = f"faalvorm {new_key_prefix}" if new_key_prefix else ""

    print(f"Nieuwe key {new_key_prefix} voor aad: {aad}")

    def rename_keys(data, level=1):
        if isinstance(data, dict):
            renamed_data = {}
            for key, value in data.items():
                # Controleer of we in een 'fail-types' pad zijn en of we op niveau 1 zitten
                if "fail-types" in directory_parts:
                    # Prefix alleen toevoegen op niveau 1 als we 'fail-types' in het pad hebben
                    new_key = (
                        f"{key} {new_key_prefix}"
                        if level <= 1 and new_key_prefix
                        else key
                    )
                else:
                    # Voor andere gevallen: prefix toepassen op niveau 1 en 2
                    new_key = (
                        f"{key} {new_key_prefix}"
                        if level <= 2 and new_key_prefix
                        else key
                    )

                # Verwerk de waarde (die mogelijk een geneste structuur heeft)
                renamed_data[new_key] = rename_keys(value, level + 1)
            return renamed_data
        elif isinstance(data, list):
            return [rename_keys(item, level) for item in data]
        return data

    return rename_keys(json_data)


# Functie om de JSON te splitsen op basis van het tweede niveau attribuut
def split_json_by_level_2(json_data, base_file_path):
    # Alleen uitvoeren als we geen "fail-types" in het bestandspad hebben
    if "fail-types" in base_file_path:
        return None  # Geen splitsing nodig voor "fail-types" bestanden

    split_files = []
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            # Controleer of de waarde op niveau 2 zit
            if isinstance(value, dict):  # We splitsen alleen niveau 2 attributen
                # Maak een nieuw bestand per niveau 2 attribuut
                split_file = {
                    key: value
                }  # Zorg dat dit een dictionary is met key en waarde
                split_files.append(
                    (key, split_file)
                )  # Voeg een tuple toe met key en de bijbehorende data
    return split_files


def json_to_single_occurrence_string(json_obj):
    def flatten_json(d, parent_key="", sep=" "):
        items = []
        for k, v in d.items():
            if k not in items:
                items.append((k, ""))
            new_key = f"{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_json(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    items.extend(flatten_json(item, f"{idx}", sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flattened = flatten_json(json_obj)
    json_str = json.dumps(flattened)
    json_str = re.sub(r'["]', "", json_str)
    json_str = re.sub(r':""', ":", json_str)
    json_str = re.sub(r"[{}[\]]", "", json_str)
    json_str = re.sub(r" , ", " ", json_str)
    return json_str


def combine_json_files_for_aads(base_dir: str):
    """
    Loops through AAD numbers and categories, processes all main.json files under 'fail-types',
    and creates a combined file with descriptions from all fail-types.

    :param base_dir: The root directory where AADs are located.
    """
    # Get the list of AAD numbers and categories dynamically
    aad_list, categories = get_aad_list_and_categories(base_dir)

    for aad_number in aad_list:
        for category in categories:
            # Set the path for the current category and AAD
            category_path = os.path.join(
                base_dir, "aads", str(aad_number), category, "fail-types"
            )

            # Check if fail-types directory exists for the given AAD and category
            if os.path.exists(category_path):
                output_file = os.path.join(
                    category_path, f"faalvormen_{aad_number}_{category}.txt"
                )
                descriptions = []  # List to store all descriptions for the AAD/category combination
                faalvorm_count = 1

                # Loop through all subdirectories (fail-types)
                for fail_type_folder in os.listdir(category_path):
                    fail_type_path = os.path.join(
                        category_path, fail_type_folder, "main.json"
                    )

                    # Check if main.json exists in this fail-type folder
                    if os.path.isfile(fail_type_path):
                        try:
                            with open(fail_type_path, "r", encoding="utf-8") as f:
                                data = json.load(f)

                                # Process all descriptions in the JSON file
                                for key, value in data.items():
                                    if key.startswith("Beschrijving faalvorm"):
                                        if len(descriptions) == 0:
                                            title = key.replace(
                                                "Beschrijving faalvorm", ""
                                            ).strip()
                                            descriptions.append(
                                                f"Opsomming lijst met faalvormen {title}"
                                            )

                                        description = value.get(
                                            "Beschrijving", ""
                                        ).strip()

                                        incidenten = value.get(
                                            "Gemiddeld aantal incidenten", ""
                                        ).strip()

                                        if description:
                                            descriptions.append(
                                                f"{faalvorm_count}. {description}. Gemiddeld aantal incidenten faalvorm: {incidenten}"
                                            )
                                            faalvorm_count += 1

                        except Exception as e:
                            print(f"Fout bij verwerken van {fail_type_path}: {e}")

                # If descriptions were found, write them to a file
                if descriptions:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(descriptions))
                    print(f"✔ Beschrijvingen opgeslagen in '{output_file}'")
                else:
                    print(
                        f"Geen beschrijvingen gevonden voor AAD {aad_number} en categorie {category}."
                    )
            else:
                print(
                    f"De directory 'fail-types' bestaat niet voor AAD {aad_number} en categorie {category}."
                )


def get_aad_list_and_categories(base_dir: str):
    """
    Dynamically retrieves the AAD numbers and categories from the directory structure.

    :param base_dir: The root directory where AADs are located.
    :return: A tuple of (aad_list, categories)
    """
    aad_list = set()
    categories = set()
    for root, _, _ in os.walk(base_dir):
        if "fail-types" in root:
            parts = root.split(os.sep)
            if len(parts) > 3:
                aad_number = parts[5]  # AAD number is at index 3
                category = parts[6]  # Category (e.g., cat-1, cat-2) is at index 4
                aad_list.add(aad_number)
                categories.add(category)
    return list(aad_list), list(categories)


# Hoofdfunctie om de commandoregelargumenten te verwerken
def main():
    parser = argparse.ArgumentParser(
        description="Verwijder HTML-tags, onnodige spaties, verwijder lege of 'null' waarden, verwijder 'tags' attributen en hernoem de sleutels van JSON-bestanden."
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
