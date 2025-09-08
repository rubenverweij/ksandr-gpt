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

                # Faalvormen samenvoegen
                combineer_faalvormen(directory)


def combineer_faalvormen(base_dir: str):
    """
    Verwerkt alle JSON-bestanden in de opgegeven base_dir en slaat de beschrijvingen op per AAD-nummer.
    De beschrijvingen worden opgeslagen in tekstbestanden per AAD-nummer.

    :param base_dir: Basis directory waar de AAD-mappen zich bevinden
    """
    # Door alle directories in de base_dir lopen
    for root, _, files in os.walk(base_dir):
        if "fail-types" in root:
            # Verkrijg het AAD-nummer uit het pad
            parts = root.split(os.sep)
            if len(parts) > 5:
                aad_nummer = parts[5]

                # Bestandsnaam voor het AAD-nummer
                output_file = f"faalvormen_{aad_nummer}.txt"
                beschrijvingen = []  # Lijst voor de beschrijvingen van dit AAD-nummer

                # Loop door de bestanden in de 'fail-types' directory
                for file in files:
                    if file == "main.json":
                        json_path = os.path.join(root, file)

                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                data = json.load(f)

                                # Verwerk de beschrijvingen in het JSON-bestand
                                for key, value in data.items():
                                    if key.startswith("Beschrijving faaltype"):
                                        titel = key.replace("Beschrijving ", "").strip()
                                        beschrijving = value.get(
                                            "Beschrijving", ""
                                        ).strip()

                                        if beschrijving:
                                            beschrijvingen.append(
                                                f"faaltype {titel}\n1) {beschrijving}\n"
                                            )

                        except Exception as e:
                            print(f"Fout bij verwerken van {json_path}: {e}")

                # Als er beschrijvingen zijn gevonden, schrijf ze naar het bestand
                if beschrijvingen:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(beschrijvingen))
                    print(f"✔ Beschrijvingen opgeslagen in '{output_file}'")
                else:
                    print(f"Geen beschrijvingen gevonden voor AAD {aad_nummer}.")


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
