import json
from bs4 import BeautifulSoup
import re
from patterns import COMPONENTS
import os


def clean_html(value):
    """Maak HTML-schoon en verwijder tags/lege velden."""
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        # Verwijder HTML-tags
        cleaned = BeautifulSoup(value, "html.parser").get_text(separator=" ")
        # Verwijder overtollige witruimte
        return re.sub(r"\s+", " ", cleaned).strip()
    if isinstance(value, dict):
        value.pop("tags", None)
        return {k: clean_html(v) for k, v in value.items() if clean_html(v) is not None}
    if isinstance(value, list):
        lijst = [clean_html(v) for v in value if clean_html(v) is not None]
        return lijst if lijst else None
    return value


def extract_number(item: str) -> int:
    """Zoek het eerste getal in de string en gebruik dat als sorteerwaarde."""
    match = re.search(r"(\d+)", item)
    return int(match.group(1)) if match else float("inf")


def maak_samenvatting_aad(base_dir: str, aad_number: str, category: str):
    """
    Loops through AAD numbers and categories, processes all main.json files under 'fail-types',
    and creates a combined file with descriptions from all fail-types.
    :param base_dir: The root directory where AADs are located.
    """
    # Set the path for the current category and AAD
    category_path = os.path.join(
        base_dir, "aads", str(aad_number), category, "fail-types"
    )
    aad_path = os.path.join(base_dir, "aads", str(aad_number), category, "main.json")
    with open(aad_path, "r") as file:
        data = clean_html(json.load(file))
    dossier = data["Dossier"]
    populatie = data["Populatiegegevens"]["Populatie per netbeheerder"]
    component = COMPONENTS[aad_number]
    template = f"""Het AAD dossier van de {component} is gepubliceerd op {dossier["Dossier"]["Publicatiedatum"]} en voor het laatst gewijzigd op {dossier["Dossier"]["Laatste update"]}.
                De omschrijving van de {component} is: {dossier["Component"]["Algemene productbeschrijving"]}. 
                In het beheerteam van dit AAD zitten de volgende personen: {", ".join(item["text"] for item in data["Dossier"]["Deelnemers"]["Beheerteam"])}.
                De netbeheerders die deelnemen aan dit dossier zijn: {dossier["Deelnemers"]["Deelnemende partijen"]}. De opdrachtgever van dit AAD is: {dossier["Deelnemers"]["Opdrachtgever"]}.
                De fabrikant van de {component} is {dossier["Component"]["Fabrikant"]}. 
                De technische specificatie is: {"".join(dossier["Technische specificaties"])}.
                Het aantal van de {component} per netbeheerder is als volgt: {" ".join(f"Het aantal van {item['Netbeheerder']} is {item['Populatie']} op peildatum {item['Peildatum']}." for item in populatie)}.
                De faalvormen van component {component} zijn:"""
    # Check if fail-types directory exists for the given AAD and category
    if os.path.exists(category_path):
        descriptions = []  # List to store all descriptions for the AAD/category combination
        # Loop through all subdirectories (fail-types)
        for fail_type_folder in os.listdir(category_path):
            fail_type_path = os.path.join(category_path, fail_type_folder, "main.json")
            # Check if main.json exists in this fail-type folder
            if os.path.isfile(fail_type_path):
                try:
                    with open(fail_type_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        descriptions.append(
                            f"""- Nummer {data["Beschrijving"]["Nummer"]} -  Naam: {data["Beschrijving"]["Naam"]}. Hoe vaak komt deze faalvorm voor: {data["Beschrijving"]["Gemiddeld aantal incidenten"]}. De beschrijving is: {data["Beschrijving"]["Beschrijving"]}"""
                        )
                except Exception as e:
                    print(f"Fout bij verwerken van {fail_type_path}: {e}")
    return template + "\n".join(sorted(maak_samenvatting_aad(), key=extract_number))


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


if __name__ == "__main__":
    print(
        maak_samenvatting_aad(
            base_dir="/home/ubuntu/ksandr_files", aad_number="10535", category="cat-1"
        )
    )
