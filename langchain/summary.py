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

    dossier = data.get("Dossier", {})
    populatie = data.get("Populatiegegevens", {}).get("Populatie per netbeheerder", [])
    component = COMPONENTS.get(aad_number, "Onbekend component")
    sentences = []
    # Publication info
    pub_date = dossier.get("Dossier", {}).get("Publicatiedatum")
    last_update = dossier.get("Dossier", {}).get("Laatste update")
    if pub_date or last_update:
        sentences.append(
            f"Het AAD dossier van de {component} is gepubliceerd op {pub_date or 'onbekend'} en voor het laatst gewijzigd op {last_update or 'onbekend'}."
        )
    # Component description
    description = dossier.get("Component", {}).get("Algemene productbeschrijving")
    if description:
        sentences.append(f"De omschrijving van de {component} is: {description}.")
    # Beheerteam
    beheerteam = dossier.get("Deelnemers", {}).get("Beheerteam", [])
    if beheerteam:
        team_names = ", ".join(item.get("text", "Onbekend") for item in beheerteam)
        sentences.append(
            f"In het beheerteam van dit AAD zitten de volgende personen: {team_names}."
        )
    # Deelnemende partijen & opdrachtgever
    deelnemende_partijen = dossier.get("Deelnemers", {}).get("Deelnemende partijen")
    opdrachtgever = dossier.get("Deelnemers", {}).get("Opdrachtgever")
    if deelnemende_partijen:
        sentences.append(
            f"De netbeheerders die deelnemen aan dit dossier zijn: {deelnemende_partijen}."
        )
    if opdrachtgever:
        sentences.append(f"De opdrachtgever van dit AAD is: {opdrachtgever}.")
    # Manufacturer
    fabrikant = dossier.get("Component", {}).get("Fabrikant")
    if fabrikant:
        sentences.append(f"De fabrikant van de {component} is {fabrikant}.")
    # Technical specifications
    tech_specs = dossier.get("Technische specificaties")
    if tech_specs:
        sentences.append(f"De technische specificatie is: {' '.join(tech_specs)}.")
    # Population per netbeheerder
    if populatie:
        pop_sentences = " ".join(
            f"Het aantal van {item.get('Netbeheerder', 'Onbekend')} is {item.get('Populatie', 'onbekend')} op peildatum {item.get('Peildatum', 'onbekend')}."
            for item in populatie
        )
        sentences.append(
            f"Het aantal van de {component} per netbeheerder is als volgt: {pop_sentences}"
        )
    # Combine all sentences into template
    template = "\n".join(sentences)

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
                        # clean_desc = re.sub(
                        #     r"\s+",
                        #     " ",
                        #     data["Beschrijving"].get("Beschrijving", "Onbekend"),
                        # ).strip()
                        descriptions.append(
                            f"""- Nummer {data["Beschrijving"].get("Nummer", "onbekend")} - {data["Beschrijving"].get("Naam", "onbekend")}. Hoe vaak komt deze faalvorm voor: {data["Beschrijving"].get("Gemiddeld aantal incidenten", "onbekend")}."""
                        )
                except Exception as e:
                    print(f"Fout bij verwerken van {fail_type_path}: {e}")
    return (
        template
        + "\n De faalvormen van component {component} zijn:"
        + "\n".join(sorted(descriptions, key=extract_number))
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


if __name__ == "__main__":
    base_dir = "/home/ubuntu/ksandr_files"
    aads, categories = get_aad_list_and_categories(base_dir)

    for aad in aads:
        # TODO add cat-2 handling keyerrors
        # for categorie in categories:
        print(
            maak_samenvatting_aad(
                base_dir=base_dir,
                aad_number=aad,
                category="cat-1",
            )
        )
