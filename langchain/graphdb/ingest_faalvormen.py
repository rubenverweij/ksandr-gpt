from neo4j import GraphDatabase
import json
import os
import re

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))


def extract_nummer_info(nummer_str):
    """Extraheer het volledige nummer en het numerieke deel uit iets als 'MAG-1'."""
    if not nummer_str:
        return None, None
    match = re.match(r"([A-Za-z\-]+)-?(\d+)?", nummer_str.strip())
    if match:
        prefix = match.group(1)  # bijv. 'MAG'
        nummer = match.group(2)  # bijv. '1'
        return prefix, int(nummer) if nummer else None
    return nummer_str, None


def create_component_faalvorm(
    session, aad_id, component_name, faalvorm_data, file_path
):
    nummer_str = faalvorm_data.get("Nummer")
    prefix, nummer_int = extract_nummer_info(nummer_str)
    cypher = """
    MERGE (a:AAD {aad_id: $aad_id})
    MERGE (c:Component {naam: $component_name})
    MERGE (f:Faalvorm {Nummer: $nummer})
      ON CREATE SET f.Naam = $naam,
                    f.NummerInt = $nummer_int,
                    f.Prefix = $prefix,
                    f.Beschrijving = $beschrijving,
                    f.MogelijkGevolg = $mogelijk_gevolg,
                    f.Uitvoering = $uitvoering,
                    f.EffectOpSubsysteem = $effect_op_subsysteem,
                    f.LevensduurBepalend = $levensduur_bepalend,
                    f.StatusOorzaak = $status_oorzaak,
                    f.StatusGevolg = $status_gevolg,
                    f.OorzaakDetail = $oorzaak_detail,
                    f.OorzaakGeneriek = $oorzaak_generiek,
                    f.AfhankelijkheidOmgevingscondities = $afhankelijkheid,
                    f.Faalindicatoren = $faalindicatoren,
                    f.Faalcurve = $faalcurve,
                    f.Faaltempo = $faaltempo,
                    f.GemiddeldAantalIncidenten = $gemiddeld_aantal_incidenten,
                    f.Bestandspad = $bestandspad
    MERGE (a)-[:HEEFT_COMPONENT]->(c)
    MERGE (c)-[:HEEFT_FAALVORM]->(f)
    """
    session.run(
        cypher,
        {
            "aad_id": aad_id,
            "component_name": component_name,
            "nummer": nummer_str,
            "nummer_int": nummer_int,
            "prefix": prefix,
            "naam": faalvorm_data.get("Naam"),
            "beschrijving": faalvorm_data.get("Beschrijving"),
            "mogelijk_gevolg": faalvorm_data.get("Mogelijk gevolg"),
            "uitvoering": faalvorm_data.get("Uitvoering"),
            "effect_op_subsysteem": faalvorm_data.get("Effect op subsysteem"),
            "levensduur_bepalend": faalvorm_data.get("Levensduur bepalend"),
            "status_oorzaak": faalvorm_data.get("Status(Oorzaak)"),
            "status_gevolg": faalvorm_data.get("Status(Gevolg)"),
            "oorzaak_detail": faalvorm_data.get("Oorzaak(detail)"),
            "oorzaak_generiek": faalvorm_data.get("Oorzaak(Generiek)"),
            "afhankelijkheid": faalvorm_data.get("Afhankelijkheid omgevingscondities"),
            "faalindicatoren": faalvorm_data.get("Faalindicator(en)"),
            "faalcurve": faalvorm_data.get("Faalcurve"),
            "faaltempo": faalvorm_data.get("Faaltempo"),
            "gemiddeld_aantal_incidenten": faalvorm_data.get(
                "Gemiddeld aantal incidenten"
            ),
            "bestandspad": file_path,
        },
    )


COMPONENTS = {
    "10535": "LK ELA12 schakelinstallatie",
    "10536": "ABB VD4 vaccuum vermogensschakelaar",
    "10540": "Eaton L-SEP installatie",
    "10542": "Siemens NXplusC schakelaar",
    "10545": "Siemens 8DJH schakelaar",
    "10546": "Eaton FMX schakelinstallatie",
    "10551": "Laagspanning",
    "1555": "Merlin Gerin RM6 schakelaar",
    "1556": "Hazemeijer CONEL schakelinstallatie",
    "1557": "Eaton 10 kV COQ schakelaar",
    "1558": "Eaton Capitole schakelaar",
    "2059": "Eaton Xiria schakelinstallatie",
    "2061": "Eaton Holec SVS schakelaar",
    "2963": "MS/LS distributie transformator",
    "318": "Eaton Magnefix MD MF schakelinstallatie",
    "655": "ABB DR12 schakelaar",
    "8825": "ABB Safe safeplus schakelinstallatie",
    "8827": "MS kabelmoffen",
    "9026": "Eaton MMS schakelinstallatie",
    "9027": "ABB BBC DB10 schakelaar",
    "9028": "HS MS vermogens transformator",
}

with driver.session() as session:
    for component_id, component in COMPONENTS.items():
        json_folder = f"/home/ubuntu/ksandr_files/aads/{component_id}/cat-1/fail-types/"
        for root, dirs, files in os.walk(json_folder):
            for filename in files:
                if filename.endswith(".json"):
                    file_path = os.path.join(root, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        faalvorm_data = data.get("Beschrijving")
                    if faalvorm_data:
                        print(f"Processing {file_path}... for component {component}")
                        create_component_faalvorm(
                            session, component_id, component, faalvorm_data, file_path
                        )

driver.close()
