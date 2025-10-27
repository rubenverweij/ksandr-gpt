from neo4j import GraphDatabase
import json
import os

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))


def create_component_faaltype(
    session, aad_id, component_name, faaltype_data, file_path
):
    cypher = """
    MERGE (a:AAD {aad_id: $aad_id})
    MERGE (c:Component {naam: $component_name})
    MERGE (f:Faaltype {Nummer: $nummer})
      ON CREATE SET f.Naam = $naam,
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
    MERGE (c)-[:HEEFT_FAALTYPE]->(f)
    """
    session.run(
        cypher,
        {
            "aad_id": aad_id,
            "component_name": component_name,
            "nummer": faaltype_data.get("Nummer"),
            "naam": faaltype_data.get("Naam"),
            "beschrijving": faaltype_data.get("Beschrijving"),
            "mogelijk_gevolg": faaltype_data.get("Mogelijk gevolg"),
            "uitvoering": faaltype_data.get("Uitvoering"),
            "effect_op_subsysteem": faaltype_data.get("Effect op subsysteem"),
            "levensduur_bepalend": faaltype_data.get("Levensduur bepalend"),
            "status_oorzaak": faaltype_data.get("Status(Oorzaak)"),
            "status_gevolg": faaltype_data.get("Status(Gevolg)"),
            "oorzaak_detail": faaltype_data.get("Oorzaak(detail)"),
            "oorzaak_generiek": faaltype_data.get("Oorzaak(Generiek)"),
            "afhankelijkheid": faaltype_data.get("Afhankelijkheid omgevingscondities"),
            "faalindicatoren": faaltype_data.get("Faalindicator(en)"),
            "faalcurve": faaltype_data.get("Faalcurve"),
            "faaltempo": faaltype_data.get("Faaltempo"),
            "gemiddeld_aantal_incidenten": faaltype_data.get(
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
                        faaltype_data = data.get("Beschrijving")
                    if faaltype_data:
                        print(f"Processing {file_path}... for component {component}")
                        create_component_faaltype(
                            session, component_id, component, faaltype_data, file_path
                        )

driver.close()
