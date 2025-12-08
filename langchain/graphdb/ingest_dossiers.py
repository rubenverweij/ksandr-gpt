from neo4j import GraphDatabase
import json
import os
from bs4 import BeautifulSoup
import re

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))


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


def create_component_faalvorm(session, aad_id, component_id, faalvorm_data, file_path):
    nummer_str = faalvorm_data.get("Nummer")
    prefix, nummer_int = extract_nummer_info(nummer_str)
    cypher = """
    MERGE (d:dossier {aad_id: $aad_id})
    MERGE (c:component {component_id: $component_id})
    MERGE (f:faalvorm {faalvorm_id: $faalvorm_id})
      ON CREATE SET f.Naam = $naam,
                    f.NummerInt = $nummer_int,
                    f.Prefix = $prefix,
                    f.Beschrijving = $beschrijving,
                    f.MogelijkGevolg = $mogelijk_gevolg,
                    f.Uitvoering = $uitvoering,
                    f.EffectOpSubsysteem = $effect_op_subsysteem,
                    f.LevensduurBepalend = $niet_repareerbaar,
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
    MERGE (d)-[:HEEFT_COMPONENT]->(c)
    MERGE (c)-[:HEEFT_FAALVORM]->(f)
    """
    session.run(
        cypher,
        {
            "aad_id": aad_id,
            "component_id": component_id,
            "faalvorm_id": nummer_str,
            "naam": faalvorm_data.get("Naam"),
            "nummer_int": nummer_int,
            "prefix": prefix,
            "beschrijving": faalvorm_data.get("Beschrijving"),
            "mogelijk_gevolg": faalvorm_data.get("Mogelijk gevolg"),
            "uitvoering": faalvorm_data.get("Uitvoering"),
            "effect_op_subsysteem": faalvorm_data.get("Effect op subsysteem"),
            "niet_repareerbaar": faalvorm_data.get("Levensduur bepalend"),
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


def clean_key(key):
    return key.lower().replace(" ", "_").replace("/", "_")


def optional(obj, key, default=None):
    """Veilig een key ophalen als deze bestaat."""
    return obj.get(key, default) if isinstance(obj, dict) else default


def merge_node(tx, label, key, props):
    query = f"""
    MERGE (n:{label} {{{key}: $keyval}})
    SET n += $props
    """
    tx.run(query, keyval=props[key], props=props)


def merge_relation(tx, a_label, a_key, a_val, rel, b_label, b_key, b_val):
    query = f"""
    MATCH (a:{a_label} {{{a_key}: $a_val}})
    MATCH (b:{b_label} {{{b_key}: $b_val}})
    MERGE (a)-[:{rel}]->(b)
    """
    tx.run(query, a_val=a_val, b_val=b_val)


def ingest_dossier(data, aad_id, component_id):
    with driver.session() as session:
        dossier_json = data.get("Dossier", {}).get("Dossier", {})
        dossier_props = {
            "aad_id": aad_id,
            "naam": optional(dossier_json, "Naam"),
            "publicatiedatum": optional(dossier_json, "Publicatiedatum"),
            "laatste_update": optional(dossier_json, "Laatste update"),
            "leeswijzer": optional(dossier_json, "Leeswijzer"),
        }
        session.execute_write(merge_node, "dossier", "aad_id", dossier_props)
        # ---------------------------------------------------------
        # Beheerteam → person nodes
        # ---------------------------------------------------------
        deelnemers = data.get("Dossier", {}).get("Deelnemers", {})
        beheerteam = deelnemers.get("Beheerteam", [])
        for member in beheerteam:
            person_id = member.get("link", "").replace("/profile/", "")
            if not person_id:
                continue
            person_props = {
                "id": person_id,
                "link": optional(member, "link"),
                "naam": optional(member, "text"),
            }
            session.execute_write(merge_node, "persoon", "id", person_props)
            session.execute_write(
                merge_relation,
                "dossier",
                "aad_id",
                aad_id,
                "heeft_beheerteam_lid",
                "persoon",
                "id",
                person_id,
            )
        # ---------------------------------------------------------
        # Component → component node
        # ---------------------------------------------------------
        comp = data.get("Dossier", {}).get("Component", {})
        comp_props = {"component_id": component_id}
        for k, v in comp.items():
            comp_props[clean_key(k)] = v
        session.execute_write(merge_node, "component", "component_id", comp_props)
        session.execute_write(
            merge_relation,
            "dossier",
            "aad_id",
            aad_id,
            "heeft_component",
            "component",
            "component_id",
            component_id,
        )
        # ---------------------------------------------------------
        # Media → document nodes
        # ---------------------------------------------------------
        media_groups = data.get("Dossier", {}).get("Media", [])
        for group in media_groups:
            for item in group:
                doc_id = f"media_{item.get('aadDocumentId')}"
                props = {"id": doc_id}
                for k, v in item.items():
                    props[clean_key(k)] = v
                session.execute_write(merge_node, "document", "id", props)
                session.execute_write(
                    merge_relation,
                    "dossier",
                    "aad_id",
                    aad_id,
                    "heeft_document",
                    "document",
                    "id",
                    doc_id,
                )
        # ---------------------------------------------------------
        # Overige bestanden → document nodes
        # ---------------------------------------------------------
        overige = data.get("Dossier", {}).get("Overige bestanden", [])
        for group in overige:
            for item in group:
                doc_id = f"doc_{item.get('documentId')}"
                props = {"id": doc_id}
                for k, v in item.items():
                    props[clean_key(k)] = v
                session.execute_write(merge_node, "document", "id", props)
                session.execute_write(
                    merge_relation,
                    "dossier",
                    "aad_id",
                    aad_id,
                    "heeft_document",
                    "document",
                    "id",
                    doc_id,
                )
        # ---------------------------------------------------------
        # Populatiegegevens → populatie + netbeheerder nodes
        # ---------------------------------------------------------
        pop_entries = data.get("Populatiegegevens", {}).get(
            "Populatie per netbeheerder", []
        )
        for p in pop_entries:
            nb_name = p.get("Netbeheerder") or "onbekend"
            nb_id = f"netbeheerder_{nb_name}".lower()
            # netbeheerder node
            nb_props = {"id": nb_id, "naam": nb_name}
            session.execute_write(merge_node, "netbeheerder", "id", nb_props)
            # populatie node
            pop_id = f"pop_{nb_name}_{aad_id}_{p.get('Populatie')}".lower()
            pop_props = {"id": pop_id}
            for k, v in p.items():
                pop_props[clean_key(k)] = v
            session.execute_write(merge_node, "populatie", "id", pop_props)
            # relaties
            session.execute_write(
                merge_relation,
                "dossier",
                "aad_id",
                aad_id,
                "heeft_populatie",
                "populatie",
                "id",
                pop_id,
            )
            session.execute_write(
                merge_relation,
                "netbeheerder",
                "id",
                nb_id,
                "heeft_populatie",
                "populatie",
                "id",
                pop_id,
            )
        # ---------------------------------------------------------
        # Populatiegegevens per type → populatie + netbeheerder nodes
        # ---------------------------------------------------------
        # pop_entries = data.get("Populatiegegevens", {}).get("Populatie per type", [])
        # for p in pop_entries:
        #     type = p.get("Type") or "onbekend"
        #     populatie_gegevens = p.get("Populatiegegevens")
        #     for populatie in populatie_gegevens:
        #         nb_name = populatie.get("Netbeheerder") or "onbekend"
        #         nb_id = f"netbeheerder_{nb_name}".lower()
        #         bouwjaren = populatie.get("Bouwjaren")
        #         for bouwjaar in bouwjaren:
        #             # populatie node
        #             try:
        #                 year = int(bouwjaar.get("Bouwjaar"))
        #             except (ValueError, TypeError):
        #                 year = None
        #             pop_id = f"pop_{nb_name}_{aad_id}_{type}_{year}".lower()
        #             pop_props = {"id": pop_id, "type": type, "bouwjaar": year}
        #             for k, v in bouwjaar.items():
        #                 if isinstance(v, int):
        #                     if v > 0:
        #                         print(f"Ingesting {pop_props} value: {k}={v}")
        #                         pop_props[clean_key(k)] = v
        #                 session.execute_write(merge_node, "populatie", "id", pop_props)
        #                 # relaties
        #                 session.execute_write(
        #                     merge_relation,
        #                     "dossier",
        #                     "aad_id",
        #                     aad_id,
        #                     "heeft_populatie",
        #                     "populatie",
        #                     "id",
        #                     pop_id,
        #                 )
        #                 session.execute_write(
        #                     merge_relation,
        #                     "netbeheerder",
        #                     "id",
        #                     nb_id,
        #                     "heeft_populatie",
        #                     "populatie",
        #                     "id",
        #                     pop_id,
        #                 )
        # ---------------------------------------------------------
        # Onderhoudsbeleid → beleid nodes
        # ---------------------------------------------------------
        beleidgroups = data.get("Onderhoudsbeleid", {})
        for key, group in beleidgroups.items():
            for item in group:
                if not item:
                    continue
                pol_id = f"pol_{abs(hash(str(item)))}"
                props = {"id": pol_id, "soort": key}
                if isinstance(item, dict):
                    nb_name = item.get("Netbeheerder", None)  # of via pop_entries
                    for k, v in item.items():
                        props[clean_key(k)] = v
                else:
                    if key == "Instandhoudingsbeleid fabrikant":
                        props["Instandhoudingsbeleid fabrikant"] = item
                        nb_name = None
                    else:
                        # als het item geen dict is, sla over of sla op als property
                        continue
                session.execute_write(merge_node, "beleid", "id", props)
                session.execute_write(
                    merge_relation,
                    "dossier",
                    "aad_id",
                    aad_id,
                    "heeft_beleid",
                    "beleid",
                    "id",
                    pol_id,
                )

                if nb_name:
                    nb_id = f"netbeheerder_{nb_name}".lower()
                    session.execute_write(
                        merge_relation,
                        "netbeheerder",
                        "id",
                        nb_id,
                        "heeft_beleid",
                        "beleid",
                        "id",
                        pol_id,
                    )
        # ---------------------------------------------------------
        # Onderhoudsbeleid → onderhoud en inspectie nodes
        # ---------------------------------------------------------
        onderhoud_inspectie = data.get("Onderhoud & inspectie", {}).get(
            "Onderhoudstypes", []
        )
        for inspectie_punten_groep in onderhoud_inspectie:
            onderhoud_inspectie = inspectie_punten_groep.get(
                "Inspectiepunten per netbeheerder", []
            )
            for nb in onderhoud_inspectie:
                netbeheerder = nb.get("Netbeheerder", "Onbekend")
                inspectie_punten = nb.get("Inspectiepunten", [])
                for inspectie_punt in inspectie_punten:
                    print(f"Netbeheerder {netbeheerder}, beleid {inspectie_punt}")
                    if not inspectie_punt:
                        continue
                    inspectie_id = f"inspectie_{abs(hash(str(inspectie_punt)))}"
                    props = {"id": inspectie_id, "soort": "onderhoud_en_inspectie"}
                    if isinstance(inspectie_punt, dict):
                        for k, v in inspectie_punt.items():
                            if v:
                                props[clean_key(k)] = v
                    else:
                        continue
                    session.execute_write(merge_node, "beleid", "id", props)
                    session.execute_write(
                        merge_relation,
                        "dossier",
                        "aad_id",
                        aad_id,
                        "heeft_beleid",
                        "beleid",
                        "id",
                        inspectie_id,
                    )
                    session.execute_write(
                        merge_relation,
                        "netbeheerder",
                        "id",
                        netbeheerder,
                        "heeft_populatie",
                        "populatie",
                        "id",
                        inspectie_id,
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
    session.run("MATCH (n) DETACH DELETE n")

print("Database deleted.")

with driver.session() as session:
    for aad_id, component_id in COMPONENTS.items():
        json_folder = f"/home/ubuntu/ksandr_files/aads/{aad_id}/cat-1/fail-types/"
        # process faalvormen
        for root, dirs, files in os.walk(json_folder):
            for filename in files:
                if filename.endswith(".json"):
                    file_path = os.path.join(root, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        faalvorm_data = data.get("Beschrijving")
                    if faalvorm_data:
                        print(f"Processing {file_path}... for component {component_id}")
                        create_component_faalvorm(
                            session, aad_id, component_id, faalvorm_data, file_path
                        )
        # Process ageing asset dossier
        json_file = f"/home/ubuntu/ksandr_files/aads/{aad_id}/cat-1/main.json"
        print(f"Ingesting: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data = clean_html(data)
        ingest_dossier(data, aad_id, component_id)

driver.close()
