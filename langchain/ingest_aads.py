from neo4j import GraphDatabase
import json
import os
from bs4 import BeautifulSoup
import re
from config import COMPONENTS

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


def create_component_faalvorm(
    session, aad_id, component_id, faalvorm_data, file_path, permission
):
    nummer_str = faalvorm_data.get("Nummer")
    prefix, nummer_int = extract_nummer_info(nummer_str)
    # NOTE: f.columns in the json extract should exist
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
    
    MERGE (p:permission {category: $permission})
    MERGE (d)-[:HEEFT_COMPONENT]->(c)
    MERGE (c)-[:HEEFT_FAALVORM]->(f)
    MERGE (f)-[:HAS_PERMISSION]->(p)
    """
    session.run(
        cypher,
        {
            "aad_id": aad_id,
            "component_id": component_id,
            "faalvorm_id": nummer_str,
            "naam": faalvorm_data.get("Naam"),
            "nummer_int": nummer_int,
            "permission": permission,
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


def create_permission_constraint(driver: GraphDatabase):
    """Create permission constraint."""
    with driver.session() as session:
        session.run(
            """
            CREATE CONSTRAINT permission_category IF NOT EXISTS
            FOR (p:permission)
            REQUIRE p.category IS UNIQUE;
        """
        )


def create_permission_nodes(driver: GraphDatabase, permission: str):
    """Create permission nodes."""
    with driver.session() as session:
        permission_props = {"category": permission}
        session.execute_write(merge_node, "permission", "category", permission_props)


def ingest_dossier(driver: GraphDatabase, data, aad_id, component_id, permission):
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
        session.execute_write(
            merge_relation,
            "dossier",
            "aad_id",
            aad_id,
            "HAS_PERMISSION",
            "permission",
            "category",
            permission,
        )

        # Beheerteam → toevoegen persoon nodes
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
            session.execute_write(
                merge_relation,
                "persoon",
                "id",
                person_id,
                "HAS_PERMISSION",
                "permission",
                "category",
                permission,
            )

        # Component → component node
        comp = data.get("Dossier", {}).get("Component", {})
        comp_props = {"component_id": component_id, "permission": permission}
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
        session.execute_write(
            merge_relation,
            "component",
            "component_id",
            component_id,
            "HAS_PERMISSION",
            "permission",
            "category",
            permission,
        )

        # Media → document nodes
        documents = data.get("Bestanden", {}).get("Bestanden", [])
        for document in documents:
            doc_id = document.get("documentId")
            props = {
                "id": doc_id,
                "naam": document.get("name"),
                "locatie": document.get("directory"),
            }
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
            session.execute_write(
                merge_relation,
                "document",
                "id",
                doc_id,
                "HAS_PERMISSION",
                "permission",
                "category",
                permission,
            )

        media_groups = data.get("Dossier", {}).get("Media", [])
        for group in media_groups:
            for item in group:
                doc_id = item.get("aadDocumentId")
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
                session.execute_write(
                    merge_relation,
                    "document",
                    "id",
                    doc_id,
                    "HAS_PERMISSION",
                    "permission",
                    "category",
                    permission,
                )

        # Overige bestanden → document nodes
        overige = data.get("Dossier", {}).get("Overige bestanden", [])
        for group in overige:
            for item in group:
                doc_id = item.get("documentId")
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
                session.execute_write(
                    merge_relation,
                    "document",
                    "id",
                    doc_id,
                    "HAS_PERMISSION",
                    "permission",
                    "category",
                    permission,
                )

        # Populatiegegevens → populatie + netbeheerder nodes
        pop_entries = data.get("Populatiegegevens", {}).get(
            "Populatie per netbeheerder", []
        )
        for p in pop_entries:
            nb_name = p.get("Netbeheerder") or "onbekend"
            nb_id = hash(nb_name)
            # netbeheerder node
            nb_props = {"id": nb_id, "naam": nb_name}
            session.execute_write(merge_node, "netbeheerder", "id", nb_props)
            # populatie node
            pop_id = f"pop_{nb_name}_{aad_id}_{p.get('Populatie')}".lower()
            pop_props = {"id": pop_id, "permission": permission}
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
            session.execute_write(
                merge_relation,
                "populatie",
                "id",
                pop_id,
                "HAS_PERMISSION",
                "permission",
                "category",
                permission,
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
                session.execute_write(
                    merge_relation,
                    "beleid",
                    "id",
                    pol_id,
                    "HAS_PERMISSION",
                    "permission",
                    "category",
                    permission,
                )

                if nb_name:
                    session.execute_write(
                        merge_relation,
                        "netbeheerder",
                        "id",
                        hash(nb_name),
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
                    nb_id = hash(netbeheerder)
                    session.execute_write(
                        merge_relation,
                        "netbeheerder",
                        "id",
                        nb_id,
                        "heeft_beleid",
                        "beleid",
                        "id",
                        inspectie_id,
                    )
                    session.execute_write(
                        merge_relation,
                        "beleid",
                        "id",
                        inspectie_id,
                        "HAS_PERMISSION",
                        "permission",
                        "category",
                        permission,
                    )


with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

print("Database deleted.")

with driver.session() as session:
    permissions = ["cat-1", "cat-2"]
    for permission in permissions:
        create_permission_nodes(driver, permission)
    create_permission_constraint(driver)
    for aad_id, component_id in COMPONENTS.items():
        for permission in permissions:
            json_folder = (
                f"/home/ubuntu/ksandr_files/aads/{aad_id}/{permission}/fail-types/"
            )
            # process faalvormen
            for root, dirs, files in os.walk(json_folder):
                for filename in files:
                    if filename.endswith(".json"):
                        file_path = os.path.join(root, filename)
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            faalvorm_data = data.get("Beschrijving")
                        if faalvorm_data:
                            print(
                                f"Processing {file_path}... for component {component_id}"
                            )
                            create_component_faalvorm(
                                session,
                                aad_id,
                                component_id,
                                faalvorm_data,
                                file_path,
                                permission,
                            )
            # Process ageing asset dossier
            json_file = (
                f"/home/ubuntu/ksandr_files/aads/{aad_id}/{permission}/main.json"
            )
            print(f"Ingesting: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data = clean_html(data)
            ingest_dossier(driver, data, aad_id, component_id, permission)

driver.close()
