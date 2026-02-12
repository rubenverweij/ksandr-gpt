"""
This module provides functions for ingesting Ageing Asset Dossiers (AADs) into a Neo4j graph database.
It includes utilities to clean and normalize HTML data, extract relevant information such as component numbers,
and create corresponding nodes and relationships in the database. The ingestion pipeline is tailored for
the Ksandr platform and processes component, dossier, and failure mode (faalvorm) records based on extracted information.
"""

from neo4j import GraphDatabase
import json
import os
from bs4 import BeautifulSoup
import re
from config import COMPONENTS, SECRETS

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=(SECRETS.get("username"), SECRETS.get("password")),
)


def clean_html(value):
    """
    Cleans a given input value by removing HTML tags, unnecessary whitespace, and specific unwanted fields.

    This function is designed to recursively process and normalize text, dictionaries, and lists,
    to return sanitized data suitable for storage or further processing. Specifically:
      - For string inputs, all HTML tags are stripped, and excessive whitespace is collapsed.
      - For dictionary inputs, the 'tags' key is removed (if present), and all key/values are recursively processed.
      - For list inputs, all items are recursively cleaned, and empty results are pruned.
      - None, empty strings, empty lists, and empty dictionaries are returned as None.
      - For other datatypes, the value is returned unchanged.

    Args:
        value (Any): The input (string, dict, list, etc.) to be cleaned.

    Returns:
        Any: The cleaned version of the input, with HTML and noise removed.
    """
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


def extract_nummer_info(nummer_str: str) -> tuple[str, None | int]:
    """
    Extracts the full string prefix and the numerical part from a component number string, such as 'MAG-1'.

    Args:
        nummer_str (str): The input string containing the component number.

    Returns:
        tuple: A tuple (prefix, nummer_int) where:
            - prefix (str): The string prefix of the number (e.g., 'MAG').
            - nummer_int (int or None): The numeric portion as an integer if present, otherwise None.

    Example:
        >>> extract_nummer_info("MAG-13")
        ('MAG', 13)
        >>> extract_nummer_info("COMPONENT")
        ('COMPONENT', None)
    """
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
    """
    Inserts or merges a 'faalvorm' (failure mode) node associated with a specific component and dossier into the Neo4j database,
    and establishes relationships to related nodes, including permissions.

    Args:
        session: Neo4j database session for executing queries.
        aad_id: Identifier of the dossier associated with this component.
        component_id: Identifier of the component to which this faalvorm belongs.
        faalvorm_data: Dictionary containing the properties and attributes of the faalvorm (failure mode).
        file_path: Path to the associated file or document (used for setting Bestandspad).
        permission: String indicating the category of permission granted to this faalvorm.

    Returns:
        None. This function executes a database transaction, but does not return a value.

    Side Effects:
        - Creates or merges nodes in the Neo4j database.
        - Sets node properties based on input data.
        - Creates relationships between dossier, component, faalvorm, and permission nodes.

    Example:
        create_component_faalvorm(
            session,
            aad_id="AAD-123",
            component_id="C-108",
            faalvorm_data={"Naam": "Scheurvorming", ...},
            file_path="/files/fm1.pdf",
            permission="read"
        )
    """
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
                "Gemiddeld aantal incidenten", faalvorm_data.get("Aantal incidenten")
            ),
            "bestandspad": file_path,
        },
    )


def clean_key(key: str) -> str:
    """
    Cleans and normalizes a dictionary key to a standardized format.

    This function takes a string `key`, converts it to lowercase, replaces spaces
    and forward slashes ('/') with underscores ('_'), making it suitable for use
    as a consistent property or key in code and databases.

    Args:
        key (str): The key string to be cleaned.

    Returns:
        str: The cleaned, normalized key.
    """
    return key.lower().replace(" ", "_").replace("/", "_")


def optional(obj: dict, key: str, default=None):
    """
    Safely retrieve a value from an object if the key exists.

    This function checks if the given `obj` is a dictionary, and if so,
    returns the value for `key` if present, or a `default` value otherwise.
    If `obj` is not a dictionary, it returns the `default`.

    Args:
        obj (dict): The dictionary to retrieve the value from.
        key: The key to look for.
        default: The value to return if the key is not found. Defaults to None.

    Returns:
        The value corresponding to `key` in the dictionary, or the default value.
    """
    return obj.get(key, default) if isinstance(obj, dict) else default


def merge_node(tx: GraphDatabase, label: str, key: str, props: dict):
    """
    Merges a node into the database if it does not already exist, and updates its properties.

    This function constructs and executes a Cypher MERGE query using the provided transaction, label,
    unique key, and property dictionary. If a node with the given label and key exists, it is matched; if not,
    a new node is created. The node's properties are then updated or set according to the provided dict.

    Args:
        tx: The Neo4j transaction object.
        label (str): The label to assign to the node.
        key (str): The property key to uniquely identify the node.
        props (dict): A dictionary of properties to set on the node.

    Side Effects:
        Modifies the database by inserting or updating the specified node.
    """
    query = f"""
    MERGE (n:{label} {{{key}: $keyval}})
    SET n += $props
    """
    tx.run(query, keyval=props[key], props=props)


def merge_relation(
    tx: GraphDatabase,
    a_label: str,
    a_key: str,
    a_val: str,
    rel: str,
    b_label: str,
    b_key: str,
    b_val: str,
):
    """
    Merges a relationship between two nodes if it does not already exist in the database.

    This function constructs and executes a Cypher MERGE query using the provided transaction and node details.
    It matches two nodes based on their label and unique key/value pairs, then creates a directed relationship
    of the specified type from node 'a' to node 'b' if it does not already exist.

    Args:
        tx: The Neo4j transaction object.
        a_label (str): The label of the source node.
        a_key (str): The property key to identify the source node.
        a_val: The value of the property key for the source node.
        rel (str): The type/name of the relationship to merge.
        b_label (str): The label of the target node.
        b_key (str): The property key to identify the target node.
        b_val: The value of the property key for the target node.

    Side Effects:
        Modifies the database by creating or ensuring the existence of the specified relationship between nodes.
    """
    query = f"""
    MATCH (a:{a_label} {{{a_key}: $a_val}})
    MATCH (b:{b_label} {{{b_key}: $b_val}})
    MERGE (a)-[:{rel}]->(b)
    """
    tx.run(query, a_val=a_val, b_val=b_val)


def create_permission_constraint(driver: GraphDatabase):
    """
    Creates a uniqueness constraint on the 'category' property of 'permission' nodes in the Neo4j database.

    This function ensures that each 'permission' node in the database has a unique 'category' property.
    If the uniqueness constraint already exists, the function leaves it unchanged.

    Args:
        driver (GraphDatabase): The Neo4j database driver.

    Side Effects:
        Modifies the database schema by creating a uniqueness constraint if it does not exist.
    """
    with driver.session() as session:
        session.run(
            """
            CREATE CONSTRAINT permission_category IF NOT EXISTS
            FOR (p:permission)
            REQUIRE p.category IS UNIQUE;
        """
        )


def create_permission_nodes(driver: GraphDatabase, permission: str):
    """
    Creates a 'permission' node in the Neo4j database with the specified category.

    This function writes a 'permission' node, identified by its 'category' property, to the database.
    If a node with the given category already exists, it leaves the existing node unchanged (merge semantics).

    Args:
        driver (GraphDatabase): The Neo4j database driver.
        permission (str): The unique category of the permission to create.

    Side Effects:
        Modifies the database by creating or merging a 'permission' node.
    """
    with driver.session() as session:
        permission_props = {"category": permission}
        session.execute_write(merge_node, "permission", "category", permission_props)


def ingest_dossier(
    driver: GraphDatabase, data: dict, aad_id: str, component_id: str, permission: str
):
    """
    Ingests dossier data into the Neo4j graph database by creating and connecting relevant nodes and relationships.

    This function takes raw dossier data, parses essential attributes, and writes them as nodes and relationships
    into the database. It ensures that 'dossier', 'persoon', and 'permission' nodes are created as needed, and
    that the defined relationships such as permissions and management team memberships are properly established.

    Args:
        driver (GraphDatabase): The Neo4j database driver.
        data (dict): The dossier data loaded from an external source (typically parsed JSON).
        aad_id (str): The unique identifier for the dossier (AAD).
        component_id (str): The unique identifier for the component related to the dossier.
        permission (str): The category of permission to assign.

    Side Effects:
        Modifies the Neo4j database by creating and merging nodes and relationships representing the dossier and
        its associated permissions and participants.
    """
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


if __name__ == "__main__":
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
