"""
This module defines utilities, constants, and Cypher query templates
for interacting with and querying the Ksandr knowledge graph (Neo4j)
within the context of ageing asset dossiers and their failure modes.

Core features:
- Construction of Cypher queries for dossiers, components, and failure modes,
  supporting permission-based filtering and component metadata access.
- Tokenization and fuzzy matching logic for NL text queries, including
  Levenshtein-based typo resilience and stopword filtering.
- Logging setup for module-internal diagnostics.

Imports configuration settings and constants for field mapping,
stopword exclusion, and lemmatized query analysis, as well as
third-party utilities for text processing.

Used by ingestion, permission, and LLM modules to enable
secure, semantic, and permission-aware retrieval from the Ksandr graph.
"""

import json
import Levenshtein
from ksandr.settings.config import (
    COLUMN_MAPPING_FAALVORM,
    NETBEHEERDERS_LOWER,
    QUANTITY_TERMS,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_QUERY_SPEC = """
WITH $aad_ids AS dossier_ids, $permissions AS permissions
UNWIND keys(permissions) AS category
UNWIND permissions[category] AS allowed_dossier_id
MATCH (d:dossier {{aad_id: allowed_dossier_id}})-[:HEEFT_COMPONENT]->(c:component)-[:HEEFT_FAALVORM]->(f:faalvorm)
MATCH (d)-[:HAS_PERMISSION]->(:permission {{category: category}})
MATCH (f)-[:HAS_PERMISSION]->(:permission {{category: category}})
WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids
{return_clause}
"""


def match_tag_combi(question: str, combi: list[str], max_afstand=1) -> bool:
    """
    Checks whether a question contains an ordered sequence of target words (a tag combination),
    allowing for minor typos (Levenshtein distance <= max_afstand) with order sensitivity.

    For each word in the target combination (`combi`), scans through the question words (starting from the last
    matched position), and tries to match it with typo tolerance. All target words must be matched in order
    (but not necessarily consecutively). If any is not found in order, returns False.

    Args:
        question (str): The user's question as a string.
        combi (list[str]): A list of target words representing a tag combination.
        max_afstand (int, optional): Maximum allowed Levenshtein distance for typo tolerance (default 1).

    Returns:
        bool: True if all target words are found in order within the question (with typo tolerance); False otherwise.
    """
    woorden = question.lower().split()
    idx = 0
    for target in combi:
        gevonden = False
        for i in range(idx, len(woorden)):
            if Levenshtein.distance(woorden[i], target) <= max_afstand:
                idx = i + 1  # volgorde bewaken
                gevonden = True
                break
        if not gevonden:
            return False
    return True


def match_query_by_tags(question: str, query: dict) -> bool:
    """
    Determines if a user's question matches a query based on tag combinations.

    This function checks whether any of the tag combinations (from the 'tags_list' field of the query dictionary)
    approximately matches the question text, allowing for minor spelling variations (typo tolerance with Levenshtein distance).
    Each tag combination is a sequence of keywords; the matching is order-sensitive within each combination and considers
    typos within a defined threshold.

    Args:
        question (str): The input question text from the user.
        query (dict): The query dictionary containing a 'tags_list' field (JSON-encoded list of tag combinations).

    Returns:
        bool: True if at least one tag combination is matched in order within the question text; False otherwise.
    """
    if "tags_list" not in query or not query["tags_list"]:
        return True

    try:
        tag_combinaties = json.loads(query["tags_list"])
    except Exception:
        return False

    for combi in tag_combinaties:
        if match_tag_combi(question, combi):
            logging.info(f"Matched on taglist: {combi}")
            return True

    return False


def build_cypher_query(request: str) -> str:
    """Build cypher query with contains support.

    Args:
        request (AskRequest): the original request including prompt and permission
        clause (str): string where clause

    Returns:
        str: cypher query
    """
    quantity = QUANTITY_TERMS
    base_query = BASE_QUERY_SPEC
    q_lower = request.lower()
    wants_quantity = any(term in q_lower for term in quantity)

    # Find requested columns
    selected_fields = set()
    for key, fields in COLUMN_MAPPING_FAALVORM.items():
        if key in q_lower:
            selected_fields.update(fields)
    selected_fields = list(selected_fields)
    return_parts = []
    if not wants_quantity:
        return_parts.extend(
            [
                "c.component_id AS component",
                "f.Naam AS faalvorm",
                "f.faalvorm_id as nummer_faalvorm",
                "f.NummerInt as index",
            ]
        )
    for f in selected_fields:
        column = f.split(":")
        return_parts.append(f"{column[0]} AS {column[1]}")
    if wants_quantity:
        return_parts.append("COUNT(f) AS aantalFaalvorm")
    return_clause = "RETURN " + ", ".join(return_parts)
    query = base_query.format(return_clause=return_clause)
    if wants_quantity:
        query += "\nORDER BY aantalFaalvorm DESC"
    else:
        query += "\nORDER BY index"
    return query.strip()


def check_for_nbs(question: str) -> list[str]:
    """
    Checks if any known netbeheerders (network operators) are mentioned in the question.

    Args:
        question (str): The input string or question to check for netbeheerder names.

    Returns:
        list: A list of matched netbeheerder variants found within the question (case-insensitive).
    """
    lijst_nb = NETBEHEERDERS_LOWER
    question_lower = question.lower()
    matched_variants = []
    for variant in lijst_nb:
        if variant.lower() in question_lower:
            print(variant)
            matched_variants.append(variant)
    return matched_variants
