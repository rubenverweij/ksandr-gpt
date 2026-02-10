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

import re
import json
import Levenshtein
from config import (
    COLUMN_MAPPING_FAALVORM,
    NETBEHEERDERS_LOWER,
    STOPWORDS,
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


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # verwijder leestekens
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return set(tokens)


def match_query_by_tags_deprecated(question: str, query: dict) -> bool:
    """
    Returns True if any tag from query["tags"] is found in the question text,
    allowing for up to 1 character typo (Levenshtein distance <= 1).
    Tags must be separated by ';'.
    """
    if "tags" not in query or not query["tags"]:
        return False

    # Split tags on ';'
    tags = [tag.strip().lower() for tag in query["tags"].split(";")]

    # Normalize question
    q_words = question.lower().split()

    # Check each tag against each word in the question
    for tag in tags:
        for word in q_words:
            if Levenshtein.distance(tag, word) <= 1:
                return True
        if tag in question:
            return True
    return False


def match_tag_combi(question, combi, max_afstand=1):
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
    # base_query = """
    # MATCH (d:dossier)-[:HEEFT_COMPONENT]->(c:component)-[:HEEFT_FAALVORM]->(f:faalvorm)
    # {where_clause}
    # {return_clause}
    # """

    q_lower = request.lower()
    # --------------------------------------------------------
    # 1. Start WHERE clauses with user-supplied base clause
    # --------------------------------------------------------
    # where_clauses = []
    # if clause:
    #     where_clauses.append(clause)  # e.g. "WHERE a.aad_id IN $aad_ids"

    # --------------------------------------------------------
    # 2. Detect quantity (count?)
    # --------------------------------------------------------
    wants_quantity = any(term in q_lower for term in quantity)

    # --------------------------------------------------------
    # 3. Detect requested columns
    # --------------------------------------------------------
    selected_fields = set()
    for key, fields in COLUMN_MAPPING_FAALVORM.items():
        if key in q_lower:
            selected_fields.update(fields)
    selected_fields = list(selected_fields)

    # --------------------------------------------------------
    # 4. Detect “contains” / “bevat” patterns
    # --------------------------------------------------------
    # contains_patterns = ["bevat de term", "m.b.t.", "bevat:"]
    # contains_term = None

    # for pat in contains_patterns:
    #     if pat in q_lower:
    #         match = re.search(pat + r"\s+(.*)", q_lower)
    #         if match:
    #             contains_term = match.group(1).strip()
    #             break

    # If contains detected, build WHERE contains clause
    # if contains_term:
    #     target_columns = []
    #     # Prefer explicit description-related keywords
    #     for key in ["beschrijving", "omschrijving", "oorzaak"]:
    #         if key in q_lower:
    #             target_columns = COLUMN_MAPPING_FAALVORM[key]
    #     # fallback → search description
    #     if not target_columns:
    #         target_columns = ["f.Beschrijving"]
    #     for col in target_columns:
    #         where_clauses.append(f'toLower({col}) CONTAINS toLower("{contains_term}")')

    # --------------------------------------------------------
    # 5. Build WHERE clause output
    # --------------------------------------------------------
    # if where_clauses:
    #     # If the first clause already begins with WHERE, don’t repeat it
    #     if where_clauses[0].strip().upper().startswith("WHERE"):
    #         where_clause = where_clauses[0]
    #         extra_filters = where_clauses[1:]
    #         if extra_filters:
    #             where_clause += " AND " + " AND ".join(extra_filters)
    #     else:
    #         where_clause = "WHERE " + " AND ".join(where_clauses)
    # else:
    #     where_clause = ""

    # --------------------------------------------------------
    # 6. Build RETURN clause
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 7. Assemble final cypher
    # --------------------------------------------------------
    query = base_query.format(return_clause=return_clause)
    if wants_quantity:
        query += "\nORDER BY aantalFaalvorm DESC"
    else:
        query += "\nORDER BY index"
    return query.strip()


def check_for_nbs(question):
    """Check if netbeheerders are present."""
    lijst_nb = NETBEHEERDERS_LOWER
    question_lower = question.lower()
    matched_variants = []
    for variant in lijst_nb:
        if variant.lower() in question_lower:
            print(variant)
            matched_variants.append(variant)
    return matched_variants
