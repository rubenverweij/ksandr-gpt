import re
import Levenshtein

STOPWORDS = {"de", "het", "een", "en", "van", "voor", "op"}


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # verwijder leestekens
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return set(tokens)


def match_query_by_tags(question: str, query: dict) -> bool:
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


def build_cypher_query(question, clause=""):
    """Build cypher query with contains support."""
    quantity = ["hoeveel", "populatie", "hoeveelheid", "aantal", "totaal", "telling"]

    columns = {
        "oorzaak": ["f.OorzaakGeneriek:oorzaak_generiek"],
        "oorzaken": ["f.OorzaakGeneriek:oorzaak_generiek"],
        # "lijst": ["f.faalvorm_id:nummer_faalvorm"],
        # "opsomming": ["f.faalvorm_id:nummer_faalvorm"],
        # "nummer": ["f.faalvorm_id:nummer_faalvorm"],
        # "id": ["f.faalvorm_id:nummer_faalvorm"],
        "component": ["c.component_id:naam_component"],
        "repareer": ["c.niet_repareerbaar:niet_repareerbaar"],
        "incidenten": ["f.GemiddeldAantalIncidenten:aantal_incidenten"],
        "meest voorkomende": ["f.GemiddeldAantalIncidenten:aantal_incidenten"],
        "asset": ["c.component_id:naam_component"],
        "gevolg": ["f.MogelijkGevolg:mogelijk_gevolg"],
        "faalindicator": ["f.Faalindicatoren:faalindicator"],
        "faaltempo": ["f.Faaltempo:faaltempo"],
        "effect": ["f.EffectOpSubsysteem:effect_op_systeem"],
        "beschrijving": ["f.Beschrijving:beschrijving"],
        "omschrijving": ["f.Beschrijving:beschrijving"],
    }

    base_query = """
    MATCH (d:dossier)-[:HEEFT_COMPONENT]->(c:component)-[:HEEFT_FAALVORM]->(f:faalvorm)
    {where_clause}
    {return_clause}
    """
    q_lower = question.lower()
    # --------------------------------------------------------
    # 1. Start WHERE clauses with user-supplied base clause
    # --------------------------------------------------------
    where_clauses = []
    if clause:
        where_clauses.append(clause)  # e.g. "WHERE a.aad_id IN $aad_ids"

    # --------------------------------------------------------
    # 2. Detect quantity (count?)
    # --------------------------------------------------------
    wants_quantity = any(term in q_lower for term in quantity)

    # --------------------------------------------------------
    # 3. Detect requested columns
    # --------------------------------------------------------
    selected_fields = set()
    for key, fields in columns.items():
        if key in q_lower:
            selected_fields.extend(fields)
    selected_fields = list(selected_fields)

    # --------------------------------------------------------
    # 4. Detect “contains” / “bevat” patterns
    # --------------------------------------------------------
    contains_patterns = ["bevat de term", "m.b.t.", "bevat:"]
    contains_term = None

    for pat in contains_patterns:
        if pat in q_lower:
            match = re.search(pat + r"\s+(.*)", q_lower)
            if match:
                contains_term = match.group(1).strip()
                break

    # If contains detected, build WHERE contains clause
    if contains_term:
        target_columns = []
        # Prefer explicit description-related keywords
        for key in ["beschrijving", "omschrijving", "oorzaak"]:
            if key in q_lower:
                target_columns = columns[key]
        # fallback → search description
        if not target_columns:
            target_columns = ["f.Beschrijving"]
        for col in target_columns:
            where_clauses.append(f'toLower({col}) CONTAINS toLower("{contains_term}")')

    # --------------------------------------------------------
    # 5. Build WHERE clause output
    # --------------------------------------------------------
    if where_clauses:
        # If the first clause already begins with WHERE, don’t repeat it
        if where_clauses[0].strip().upper().startswith("WHERE"):
            where_clause = where_clauses[0]
            extra_filters = where_clauses[1:]
            if extra_filters:
                where_clause += " AND " + " AND ".join(extra_filters)
        else:
            where_clause = "WHERE " + " AND ".join(where_clauses)
    else:
        where_clause = ""

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
    query = base_query.format(where_clause=where_clause, return_clause=return_clause)
    if wants_quantity:
        query += "\nORDER BY aantalFaalvorm DESC"
    return query.strip()


def check_for_nbs(question):
    """Check if netbeheerders are present."""
    lijst_nb = [
        "coteq",
        "enduris",
        "enexis",
        "liander",
        "stedin",
        "westland",
        "rendo",
        "tennet",
    ]
    question_lower = question.lower()
    matched_variants = []
    for variant in lijst_nb:
        if variant.lower() in question_lower:
            print(variant)
            matched_variants.append(variant)
    return matched_variants
