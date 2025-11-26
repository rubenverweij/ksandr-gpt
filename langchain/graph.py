import re
import Levenshtein

STOPWORDS = {"de", "het", "een", "en", "van", "voor", "op"}


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # verwijder leestekens
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return set(tokens)


# Voorbeeld database
example_db = [
    {
        "questions": [
            "Geef het beheerteam van een dossier",
            "Wie zitten in het beheerteam van aad",
            "Haal het beheerteam op",
        ],
        "must_have": ["beheerteam"],  # verplichte woorden
        "cypher": """ 
        MATCH (d:dossier)-[:heeft_beheerteam_lid]->(p:persoon)
        {where_clause}
        RETURN 
            d.aad_id AS dossier_id,
            p.naam AS persoon_naam,
            p.id AS persoon_id,
            p.link AS profiel_link
        ORDER BY dossier_id, persoon_naam
        """,
    },
    {
        "questions": ["Geef een lijst van documenten voor dossier"],
        "must_have": ["documenten"],
        "cypher": """ 
        MATCH (d:dossier)-[:heeft_component]->(c:component)
        MATCH (d)-[:heeft_document]->(doc)
        RETURN doc
        """,
    },
    {
        "questions": ["Geef de populatiegegevens van component"],
        "must_have": ["populatiegegevens"],
        "cypher": """ 
        WITH $aad_ids AS dossier_ids, $netbeheerders AS nbs
        MATCH (d:dossier)
        WHERE size(dossier_ids) = 0 OR d.aad_id IN dossier_ids

        MATCH (nb:netbeheerder)
        WHERE size(nbs) = 0 OR ANY(t IN nbs WHERE toLower(nb.naam) CONTAINS toLower(t))

        MATCH (nb)-[:heeft_populatie]->(p:populatie)
        MATCH (d)-[:heeft_populatie]->(p)
        MATCH (d)-[:heeft_component]->(c:component)

        RETURN 
            nb.naam AS netbeheerder,
            d.aad_id AS dossier_id,
            c.component_id AS component_naam,
            p.populatie AS populatie,
            p.aantal_velden AS aantal_velden
        ORDER BY dossier_id, component_naam
        """,
    },
]


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
    return False


# FIXME deprecated
# def _postprocess_output_cypher(output_cypher: str) -> str:
#     # Remove any explanation. E.g.  MATCH...\n\n**Explanation:**\n\n -> MATCH...
#     # Remove cypher indicator. E.g.```cypher\nMATCH...```` --> MATCH...
#     # Note: Possible to have both:
#     #   E.g. ```cypher\nMATCH...````\n\n**Explanation:**\n\n --> MATCH...
#     partition_by = "**Explanation:**"
#     output_cypher, _, _ = output_cypher.partition(partition_by)
#     output_cypher = output_cypher.strip("`\n")
#     output_cypher = output_cypher.lstrip("cypher\n")
#     output_cypher = output_cypher.strip("`\n ")
#     return output_cypher


def build_cypher_query(question, clause=""):
    """Build cypher query with contains support."""
    quantity = ["hoeveel", "populatie", "hoeveelheid", "aantal", "totaal", "telling"]

    columns = {
        "oorzaak": ["f.OorzaakGeneriek"],
        "oorzaken": ["f.OorzaakGeneriek"],
        "lijst": ["f.NummerInt"],
        "opsomming": ["f.NummerInt"],
        "nummer": ["f.NummerInt"],
        "id": ["f.Prefix", "f.NummerInt"],
        "component": ["c.naam"],
        "incidenten": ["f.GemiddeldAantalIncidenten"],
        "meest voorkomende": ["f.GemiddeldAantalIncidenten"],
        "asset": ["c.naam"],
        "gevolg": ["f.MogelijkGevolg"],
        "faalindicator": ["f.Faalindicatoren"],
        "faaltempo": ["f.Faaltempo"],
        "effect": ["f.EffectOpSubsysteem"],
        "beschrijving": ["f.Beschrijving"],
        "omschrijving": ["f.Beschrijving"],
    }

    base_query = """
    MATCH (d:dossier)-[:HEEFT_COMPONENT]->(c:component)-[:HEEFT_FAALVORM]->(f:faalvorm)
    {where_clause}
    RETURN c.component_id AS component, f.Naam AS faalvorm 
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
    selected_fields = []
    for key, fields in columns.items():
        if key in q_lower:
            selected_fields.extend(fields)

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
        return_parts.extend(["c.component_id AS component", "f.Naam AS faalvorm"])
    for f in selected_fields:
        alias = f.split(".")[-1]
        return_parts.append(f"{f} AS {alias}")
    if wants_quantity:
        return_parts.append("COUNT(f) AS aantalFaalvorm")
    return_clause = "RETURN " + ", ".join(return_parts)

    # --------------------------------------------------------
    # 7. Assemble final cypher
    # --------------------------------------------------------
    query = base_query.format(where_clause=where_clause).replace(
        "RETURN c.component_id AS component, f.Naam AS faalvorm",
        return_clause,
    )
    if wants_quantity:
        query += "\nORDER BY aantalFaalvorm DESC"
    return query.strip()


def match_query(user_question):
    user_tokens = tokenize(user_question)
    best_match = None
    max_overlap = 0

    for entry in example_db:
        # Check verplichte woorden
        if not all(term in user_tokens for term in entry.get("must_have", [])):
            continue  # sla deze query over

        # Bereken overlap met alle voorbeeldvragen
        for q in entry["questions"]:
            db_tokens = tokenize(q)
            overlap = len(user_tokens & db_tokens)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = entry

    return best_match["cypher"], max_overlap


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
