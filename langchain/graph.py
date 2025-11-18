import re


def _postprocess_output_cypher(output_cypher: str) -> str:
    # Remove any explanation. E.g.  MATCH...\n\n**Explanation:**\n\n -> MATCH...
    # Remove cypher indicator. E.g.```cypher\nMATCH...```` --> MATCH...
    # Note: Possible to have both:
    #   E.g. ```cypher\nMATCH...````\n\n**Explanation:**\n\n --> MATCH...
    partition_by = "**Explanation:**"
    output_cypher, _, _ = output_cypher.partition(partition_by)
    output_cypher = output_cypher.strip("`\n")
    output_cypher = output_cypher.lstrip("cypher\n")
    output_cypher = output_cypher.strip("`\n ")
    return output_cypher


def build_cypher_query(question, clause=""):
    """Build cypher query with contains support."""
    quantity = [
        "hoeveel",
        "populatie",
        "hoeveelheid",
        "aantal",
        "totaal",
        "telling",
        "som",
    ]

    columns = {
        "oorzaak": ["f.OorzaakGeneriek"],
        "oorzaken": ["f.OorzaakGeneriek"],
        "lijst": ["f.NummerInt"],
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
    MATCH (a:AAD)-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALVORM]->(f:Faalvorm)
    {where_clause}
    RETURN c.naam AS component, f.Naam AS faalvorm 
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
    contains_patterns = ["bevat de term", "sprake is van", "m.b.t.", "bevat:"]
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
        return_parts.extend(["c.naam AS component", "f.Naam AS faalvorm"])
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
        "RETURN c.naam AS component, f.Naam AS faalvorm",
        return_clause,
    )
    if wants_quantity:
        query += "\nORDER BY aantalFaalvorm DESC"
    return query.strip()
