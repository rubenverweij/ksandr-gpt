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


def build_cypher_query(question):
    """Build cypher query with contains support."""
    quantity = ["hoeveel", "hoeveelheid", "aantal", "totaal", "telling", "som"]

    columns = {
        "oorzaak": ["f.OorzaakGeneriek"],
        "lijst": ["f.NummerInt"],
        "nummer": ["f.NummerInt"],
        "id": ["f.Prefix", "f.NummerInt"],
        "component": ["c.naam"],
        "asset": ["c.naam"],
        "gevolg": ["f.MogelijkGevolg"],
        "faalindicator": ["f.Faalindicatoren"],
        "faaltempo": ["f.Faaltempo"],
        "effect": ["f.EffectOpSubsysteem"],
        "beschrijving": ["f.beschrijving"],  # <-- nieuw
        "omschrijving": ["f.beschrijving"],
    }

    base_query = """
    MATCH (a:AAD)-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALVORM]->(f:Faalvorm)
    {where_clause}
    RETURN c.naam AS component, f.Naam AS faalvorm 
    """

    q_lower = question.lower()
    where_clauses = ["{where_clause}"]

    # --- 1. Detect quantity (COUNT required?)
    wants_quantity = any(term in q_lower for term in quantity)

    # --- 2. Detect request columns
    selected_fields = []
    for key, fields in columns.items():
        if key in q_lower:
            selected_fields.extend(fields)

    # --- 3. Detect "contains" / "bevat" patterns
    contains_patterns = ["bevat", "sprake is van", "m.b.t."]
    contains_term = None

    for pat in contains_patterns:
        if pat in q_lower:
            # everything after the pattern
            match = re.search(pat + r"\s+(.*)", q_lower)
            if match:
                contains_term = match.group(1).strip()
                break

    # If contains detected, build WHERE clause
    if contains_term:
        target_columns = []

        # Which column should we search?
        # Prefer explicit column names (beschrijving, omschrijving, tekst)
        for key in ["beschrijving", "omschrijving", "oorzaak"]:
            if key in q_lower:
                target_columns = columns[key]

        # fallback: if no explicit column mentioned → search description
        if not target_columns:
            target_columns = ["f.beschrijving"]

        for col in target_columns:
            where_clauses.append(f'toLower({col}) CONTAINS toLower("{contains_term}")')

    # Assemble WHERE clause (AND conditions)
    if where_clauses:
        where_clause = " AND ".join(where_clauses)

    # --- 4. Build RETURN clause
    return_parts = []

    if not wants_quantity:
        return_parts.extend(["c.naam AS component", "f.Naam AS faalvorm"])

    for f in selected_fields:
        alias = f.split(".")[-1]
        return_parts.append(f"{f} AS {alias}")

    if wants_quantity:
        return_parts.append("COUNT(f) AS aantalFaalvorm")

    return_clause = "RETURN " + ", ".join(return_parts)

    # --- 5. Assemble final query
    query = base_query.format(where_clause=where_clause).replace(
        "RETURN c.naam AS component, f.Naam AS faalvorm", return_clause
    )

    if wants_quantity:
        query += "\nORDER BY aantalFaalvorm DESC"

    return query.strip()
