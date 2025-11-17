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
    """Build cypher query."""
    quantity = ["hoeveel", "hoeveelheid", "aantal", "totaal", "telling", "som"]
    columns = {
        "oorzaak": ["f.OorzaakGeneriek"],
        "lijst": ["f.NummerInt"],
        "nummer": ["f.NummerInt"],
        "component": ["c.naam"],
        "asset": ["c.naam"],
        "gevolg": ["f.MogelijkGevolg"],
        "faalindicator": ["f.Faalindicatoren"],
        "faaltempo": ["f.Faaltempo"],
        "effect": ["f.EffectOpSubsysteem"],
    }
    base_query = """
    MATCH (a:AAD)-[:HEEFT_COMPONENT]->(c:Component)-[:HEEFT_FAALVORM]->(f:Faalvorm)
    {where_clause}
    RETURN c.naam AS component, f.Naam AS faalvorm 
    """
    q_lower = question.lower()
    # 1. Detect quantity
    wants_quantity = any(term in q_lower for term in quantity)

    # 2. Detect requested columns
    selected_fields = []
    for key, fields in columns.items():
        if key in q_lower:
            selected_fields.extend(fields)

    # 3. Build RETURN clause
    return_parts = []

    if not wants_quantity:
        # Only include default fields when NOT counting
        return_parts.extend(["c.naam AS component", "f.Naam AS faalvorm"])

    # Add selected fields always
    for f in selected_fields:
        alias = f.split(".")[-1]
        return_parts.append(f"{f} AS {alias}")

    # Add count if required
    if wants_quantity:
        return_parts.append("COUNT(f) AS aantalFaalvorm")

    return_clause = "RETURN " + ", ".join(return_parts)

    # 5. Final assembly
    query = base_query.replace(
        "RETURN c.naam AS component, f.Naam AS faalvorm", return_clause
    )
    if wants_quantity:
        query += "\nORDER BY aantal DESC"
    return query.strip()
