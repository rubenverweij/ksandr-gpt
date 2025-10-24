# Dictionary met Cypher-templates (case-insensitive)
cypher_templates = {
    "alle_faalvormen_per_component": {
        "query": """
            MATCH (c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
            WHERE toLower(c.naam) = toLower($component)
            RETURN f.Nummer AS Nummer,
                   f.Naam AS Naam,
                   f.Beschrijving AS Beschrijving,
                   f.MogelijkGevolg AS MogelijkGevolg,
                   f.Faaltempo AS Faaltempo
            ORDER BY f.Nummer
        """,
        "parameters": ["component"],
    },
    "meest_voorkomende_faalvormen": {
        "query": """
            MATCH (f:Faaltype)
            RETURN f.Nummer AS Nummer,
                   f.Naam AS Naam,
                   f.GemiddeldAantalIncidenten AS GemiddeldAantalIncidenten
            ORDER BY CASE f.GemiddeldAantalIncidenten
                        WHEN 'Zeer regelmatig (>5)' THEN 5
                        WHEN 'Regelmatig (3-5)' THEN 4
                        WHEN 'Incidenteel (1-2)' THEN 3
                        WHEN 'Onbekend' THEN 2
                        ELSE 1
                     END DESC
            LIMIT 5
        """,
        "parameters": [],
    },
    "faalvormen_met_indicator": {
        "query": """
            MATCH (f:Faaltype)
            WHERE toLower(f.Faalindicatoren) CONTAINS toLower($indicator)
            RETURN f.Nummer AS Nummer,
                   f.Naam AS Naam,
                   f.Faalindicatoren AS Indicatoren
        """,
        "parameters": ["indicator"],
    },
    "gemeenschappelijke_faalvormen_tussen_2_componenten": {
        "query": """
        MATCH (c1:Component)-[:HEEFT_FAALTYPE]->(f1:Faaltype)
        MATCH (c2:Component)-[:HEEFT_FAALTYPE]->(f2:Faaltype)
        WHERE toLower(c1.naam) = toLower($component1)
        AND toLower(c2.naam) = toLower($component2)
        AND toLower(f1.Naam) = toLower(f2.Naam)
        RETURN f1.Nummer AS Nummer,
            f1.Naam AS Naam,
            f1.Beschrijving AS Beschrijving
        ORDER BY f1.Nummer
        """,
        "parameters": ["component1", "component2"],
    },
    "faalvormen_voor_component_voor_oorzaak": {
        "query": """
        MATCH (c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
        WHERE toLower(c.naam) = toLower($component)
          AND toLower(f.OorzaakGeneriek) = toLower($generieke_oorzaak)
        RETURN f.Nummer AS Nummer,
               f.Naam AS Naam,
               f.Beschrijving AS Beschrijving,
               f.OorzaakGeneriek AS GeneriekeOorzaak
        ORDER BY f.Nummer
        """,
        "parameters": ["component", "generieke_oorzaak"],
    },
    "meest_voorkomende_faalvormen_per_component": {
        "query": """
        UNWIND $componenten AS compNaam
        MATCH (c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
        WHERE toLower(c.naam) = toLower(compNaam)
        RETURN c.naam AS Component,
               f.Nummer AS Nummer,
               f.Naam AS Naam,
               f.GemiddeldAantalIncidenten AS Frequentie
        ORDER BY c.naam,
                 CASE f.GemiddeldAantalIncidenten
                   WHEN 'Zeer regelmatig (>5)' THEN 1
                   WHEN 'Regelmatig (2-5)' THEN 2
                   WHEN 'Incidenteel' THEN 3
                   ELSE 4
                 END ASC
        """,
        "parameters": ["componenten"],
    },
    "faalvormen_filter": {
        "query": """
        UNWIND $componenten AS compNaam
        MATCH (c:Component)-[:HEEFT_FAALTYPE]->(f:Faaltype)
        WHERE toLower(c.naam) = toLower(compNaam)
          AND ($generieke_oorzaak IS NULL OR toLower(f.OorzaakGeneriek) = toLower($generieke_oorzaak))
        RETURN c.naam AS Component,
               f.Nummer AS Nummer,
               f.Naam AS Naam,
               f.OorzaakGeneriek AS GeneriekeOorzaak,
               f.GemiddeldAantalIncidenten AS Frequentie
        ORDER BY c.naam,
                 CASE f.GemiddeldAantalIncidenten
                   WHEN 'Zeer regelmatig (>5)' THEN 1
                   WHEN 'Regelmatig (2-5)' THEN 2
                   WHEN 'Incidenteel' THEN 3
                   ELSE 4
                 END ASC
        """,
        "parameters": ["componenten", "generieke_oorzaak"],
    },
}
