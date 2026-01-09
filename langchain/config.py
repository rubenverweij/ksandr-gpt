LIJST_SPECIFIEKE_COMPONENTEN = [
    "db10",
    "dr12",
    "bcc",
    "md",
    "mf",
    "magnefix",
    "merlin",
    "safeplus",
    "conel",
    "svs",
    "coq",
    "8djh",
    "vd4",
    "ela12",
    "l-sep",
    "rm6",
    "nvc00",
    "nxplusc",
    "laagspanning",
    "mms",
    "fmx",
    "xiria",
    "capitole",
]

NETBEHEERDERS = {
    "Coteq Netbeheer": ["Coteq", "COTEQ", "coteq"],
    "Enduris B.V.": ["Enduris", "ENDURIS", "enduris"],
    "Enexis B.V.": ["Enexis", "ENEXIS", "enexis"],
    "Liander N.V.": ["Liander", "LIANDER", "liander"],
    "Stedin Netbeheer B.V.": ["Stedin", "STEDIN", "stedin"],
    "Westland Infra B.V.": ["Westland", "WESTLAND", "westland"],
    "Rendo N.V.": ["Rendo", "RENDO", "rendo"],
    "Tennet": ["Tennet", "TENNET", "tennet"],
}

NETBEHEERDERS_LOWER = [
    "coteq",
    "enduris",
    "enexis",
    "liander",
    "stedin",
    "westland",
    "rendo",
    "tennet",
]

LEMMA_EXCLUDE = (
    [
        "lijst",
        "dossier",
        "onderscheid",
        "probleem",
        "verschil",
        "bestand",
        "overleg",
        "gebruik",
        "link",
        "werking",
        "samenvatting",
        "onderdeel",
        "installatie",
        "opsomming",
    ]
    + LIJST_SPECIFIEKE_COMPONENTEN
    + [variant for variants in NETBEHEERDERS.values() for variant in variants]
)

LEMMA_INCLUDE = []


# Definieer de componenten
COMPONENTS = {
    "10535": "LK ELA12 schakelinstallatie",
    "10536": "ABB VD4 vaccuum vermogensschakelaar",
    "10540": "Eaton L-SEP installatie",
    "10542": "Siemens NXplusC schakelaar",
    "10545": "Siemens 8DJH schakelaar",
    "10546": "Eaton FMX schakelinstallatie",
    "10551": "Laagspanning",
    "1555": "Merlin Gerin RM6 schakelaar",
    "1556": "Hazemeijer CONEL schakelinstallatie",
    "1557": "Eaton 10 kV COQ schakelaar",
    "1558": "Eaton Capitole schakelaar",
    "2059": "Eaton Xiria schakelinstallatie",
    "2061": "Eaton Holec SVS schakelaar",
    "2963": "MS/LS distributie transformator",
    "318": "Eaton Magnefix MD MF schakelinstallatie",
    "655": "ABB DR12 schakelaar",
    "8825": "ABB Safe safeplus schakelinstallatie",
    "8827": "MS kabelmoffen",
    "9026": "Eaton MMS schakelinstallatie",
    "9027": "ABB BBC DB10 schakelaar",
    "9028": "HS MS vermogens transformator",
}

# Cypher config
COLUMN_MAPPING_FAALVORM = {
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

STOPWORDS = {"de", "het", "een", "en", "van", "voor", "op"}

QUANTITY_TERMS = ["hoeveel", "populatie", "hoeveelheid", "aantal", "totaal", "telling"]
