from typing import List, Dict

PATH_SUMMARY = "/root/onprem_data/summary"

PATROON_UITBREIDING: Dict[str, List[str]] = {
    "onderhoud": ["Onderhoud", "onderhoud"],
    "aantal": ["Populatie", "aantal", "populatie"],
    "populatie": ["Populatie", "aantal", "populatie"],
    "-": [],
    "faalvorm": ["Faalvorm", "faalvorm"],
    "vervanging": ["Vervanging", "vervanging"],
    "inspectie": ["Inspectie", "inspectie"],
    "eaton": ["Eaton"],
    "beleid": ["beleid", "Beleid"],
}

LIJST_ALGEMENE_WOORDEN = [
    "eaton",
    "siemens",
    "abb",
    "transformator",
    "merlin",
    "gerin",
    "holec",
    "conel",
    "hazemeijer",
    "lk",
]

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

LEMMA_EXCLUDE = ["lijst", "dossier", "onderscheid"] + LIJST_SPECIFIEKE_COMPONENTEN
LEMMA_INCLUDE = ["faalvorm", "inspectie", "fabrikant"]

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
