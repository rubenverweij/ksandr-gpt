import re

# Definieer de componenten
COMPONENTS = {
    "10535": "LK ELA12 schakelinstallatie",
    "10536": "ABB VD4 vaccuum vermogensschakelaar",
    "10540": "Eaton L-SEP installatie",
    "10542": "Siemens NXplusC schakelaar",
    "10545": "Siemens 8DJH schakelaar",
    "10546": "Eaton FMX schakelinstallatie",
    "1555": "Merlin Gerin RM6 schakelaar",
    "1556": "Hazemeijer CONEL schakelinstallatie",
    "1557": "Eaton 10 kV COQ schakelaar",
    "1558": "Eaton Capitole schakelaar",
    "2059": "Eaton Xiria schakelinstallatie",
    "2061": "Eaton Holec SVS schakelaar",
    "2963": "MS/LS distributie transformator",
    "318": "Eaton Magnefix MD MF schakelinstallatie",
    "655": "ABB DR12 schakelaar",
    "8825": "ABB Safe schakelinstallatie",
    "8827": "kabelmoffen",
    "9026": "Eaton MMS schakelinstallatie",
    "9027": "ABB BBC DB10 schakelaar",
    "9028": "HS MS vermogens transformator",
}


algemene_woorden = [
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

specifieke_componenten = [
    "db10",
    "bcc",
    "md",
    "mf",
    "magnefix",
    "merlin",
    "svs",
    "coq",
    "8djh",
    "vd4",
    "ela12",
    "l-sep",
    "rm6",
    "nxplusc",
    "mms",
    "fmx",
    "xiria",
    "capitole",
]


def vind_relevante_componenten(vraag, componenten_dict):
    """
    Zoekt naar relevante componenten op basis van de vraag. Eerst zoekt het naar specifieke componenten, en daarna naar
    algemene woorden als er geen specifieke match is.

    Parameters:
    vraag (str): De vraag van de gebruiker.
    componenten_dict (dict): Een dictionary van componenten, met de sleutel als ID en de waarde als naam.

    Returns:
    list: Lijst van de sleutels van de relevante componenten.
    """
    vraag = vraag.lower()

    gevonden_sleutels = []
    for sleutel, waarde in componenten_dict.items():
        for component in specifieke_componenten:
            if component in waarde.lower() and component in vraag:
                gevonden_sleutels.append(sleutel)
                break

    # FIXME kan nog niet omgaan met verschillende dossiers
    # if not gevonden_sleutels:
    #     for sleutel, waarde in componenten_dict.items():
    #         for woord in algemene_woorden:
    #             if woord in waarde.lower() and woord in vraag:
    #                 gevonden_sleutels.append(sleutel)
    #                 break

    return {"type_id": gevonden_sleutels[0]} if len(gevonden_sleutels) == 1 else None


def uniek_antwoord(tekst):
    zinnen = re.split(r"(?<=[.!?])\s+", tekst.strip())
    unieke_zinnen = set(zinnen)
    return " ".join(sorted(unieke_zinnen))


if __name__ == "__main__":
    # Voorbeeld van gebruik:
    vragen = [
        "Wat is het onderhoudsbeleid van de EATON MMS?",
        "Wat is het onderhoudsbeleid van EATON?",
        "Wat is het onderhoudsbeleid van de Siemens schakelinstallatie?",
        "Wat weet je van de DB10?",
        "Wat is het dossier van de 10 kV COQ?",
        "Wat is het dossier van de COnel?",
    ]

    for vraag in vragen:
        print(f"Vraag: {vraag}")
        gevonden_sleutels = vind_relevante_componenten(
            vraag,
            COMPONENTS,
        )
        print(f"Gevonden component sleutels: {gevonden_sleutels}")
        print("-" * 40)
