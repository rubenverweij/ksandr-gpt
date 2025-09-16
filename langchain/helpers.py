import re
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import spacy
from typing import List

# Gebruiken we voor het definieren van zelfstandig naamwoorden
NLP = spacy.load("nl_core_news_sm")

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


def remove_repetitions(text: str) -> str:
    # Split the text by newline first to preserve the structure
    sentences = text.split("\n")

    # Create a list to store the unique sentences and a set to track already seen sentences
    unique_sentences = set()
    cleaned_sentences = []

    for sentence in sentences:
        # Split into individual sentences (this works for sentences ending with punctuation)
        sub_sentences = re.split(
            r"(?<=\w[.!?])\s*", sentence.strip()
        )  # Split on sentence boundary

        cleaned_sub_sentences = []
        for sub_sentence in sub_sentences:
            # If the sentence has not been seen before, keep it
            if sub_sentence and sub_sentence not in unique_sentences:
                unique_sentences.add(sub_sentence)
                cleaned_sub_sentences.append(sub_sentence)

        # Rejoin sub-sentences if any unique sentences are present in the current sentence block
        if cleaned_sub_sentences:
            cleaned_sentences.append(" ".join(cleaned_sub_sentences))

    # Rebuild the text with cleaned sentences, preserving the newline structure
    return "\n".join(cleaned_sentences)


def remove_last_unfinished_sentence(text: str) -> str:
    # Look for the last complete sentence ending with a period
    match = re.search(r"(.*?\.\s*)([^\.\n]*?)$", text, re.DOTALL)
    if match:
        before_last, after_last = match.groups()
        # If the trailing part after the last '.' has no '.' and is not empty,
        # we remove it
        if after_last.strip() and not after_last.strip().endswith("."):
            return before_last.rstrip()
    return text.rstrip()


def uniek_antwoord(tekst):
    return remove_last_unfinished_sentence(remove_repetitions(tekst))


def get_embedding_function():
    embedding_encode_kwargs: dict = {"normalize_embeddings": False}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs,
    )
    return embeddings


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


def extract_nouns_and_propn(text: str) -> List[str]:
    """Extract common nouns and proper nouns from Dutch text."""
    doc = NLP(text)
    return [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]


def similarity_search_with_nouns(
    query: str,
):
    nouns = extract_nouns_and_propn(query)
    if not nouns:
        return None
    return {"$or": [{"$contains": noun} for noun in nouns]}


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

    print(
        remove_last_unfinished_sentence(
            remove_repetitions(
                text="De Xiria is een serie. \n De Xiria is een gave serie. De Xiria is een"
            )
        )
    )
