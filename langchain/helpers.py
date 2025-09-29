import re
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import spacy
from config import (
    PATROON_UITBREIDING,
    COMPONENTS,
    LIJST_SPECIFIEKE_COMPONENTEN,
    PATH_SUMMARY,
)
from typing import List
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reranker = CrossEncoder(
    "NetherlandsForensicInstitute/robbert-2023-dutch-base-cross-encoder", device=device
)

# Gebruiken we voor het definieren van zelfstandig naamwoorden
NLP = spacy.load("nl_core_news_sm")


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


def clean_answer(answer: str) -> str:
    marker = "Ik weet het antwoord niet."
    index = answer.find(marker)
    if index != -1:
        return answer[index:].strip()
    return answer.strip()


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
    return clean_answer(remove_last_unfinished_sentence(remove_repetitions(tekst)))


def get_embedding_function():
    embedding_encode_kwargs: dict = {"normalize_embeddings": True}
    device = torch.device("cuda")
    embedding_model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name="NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
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
        for component in LIJST_SPECIFIEKE_COMPONENTEN:
            if component in waarde.lower() and component in vraag:
                gevonden_sleutels.append(sleutel)
                break

    # FIXME kan nog niet omgaan met verschillende dossiers
    # if not gevonden_sleutels:
    #     for sleutel, waarde in componenten_dict.items():
    #         for woord in LIJST_ALGEMENE_WOORDEN:
    #             if woord in waarde.lower() and woord in vraag:
    #                 gevonden_sleutels.append(sleutel)
    #                 break

    return {"type_id": gevonden_sleutels[0]} if len(gevonden_sleutels) == 1 else None


def extract_nouns_and_propn(text: str, include_nouns) -> List[str]:
    """Extract common nouns and proper nouns from Dutch text."""
    doc = NLP(text)
    list_nouns = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
    expanded_nouns = []
    for noun in list_nouns:
        for pattern, extras in PATROON_UITBREIDING.items():
            if pattern.lower() in noun.lower():  # case-insensitive match
                if len(extras) > 0:
                    expanded_nouns.extend(extras)
                expanded_nouns.append(noun)
    if include_nouns:
        total_list = expanded_nouns + list_nouns
        return list(set(total_list))  # remove duplicates
    else:
        return list(set(expanded_nouns))


def similarity_search_with_nouns(query: str, include_nouns):
    nouns = extract_nouns_and_propn(query, include_nouns)
    if not nouns:
        return None
    if len(nouns) == 1:
        return {"$contains": nouns[0]}
    else:
        return {"$or": [{"$contains": noun} for noun in nouns]}


def find_relevant_context(
    prompt: str,
    filter_chroma: dict[str, str],
    db: Chroma,
    source_max_reranker: int,
    source_max_dense: int,
    score_threshold: float,
    where_document,
    include_summary: int,
    nx_max=20000,
):
    """Find the relevant context from Chroma based on prompt and filter."""

    results = db.similarity_search_with_score(
        prompt, k=source_max_dense, filter=filter_chroma, where_document=where_document
    )
    # Filter by score
    results = [(doc, score) for doc, score in results if score < score_threshold]

    if source_max_reranker:
        results = rerank(prompt, results, top_m=source_max_reranker)

    summary = ""
    if include_summary:
        if filter_chroma:
            type_id = filter_chroma.get("type_id")
            if type_id:
                file_path = f"{PATH_SUMMARY}/{type_id}.txt"
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        summary = f.read() + "\n"
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    # Combine page content
    context_text = summary + "\n".join([doc.page_content for doc, _ in results])
    # Truncate by max characters
    # context_text = context_text[:nx_max]
    return context_text, results, summary


def rerank(query, candidates, top_m: int):
    """
    query: str
    candidates: list of (text, metadata)
    Returns top_m candidates ranked by relevance.
    """
    # Prepare input pairs (query, candidate text)
    input_pairs = [(query, text.page_content) for text, _ in candidates]
    input_docs_no_score = [text for text, _ in candidates]
    # Get relevance scores
    scores = reranker.predict(input_pairs)
    # Combine candidates with scores
    scored = list(zip(input_docs_no_score, scores))
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_m]


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
    print(extract_nouns_and_propn("Wat is het vervangingsbeleid van de xiria?"))
