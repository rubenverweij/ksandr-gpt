import re
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import pickle
import string
import spacy
from config import (
    COMPONENTS,
    LIJST_SPECIFIEKE_COMPONENTEN,
    PATH_SUMMARY,
    LEMMA_EXCLUDE,
    LEMMA_INCLUDE,
    NETBEHEERDERS,
)
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reranker = CrossEncoder(
    "NetherlandsForensicInstitute/robbert-2023-dutch-base-cross-encoder", device=device
)

# Gebruiken we voor het definieren van zelfstandig naamwoorden
NLP = spacy.load("nl_core_news_sm")

# Variables to make use of keywords that are relevant
FREQUENCY_THRESHOLD = 5  # Only include nouns used at least this many times
FREQUENCY_THRESHOLD_MAX = 6000

# Load saved data
with open("/root/onprem_data/keywords/noun_counter.pkl", "rb") as f:
    LEMMA_COUNTS = pickle.load(f)

with open("/root/onprem_data/keywords/lemmas.pkl", "rb") as f:
    LEMMA_TO_VARIANTS = pickle.load(f)


def verwijder_herhalingen(text: str) -> str:
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


def schoon_antwoord(answer: str) -> str:
    marker = "Ik weet het antwoord niet."
    index = answer.find(marker)
    if index != -1:
        return answer[index:].strip()
    return answer.strip()


def verwijder_onafgeronde_zinnen(text: str) -> str:
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
    return schoon_antwoord(verwijder_onafgeronde_zinnen(verwijder_herhalingen(tekst)))


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
    Zoekt naar relevante componenten op basis van de vraag. Retourneert een Chroma filter.

    Parameters:
    vraag (str): De vraag van de gebruiker.
    componenten_dict (dict): Een dictionary van componenten, met de sleutel als ID en de waarde als naam.

    Returns:
    dict: Chroma filter met 'type_id' als lijst van relevante component-IDs (kan leeg zijn).
    """
    vraag = vraag.lower()
    gevonden_sleutels = []
    for sleutel, waarde in componenten_dict.items():
        for component in LIJST_SPECIFIEKE_COMPONENTEN:
            if component in waarde.lower() and component in vraag:
                gevonden_sleutels.append(sleutel)
                break
    if gevonden_sleutels:
        return {"type_id": {"$in": gevonden_sleutels}}
    else:
        return None


def extraheer_zelfstandig_naamwoorden(text):
    doc = NLP(text)
    nouns = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            cleaned = token.text.strip(string.punctuation).lower()
            if cleaned:
                nouns.append(token)
    return nouns


def extract_netbeheerder_variants(question, netbeheerders_dict):
    question_lower = question.lower()
    matched_variants = set()
    for variants in netbeheerders_dict.values():
        for variant in variants:
            if variant.lower() in question_lower:
                matched_variants.update(variants)
                break  # Eén match per netbeheerder is genoeg
    return matched_variants


def maak_chroma_filter(question, include_nouns):
    noun_variants = set()
    netbeheerder_variants = set()
    # Voeg netbeheerder-varianten toe als ze voorkomen in de vraag
    netbeheerder_variants.update(extract_netbeheerder_variants(question, NETBEHEERDERS))
    # Verwerk zelfstandige naamwoorden als include_nouns True is
    if include_nouns:
        for token in extraheer_zelfstandig_naamwoorden(question):
            surface = token.text.strip(string.punctuation)
            lemma = token.lemma_.lower()
            count = LEMMA_COUNTS.get(lemma, 0)
            if (
                count >= FREQUENCY_THRESHOLD
                and count < FREQUENCY_THRESHOLD_MAX
                and lemma not in LEMMA_EXCLUDE
            ) or lemma in LEMMA_INCLUDE:
                variants = LEMMA_TO_VARIANTS.get(lemma, {surface})
                noun_variants.update(variants)
    # Extract jaartallen
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    # Bouw Chroma-filter
    filters = []
    if years:
        # flatten and remove duplicates, since re.findall with groups returns tuples
        year_values = sorted(set(years))
        if len(year_values) == 1:
            filters.append({"$contains": year_values[0]})
        else:
            filters.append({"$or": [{"$contains": year} for year in year_values]})
    if noun_variants:
        filters.append({"$or": [{"$contains": variant} for variant in noun_variants]})
    if netbeheerder_variants:
        filters.append(
            {"$or": [{"$contains": variant} for variant in netbeheerder_variants]}
        )
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def count_tokens(model, text: str) -> int:
    return len(model.tokenize(text.encode("utf-8")))


def trim_context_to_fit(
    model, template: str, context_text: str, question: str, n_ctx: int, max_tokens: int
) -> str:
    # Build a prompt with an empty context just to measure overhead
    dummy_prompt = template.format(context="", question=question)
    prompt_overhead_tokens = count_tokens(model, dummy_prompt)
    # Available space for context
    available_tokens_for_context = n_ctx - max_tokens - prompt_overhead_tokens
    context_tokens = model.tokenize(context_text.encode("utf-8"))
    if len(context_tokens) > available_tokens_for_context:
        trimmed_tokens = context_tokens[
            -available_tokens_for_context:
        ]  # Keep latest context
        context_text = model.detokenize(trimmed_tokens).decode("utf-8", errors="ignore")

    return available_tokens_for_context, context_text


def geef_categorie_prioriteit(documents, source_max_dense):
    ranked = sorted(
        documents,
        key=lambda doc_tuple: (
            doc_tuple[1] * 0.7
            if doc_tuple[0].metadata.get("extension") == "json"
            else doc_tuple[1]
        ),
        reverse=True,
    )
    return ranked[:source_max_dense]


def vind_relevante_context(
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
    max_dense = source_max_dense + 2
    results = db.similarity_search_with_score(
        prompt, k=max_dense, filter=filter_chroma, where_document=where_document
    )
    results = geef_categorie_prioriteit(results, source_max_dense)
    if source_max_reranker:
        results = herschik(prompt, results, top_m=source_max_reranker)
    # Filter by score
    results = [(doc, score) for doc, score in results if score < score_threshold]
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
    context_text = summary + "\n".join(
        [
            f"Bestand locatie en link {COMPONENTS.get(doc.metadata.get('source', ''), '')} en betreft {COMPONENTS.get(doc.metadata.get('type_id', ''), '')}. {doc.page_content}"
            for doc, _ in results
        ]
    )
    # Truncate by max characters
    # context_text = context_text[:nx_max]
    return context_text, results, summary


def herschik(query, candidates, top_m: int):
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
        verwijder_onafgeronde_zinnen(
            verwijder_herhalingen(
                text="De Xiria is een serie. \n De Xiria is een gave serie. De Xiria is een"
            )
        )
    )
