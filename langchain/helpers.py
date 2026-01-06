import re
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import pickle
import json
import os
from typing import Dict, Optional, Union, List, Any
import string
import time
import spacy
from config import (
    COMPONENTS,
    LIJST_SPECIFIEKE_COMPONENTEN,
    LEMMA_EXCLUDE,
    NETBEHEERDERS,
)
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from templates import SYSTEM_PROMPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reranker = CrossEncoder(
    "NetherlandsForensicInstitute/robbert-2023-dutch-base-cross-encoder", device=device
)

# Gebruiken we voor het definieren van zelfstandig naamwoorden
NLP = spacy.load("nl_core_news_sm")

# Variables to make use of keywords that are relevant
FREQUENCY_THRESHOLD = 5  # Only include nouns used at least this many times
FREQUENCY_THRESHOLD_MAX = 5e5
MIN_WORDS = 3  # <--- minimale aantal woorden voor een zin
SENTENCE_END_RE = re.compile(r"[.!?]")

# Load saved data
with open("/root/onprem_data/keywords/lemma_counter.pkl", "rb") as f:
    LEMMA_COUNTS = pickle.load(f)

with open("/root/onprem_data/keywords/lemmas.pkl", "rb") as f:
    LEMMA_TO_VARIANTS = pickle.load(f)


def is_valid_sentence(sentence: str, min_words: int = MIN_WORDS) -> bool:
    words = [w for w in sentence.strip().split() if w]
    return len(words) >= min_words


def get_7grams(words):
    return [tuple(words[i : i + 7]) for i in range(len(words) - 7 + 1)]


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def sentence_in_previous(current: str, previous: list[str]) -> bool:
    """Check if current sentence is fully contained inside any earlier one."""
    cur = normalize(current)
    for prev in previous:
        if cur in normalize(prev):
            return True
    return False


def clean_text_with_dup_detection(text) -> str:
    """
    Processes a text stream, removes or stops at:
    - full-sentence duplicates
    - sentences contained inside previous
    - previous contained inside current
    - 4-gram duplicate windows

    Returns cleaned string (up to the first duplication).
    """

    # Support: full text or streamed chunks
    if isinstance(text, str):
        stream = text
    else:
        stream = text  # assume generator or list of chunks

    seen_sentences = []
    seen_7grams = set()
    buffer = ""
    cleaned = ""

    for chunk in stream:
        token = chunk or ""
        buffer += token
        # Not the end of a sentence yet → continue accumulating
        if not SENTENCE_END_RE.search(token):
            continue
        # Split buffer at sentence boundaries
        parts = re.split(r"(?<=[.!?])\s*", buffer)
        completed = parts[:-1]  # full sentences
        buffer = parts[-1]  # remainder
        if not completed:
            continue
        sentence = completed[-1].strip()
        # Validate sentence length
        if not (sentence and is_valid_sentence(sentence)):
            continue
        if sentence_in_previous(sentence, seen_sentences):
            print(f"[Sentence contained in previous] {sentence}")
            return cleaned.strip()
        for prev in seen_sentences:
            if normalize(prev) in normalize(sentence):
                print(f"[Previous sentence is contained in current] {sentence}")
                return cleaned.strip()
        words = sentence.split()
        grams = get_7grams(words)
        if any(g in seen_7grams for g in grams):
            print(f"[7-gram duplicate detected] {sentence}")
            return cleaned.strip()
        cleaned += sentence + " "
        seen_sentences.append(sentence)
        for g in grams:
            seen_7grams.add(g)
    return cleaned.strip()


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
    return schoon_antwoord(verwijder_onafgeronde_zinnen(tekst))


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


def bouw_permissie_filter_vector_store(
    permissie_data: Optional[Dict[str, Union[Dict[str, List[int]], List[int], bool]]],
) -> Dict[str, Any]:
    """
    Bouwt een permissiefilter op basis van de opgegeven permissiegegevens.

    Parameters:
    permissie_data (dict of None): Een dictionary die de toegangsrechten specificeert.
        - De sleutels zijn bronnen (bijv. 'user', 'group', etc.).
        - De waarden kunnen zijn:
            - Een dictionary met categorieën als sleutel en lijsten van IDs als waarde.
            - Een lijst van IDs.
            - Een boolean (toegang of geen toegang).

    Returns:
    dict: Een filter die gebruikt kan worden in een query (bijv. voor Chroma of een database).
    """
    if not permissie_data:
        # Als er geen permissiegegevens zijn, is toegang tot 'table' niet toegestaan
        return {"table": False}

    permissions = []

    for bron, waarde in permissie_data.items():
        if isinstance(waarde, dict):
            # Als de waarde een dictionary is, doorloop dan elke categorie en bijbehorende ID's
            for categorie, ids in waarde.items():
                for id_ in ids:
                    permissions.append(f"{categorie}_{id_}")
        elif isinstance(waarde, list):
            # Als de waarde een lijst is, voeg elk ID toe met de bron als categorie
            for id_ in waarde:
                permissions.append(f"{id_}_{bron}")
        elif isinstance(waarde, bool):
            # Als de waarde een boolean is, voeg 'true_bron' of 'false_bron' toe
            permissions.append(f"{'true' if waarde else 'false'}_{bron}")

    permissions.append("true_general")
    # Retourneer een filter waarin gezocht wordt naar permissies die overeenkomen met de gegenereerde lijst
    return {"permission_and_type": {"$in": permissions}}


def haal_dossiers_op(vraag):
    """ "Haal relevante dossiers op."""
    vraag = vraag.lower()
    gevonden_sleutels = []
    for sleutel, waarde in COMPONENTS.items():
        for component in LIJST_SPECIFIEKE_COMPONENTEN:
            if component in waarde.lower() and component in vraag:
                gevonden_sleutels.append(sleutel)
                break
    return gevonden_sleutels


def maak_metadata_filter(request, componenten_dict, include_permission):
    """
    Maakt een filter op basis van de vraag in de request en beschikbare componenten.
    Wordt gebruikt om relevante data op te halen uit een vectorstore zoals Chroma.

    Parameters:
    request (AskRequest): De vraag van de gebruiker inclusief permissiegegevens.
    componenten_dict (dict): Een dictionary waarbij de sleutel een component-ID is en de waarde de naam.
    include_permission: wel of niet permissie meenemen

    Returns:
    dict: Een samengestelde Chroma-filter met type-ID's (indien gevonden) en permissies.
    """
    vraag = (
        request.prompt.lower()
    )  # Converteer de gebruikersvraag naar kleine letters voor betere matching
    permissie_filter = None
    if include_permission:
        permissie_filter = bouw_permissie_filter_vector_store(
            request.permission
        )  # Bouw het permissiefilter op basis van request

    gevonden_sleutels = []

    # Doorloop alle componenten om te kijken of er specifieke componenten in de vraag worden genoemd
    for sleutel, waarde in componenten_dict.items():
        for component in LIJST_SPECIFIEKE_COMPONENTEN:
            if component in waarde.lower() and component in vraag:
                gevonden_sleutels.append(sleutel)
                break  # Stop met zoeken zodra een match is gevonden voor deze component

    if gevonden_sleutels:
        if include_permission:
            # Als er relevante componenten zijn gevonden, combineer ze met het permissiefilter
            return {"$and": [{"type_id": {"$in": gevonden_sleutels}}, permissie_filter]}
        else:
            return {"type_id": {"$in": gevonden_sleutels}}
    else:
        # Als er geen componenten zijn gevonden, gebruik alleen het permissiefilter
        return permissie_filter


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
    filters = []
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
            ):
                variants = LEMMA_TO_VARIANTS.get(lemma, {surface})
                noun_variants.update(variants)
        if noun_variants:
            if len(noun_variants) == 1:
                filters.append({"$contains": noun_variants[0]})
            else:
                filters.append(
                    {"$or": [{"$contains": variant} for variant in noun_variants]}
                )
    # Extract jaartallen
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    # Bouw Chroma-filter
    if years:
        # flatten and remove duplicates, since re.findall with groups returns tuples
        year_values = sorted(set(years))
        if len(year_values) == 1:
            filters.append({"$contains": year_values[0]})
        else:
            filters.append({"$or": [{"$contains": year} for year in year_values]})
    if netbeheerder_variants:
        filters.append(
            {"$or": [{"$contains": variant} for variant in netbeheerder_variants]}
        )
    if len(filters) == 1:
        return filters[0]
    if len(filters) > 1:
        return {"$and": filters}
    return None


def count_tokens(model, text: str) -> int:
    return len(model.tokenize(text.encode("utf-8")))


def trim_context_to_fit(
    model, template: str, context_text: str, question: str, n_ctx: int, max_tokens: int
) -> str:
    # Build a prompt with an empty context just to measure overhead
    dummy_prompt = template.format(
        system_prompt=SYSTEM_PROMPT, context="", question=question
    )
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
    adjusted = [
        (
            doc,
            score * 0.85 if doc.metadata.get("extension") == "json" else score,
        )
        for doc, score in documents
    ]
    # lower lagere scores zijn beter
    ranked = sorted(adjusted, key=lambda x: x[1])
    return ranked[:source_max_dense]


def vind_relevante_context(
    prompt: str,
    filter_chroma: dict[str, str],
    db: Chroma,
    # db_json: Chroma,
    # include_db_json: bool,
    source_max_reranker: int,
    source_max_dense: int,
    score_threshold: float,
    # score_threshold_json: float,
    where_document,
    # include_summary: int,
):
    """Find the relevant context from Chroma based on prompt and filter."""
    time_start = time.time()

    # FIXME deprecated
    # Zoek eerst door de website/aads
    # if include_db_json:
    #     results = db_json.similarity_search_with_score(
    #         prompt,
    #         k=source_max_dense,
    #         filter=filter_chroma,
    #         where_document=where_document,
    #     )
    #     results = [
    #         (doc, score) for doc, score in results if score < score_threshold_json
    #     ]
    # else:
    #     results = []
    results = []
    if len(results) == 0:
        results = db.similarity_search_with_score(
            prompt,
            k=source_max_dense,
            filter=filter_chroma,
            where_document=where_document,
        )
        results = [(doc, score) for doc, score in results if score < score_threshold]
    time_sim_search = time.time()
    # FIXME deprecated
    # results = geef_categorie_prioriteit(results, source_max_dense)
    time_prio_cats = time.time()
    if source_max_reranker:
        results = herschik(prompt, results, top_m=source_max_reranker)
    # Filter by score
    time_reranker = time.time()

    # FIXME deprecated
    # summary = ""
    # if include_summary:
    #     if filter_chroma:
    #         type_id = filter_chroma.get("type_id")
    #         if type_id:
    #             file_path = f"{PATH_SUMMARY}/{type_id}.txt"
    #             try:
    #                 with open(file_path, "r", encoding="utf-8") as f:
    #                     summary = f.read() + "\n"
    #             except FileNotFoundError:
    #                 print(f"File not found: {file_path}")
    #             except Exception as e:
    #                 print(f"Error reading file {file_path}: {e}")
    # Combine page content
    context_text = "\n".join(
        [
            f"Het ID van deze bron is: {os.path.splitext(os.path.basename(doc.metadata.get('source', '')))[0]} en betreft component: {COMPONENTS.get(doc.metadata.get('type_id', ''), '')}. {doc.page_content}"
            for doc, _ in results
        ]
    )
    time_build_context = time.time()
    # Truncate by max characters
    return (
        context_text,
        results,
        {
            "time_stages": {
                "similarity_search_with_score": time_sim_search - time_start,
                "geef_categorie_prioriteit": time_prio_cats - time_sim_search,
                "herschik": time_reranker - time_prio_cats,
                "build_context": time_build_context - time_reranker,
            }
        },
    )


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


def detect_aad(question):
    if "faalvorm" in question:
        return 1
    return 0


def source_document_dummy():
    return [
        {
            "id": 12345678910,
            "metadata": {
                "source": "/root/ksandr_files/aads/10546/cat-1/main.json",
                "source_search": "/root/ksandr_files/aads/10546/cat-1/main.json",
                "file_path": "/root/ksandr_files/aads/10546/cat-1/main.json",
                "score": 0.50,
            },
            "type": "Document",
        }
    ]


def summary_request(question: str):
    if "samenvatting van document" in question:
        return True
    else:
        False


def get_summary(question):
    number = re.search(r"\d+", question).group()
    try:
        with open(
            f"/root/onprem_data/summary/{number}.txt", "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        return data.get("summary_cleaned")
    except OSError:
        return f"Samenvatting van document {number} kan niet worden gevonden"


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
        gevonden_sleutels = maak_metadata_filter(
            vraag,
            COMPONENTS,
        )
        print(f"Gevonden component sleutels: {gevonden_sleutels}")
        print("-" * 40)
