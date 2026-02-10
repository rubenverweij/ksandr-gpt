"""
This module provides helper functions, data structures, and request models for the Ksandr platform.

It supports natural language processing tasks such as noun extraction, constructing filters for metadata querying,
embedding generation, query reranking, and input cleaning for downstream language model processing.
Also included are pydantic-based schemas for request validation, configuration constants, and utility functions
used throughout the Ksandr LLM API infrastructure.
"""

import re
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import pickle
import json
import os
from typing import Dict, Optional, Union, List, Any, Set
import string
import time
import spacy
from config import (
    COMPONENTS,
    LIJST_SPECIFIEKE_COMPONENTEN,
    LEMMA_EXCLUDE,
    NETBEHEERDERS,
    LOCATION_QUESTIONS,
    WEBLOCATION_TEMPLATE,
)
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from templates import SYSTEM_PROMPT
from pydantic import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reranker = CrossEncoder(
    "NetherlandsForensicInstitute/robbert-2023-dutch-base-cross-encoder", device=device
)
WORD_RE = re.compile(r"^[A-Za-zÀ-ÿ]+$")  # ondersteunt ook accenten


class AskRequest(BaseModel):
    prompt: str
    permission: Optional[Dict[str, Union[Dict[str, List[int]], List[int], bool]]] = None
    user_id: Optional[str] = "123"
    rag: Optional[int] = 1

    class Config:
        extra = "allow"  # Allow extra fields


class LLMRequest(BaseModel):
    n_ctx: int


class FileRequest(BaseModel):
    file_path: str | None = None
    summary_file_path: str | None = None
    summary_length: int = 500
    type: str | None = None
    id: str | None = None
    content: str | None = None

    class Config:
        extra = "allow"  # Allow extra fields


class ContextRequest(BaseModel):
    prompt: str


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
    """
    Checks whether a given sentence has at least a minimum number of words.

    Args:
        sentence (str): The sentence to check.
        min_words (int, optional): The minimum number of words required for the sentence to be considered valid.
            Defaults to MIN_WORDS, which is 3.

    Returns:
        bool: True if the sentence contains at least min_words words, False otherwise.
    """
    words = [w for w in sentence.strip().split() if w]
    return len(words) >= min_words


def get_7grams(words) -> list[tuple]:
    """
    Generates a list of 7-grams (tuples of seven consecutive words) from the provided list of words.

    Args:
        words (list of str): The input list of words from which to extract 7-grams.

    Returns:
        list of tuple: A list of 7-word tuples, where each tuple represents a 7-gram from the input.
    """
    return [tuple(words[i : i + 7]) for i in range(len(words) - 7 + 1)]


def normalize(s: str) -> str:
    """
    Normalizes the input string by lowercasing all characters and reducing any sequence of whitespace
    characters to a single space.

    Args:
        s (str): The input string to normalize.

    Returns:
        str: The normalized string, lowercased and with excess whitespace removed.
    """
    return " ".join(s.lower().split())


def sentence_in_previous(current: str, previous: list[str]) -> bool:
    """
    Checks if the normalized version of the current sentence is present in any of the normalized previous sentences.

    Args:
        current (str): The sentence to check for duplication.
        previous (list[str]): List of previous sentences for comparison.

    Returns:
        bool: True if the current sentence is a substring of any previous sentence after normalization, False otherwise.
    """
    cur = normalize(current)
    for prev in previous:
        if cur in normalize(prev):
            return True
    return False


def filter_words_only(words) -> list[str]:
    """
    Filters the input list of words, returning only those that match the WORD_RE pattern.

    Args:
        words (list of str): List of words to filter.

    Returns:
        list of str: List containing only words that match the WORD_RE regular expression.
    """
    return [w for w in words if WORD_RE.match(w)]


def clean_text_with_dup_detection(text) -> str:
    """
    Cleans the input text and removes duplicate or near-duplicate sentences and list items.

    Processes the given text by splitting it into blocks, detecting headers and list items,
    and filtering out repeated or highly similar content using sentence and 7-gram matching.
    Useful for preprocessing unstructured documents to reduce redundancy.

    Args:
        text (str or iterable of str): Input text or list of text blocks to clean.

    Returns:
        str: The cleaned text with duplicates removed, ready for further processing.
    """

    stream = text if isinstance(text, str) else "".join(text)

    HEADER_RE = re.compile(r"^\s*(\*\*.+?\*\*|[A-Z][^.!?]{0,80}):\s*$")
    LIST_ITEM_RE = re.compile(r"^(\s*(?:\d+\.\s+|[-*•]\s+))(.*)$")

    seen_sentences = []
    seen_7grams = set()

    seen_list_sentences = []
    seen_list_7grams = set()

    cleaned_blocks = []

    blocks = re.split(r"\n{2,}", stream)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # ── HEADER ─────────────────────────────
        if HEADER_RE.match(block):
            cleaned_blocks.append(block)
            continue

        # ── LIST ITEM ──────────────────────────
        m = LIST_ITEM_RE.match(block)
        if m:
            prefix, content = m.groups()
            content = content.strip()

            if not content or not is_valid_sentence(content):
                continue

            # Dedup on content
            if sentence_in_previous(content, seen_list_sentences):
                continue

            if any(
                normalize(prev) in normalize(content) for prev in seen_list_sentences
            ):
                continue

            words = filter_words_only(content.split())
            grams = get_7grams(words) if len(words) >= 7 else []

            if any(g in seen_list_7grams for g in grams):
                continue

            # Accept list item
            seen_list_sentences.append(content)
            seen_list_7grams.update(grams)
            cleaned_blocks.append(prefix + content)
            continue

        # ── PARAGRAPH ──────────────────────────
        sentences = re.split(r"(?<=[.!?])\s+", block)
        kept = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or not is_valid_sentence(sentence):
                continue

            if sentence_in_previous(sentence, seen_sentences):
                continue

            if any(normalize(prev) in normalize(sentence) for prev in seen_sentences):
                continue

            words = filter_words_only(sentence.split())
            grams = get_7grams(words) if len(words) >= 7 else []

            if any(g in seen_7grams for g in grams):
                continue

            kept.append(sentence)
            seen_sentences.append(sentence)
            seen_7grams.update(grams)

        if kept:
            cleaned_blocks.append(" ".join(kept))

    return "\n\n".join(cleaned_blocks)


def clean_answer_when_unknown(answer: str) -> str:
    """
    Cleans the answer string by checking for a specific marker indicating the model does not know the answer.

    If the marker "Ik weet het antwoord niet." is present in the answer, the function returns the substring starting
    from that marker. Otherwise, it returns the stripped original answer.

    Args:
        answer (str): The answer string to be cleaned.

    Returns:
        str: A cleaned version of the answer, either the marker message or the stripped answer.
    """
    marker = "Ik weet het antwoord niet."
    index = answer.find(marker)
    if index != -1:
        return answer[index:].strip()
    return answer.strip()


def remove_unfinished_sentences(text: str) -> str:
    """
    Removes any incomplete sentences at the end of the given text.

    This function attempts to find the last complete sentence in the text,
    defined as a sentence ending with a period. If the text ends with a
    partial or unfinished sentence that does not end with a period, it is removed.

    Args:
        text (str): The input text from which incomplete ending sentences should be removed.

    Returns:
        str: The text truncated so that it ends with the last complete sentence.
    """
    # Look for the last complete sentence ending with a period
    match = re.search(r"(.*?[\.|]\s*)([^\.|\n]*?)$", text, re.DOTALL)
    if match:
        before_last, after_last = match.groups()
        # If the trailing part after the last '.' has no '.' and is not empty,
        # we remove it
        if after_last.strip() and not after_last.strip().endswith("."):
            return before_last.rstrip()
    return text.rstrip()


def clean_llm_answer(text) -> str:
    """
    Processes a text by first removing incomplete sentences at the end and then cleaning the answer.

    This function successively uses 'verwijder_onafgeronde_zinnen' to ensure that the text ends with a complete sentence,
    and 'schoon_antwoord' to appropriately extract any redundant text or standard answers.

    Args:
        text (str): The text to be processed.

    Returns:
        str: The cleaned text without incomplete sentences at the end.
    """
    return clean_answer_when_unknown(remove_unfinished_sentences(text))


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings object configured for Dutch sentence embeddings.

    This function initializes the HuggingFaceEmbeddings using the
    "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers" model,
    with normalization of embeddings enabled and computation set to run on CUDA-enabled GPU.

    Returns:
        HuggingFaceEmbeddings: Embeddings function for use in vector stores or other NLP tasks.
    """
    embedding_encode_kwargs: dict = {"normalize_embeddings": True}
    device = torch.device("cuda")
    embedding_model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name="NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs,
    )
    return embeddings


def build_permission_filter_vector_store(
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


def get_aad_based_on_question(vraag: str) -> list[str]:
    """Return list with AAD based on question.

    Args:
        vraag (str): Question as string

    Returns:
        list[str]: list with AAD ids
    """
    vraag = vraag.lower()
    gevonden_sleutels = []
    for sleutel, waarde in COMPONENTS.items():
        for component in LIJST_SPECIFIEKE_COMPONENTEN:
            if component in waarde.lower() and component in vraag:
                gevonden_sleutels.append(sleutel)
                break
    return gevonden_sleutels


def create_metadata_filter(
    request: AskRequest, componenten_dict: dict, include_permission: bool
) -> dict:
    """
    Creates a filter based on the question in the request and available components.
    Used to retrieve relevant data from a vectorstore such as Chroma.

    Parameters:
    request (AskRequest): The user's question including permission data.
    componenten_dict (dict): A dictionary where the key is a component ID and the value is the name.
    include_permission: whether or not to include permission

    Returns:
    dict: A composite Chroma filter with type IDs (if found) and permissions.
    """
    vraag = (
        request.prompt.lower()
    )  # Converteer de gebruikersvraag naar kleine letters voor betere matching
    permissie_filter = None
    if include_permission:
        permissie_filter = build_permission_filter_vector_store(
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


def extract_nouns(text):
    """
    Extracts all nouns and proper nouns from the provided text using the loaded NLP model.

    Parameters:
    text (str): The input text from which to extract nouns.

    Returns:
    list: A list of spaCy Token objects representing the nouns and proper nouns found in the text.
    """
    doc = NLP(text)
    nouns = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            cleaned = token.text.strip(string.punctuation).lower()
            if cleaned:
                nouns.append(token)
    return nouns


def extract_dso_variants(question, netbeheerders_dict):
    """
    Extracts all variants of netbeheerders (network managers) mentioned in the question.

    Parameters:
    question (str): The user's question or input text.
    netbeheerders_dict (dict): Dictionary mapping netbeheerder names to lists of their variants.

    Returns:
    set: A set containing all matched variants (across all netbeheerders) found in the question.
    """
    question_lower = question.lower()
    matched_variants = set()
    for variants in netbeheerders_dict.values():
        for variant in variants:
            if variant.lower() in question_lower:
                matched_variants.update(variants)
                break  # Eén match per netbeheerder is genoeg
    return matched_variants


def create_chroma_filter(question, include_nouns) -> dict | None:
    """
    Constructs a filter for use with the Chroma vector store, based on extracted features from a question.

    Parameters:
    question (str): The input question or text from which to extract filtering features.
    include_nouns (bool): Whether to include relevant noun variants (excluding very common ones) in the constructed filter.

    Returns:
    dict or None: A Chroma-compatible filter dictionary combining relevant noun variants,
                  netbeheerder (grid operator) variants, and years found in the question,
                  or None if no relevant attributes are present.
    """
    noun_variants = set()
    filters = []
    netbeheerder_variants = set()
    # Voeg netbeheerder-varianten toe als ze voorkomen in de vraag
    netbeheerder_variants.update(extract_dso_variants(question, NETBEHEERDERS))
    # Verwerk zelfstandige naamwoorden als include_nouns True is
    if include_nouns:
        for token in extract_nouns(question):
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
    """
    Tel het aantal tokens in de opgegeven tekst met behulp van het doorgegeven model.

    Args:
        model: Het model met een 'tokenize' methode die tekst omzet in tokens (meestal bytes).
        text (str): De tekst waarvan het aantal tokens geteld moet worden.

    Returns:
        int: Het aantal tokens in de tekst volgens het tokenisatieschema van het model.
    """
    return len(model.tokenize(text.encode("utf-8")))


def trim_context_to_fit(
    model, template: str, context_text: str, question: str, n_ctx: int, max_tokens: int
) -> str:
    """
    Trim het contextgedeelte van de prompt zodat het binnen de toegestane contextlengte en maximaal aantal tokens past.

    Deze functie meet het aantal tokens dat wordt ingenomen door de prompt (zonder context) en bepaalt vervolgens hoeveel tokens
    er maximaal overblijven voor `context_text`. Indien nodig, wordt `context_text` van voren afgeknipt zodat alleen de laatste
    tokens gebruikt worden, zonder het maximum te overschrijden.

    Args:
        model: Het model (met tokenize en detokenize methodes) waarmee de tekst wordt getokeniseerd.
        template (str): Het prompt-template met placeholders voor system_prompt, context en question.
        context_text (str): De contexttekst die aangepast (afgeknipt) kan worden.
        question (str): De vraag die gesteld wordt aan het model.
        n_ctx (int): Maximale contextlengte in tokens voor het model.
        max_tokens (int): Maximaal aantal tokens gereserveerd voor de modeloutput.

    Returns:
        Tuple[int, str]: Het maximaal aantal beschikbare tokens voor context,
        en de (mogelijk afgeknipte) context_text.
    """
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


def give_aad_priority(documents, source_max_dense):
    """
    Sorteert documenten op basis van categorieprioriteit.

    Documenten met het metadata-veld "extension" gelijk aan "json" krijgen een lagere prioriteit
    door hun score met 0.85 te vermenigvuldigen. Documenten met een lagere (aangepaste) score worden
    als meer relevant beschouwd. Geeft de top `source_max_dense` documenten terug, gesorteerd op de (aangepaste) score.

    Args:
        documents (list of tuple): Lijst van (doc, score) tuples.
        source_max_dense (int): Aantal topdocumenten dat teruggegeven moet worden.

    Returns:
        list of tuple: Top `source_max_dense` (doc, score) tuples, gesorteerd op aangepaste score,
        waarbij lagere scores als beter worden beschouwd.
    """
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


def find_relevant_sources(
    prompt: str,
    filter_chroma: dict[str, str],
    db: Chroma,
    source_max_reranker: int,
    source_max_dense: int,
    score_threshold: float,
    where_document,
) -> tuple:
    """
    Vind de relevante context uit de Chroma database op basis van de gegeven prompt en filter.

    Args:
        prompt (str): De zoekvraag of prompt die gebruikt wordt voor context retrieval.
        filter_chroma (dict[str, str]): Een filterdict voor Chroma om de zoekresultaten te beperken.
        db (Chroma): De Chroma database waarin gezocht wordt.
        source_max_reranker (int): Maximum aantal bronnen dat na similarity search overblijft voor rerank.
        source_max_dense (int): Maximum aantal bronnen dat wordt opgehaald uit de dense similarity search.
        score_threshold (float): Drempelwaarde voor de similarity score; hogere scores worden gefilterd.
        where_document: Extra filter op documenten (kan None zijn).

    Returns:
        tuple:
            - context_text (str): De samengestelde contexttekst als input voor het LLM.
            - results (list): Lijst van tuples (doc, score) van gevonden documenten na filtering.
            - time_stages (dict): Tijdmetingen voor verschillende stadia in de pipeline.

    Gebruik:
        Wordt gebruikt om relevante tekstuele context te vinden bij een vraag, waarbij Chroma als vectorstore fungeert en verdere filtering/ranking wordt toegepast.
    """
    time_start = time.time()

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
        results = rerank_candidates(prompt, results, top_m=source_max_reranker)
    # Filter by score
    time_reranker = time.time()
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


def rerank_candidates(query, candidates, top_m: int):
    """
    Rerank candidates given a query using the reranker model.

    Parameters
    ----------
    query : str
        The reference query or question to compare with the candidates.
    candidates : list of (text, metadata)
        List of tuples containing candidate page_content and associated scores or metadata.

    Returns
    -------
    list
        Top `top_m` candidates (tuples) ranked by relevance as determined by the reranker.
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
    """
    Detects if the question pertains to an AAD.

    This function can be used to heuristically determine if the input question is about
    AADs by checking for the word "faalvorm" in the input string.
    Returns 1 if such a pattern is detected, 0 otherwise.

    Parameters
    ----------
    question : str
        The user's question or input.

    Returns
    -------
    int
        1 if the question is likely about an AAD ('faalvorm' present), 0 otherwise.
    """
    if "faalvorm" in question:
        return 1
    return 0


def detect_location(question) -> int:
    """
    Detects if the given question likely pertains to a location query.

    This function checks the provided question for the presence of certain keyword patterns
    specified in the LOCATION_QUESTIONS list. If any of the patterns (combinations of substrings)
    are found in order in the question (case-insensitive), it returns 1 (True), indicating a likely
    location-related question. Otherwise, it returns 0 (False).

    Parameters
    ----------
    question : str
        The user's question or input to analyze.

    Returns
    -------
    int
        1 if a location pattern is detected, 0 otherwise.
    """
    zin_lc = question.lower()
    for combi in LOCATION_QUESTIONS:
        pattern = r".*".join(combi)
        if re.search(pattern, zin_lc):
            return 1
    return 0


def build_links(aads) -> str:
    """
    Build clickable weblinks for given AAD dossier IDs.

    Parameters
    ----------
    aads : list
        List of AAD dossier IDs (as strings or integers).

    Returns
    -------
    str
        Multiline string with headers per AAD and corresponding weblocatie links,
        or an example section if the list is empty.
    """
    answer_list = []
    if len(aads) > 0:
        for aad in aads:
            answer_list.append(
                f"Weblocaties voor het AAD dossier van {COMPONENTS[str(aad)]}"
            )
            for loc in WEBLOCATION_TEMPLATE:
                answer_list.append(loc.format(id=aad))
            answer_list.append("")
        return "\n".join(answer_list)
    else:
        aad = "318"  # example
        answer_list.append(
            f"Voorbeeld locaties voor het AAD dossier van {COMPONENTS[str(aad)]}"
        )
        for loc in WEBLOCATION_TEMPLATE:
            answer_list.append(loc.format(id=aad))
        answer_list.append("")
        return "\n".join(answer_list)


def source_document_dummy():
    """
    Returns a dummy source document dictionary, for use in testing or placeholder data.

    Returns
    -------
    list of dict
        A list with one example document dictionary, containing keys 'id', 'metadata', and 'type'.
    """
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


def summary_request(question: str) -> bool:
    """
    Bepaalt of de gebruikersvraag betrekking heeft op het opvragen van een samenvatting van een document.

    Parameters
    ----------
    question : str
        De gebruikersvraag die geanalyseerd moet worden.

    Returns
    -------
    bool
        True als de vraag gerelateerd is aan het opvragen van een samenvatting van een document, anders False.
    """
    question = question.lower()
    if "samenvatting van document" in question:
        return True
    elif "samenvatting van bestand" in question:
        return True
    elif "samenvatting van de notulen" in question:
        return True
    else:
        False


def get_summary(question) -> str:
    """
    Geeft de schone samenvatting van een document op basis van een vraag.

    Parameters
    ----------
    question : str
        De gebruikersvraag waaruit het documentnummer wordt gehaald.

    Returns
    -------
    str
        De samenvatting van het document als deze gevonden wordt, anders een foutmelding.
    """
    number = re.search(r"\d+", question).group()
    try:
        with open(
            f"/root/onprem_data/summary/{number}.txt", "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        return data.get("summary_cleaned")
    except OSError:
        return f"Samenvatting van document {number} kan niet worden gevonden"


def text_quality_metrics(
    text: str,
    dictionary: Set[str] | None = None,
) -> Dict[str, float]:
    """
    Bereken kwaliteitsmetrics voor PDF/OCR textextractie.

    Metrics:
    - avg_word_length
    - short_word_ratio (woorden met lengte 1-2)
    - dictionary_hit_ratio
    - avg_chars_per_line
    - whitespace_ratio

    Parameters
    ----------
    text : str
        Geëxtraheerde tekst
    dictionary : set[str], optional
        Set met geldige woorden (lowercase). Indien None, wordt
        dictionary_hit_ratio = NaN.

    Returns
    -------
    dict
        Metrics als floats
    """

    if not text.strip():
        return {
            "avg_word_length": 0.0,
            "short_word_ratio": 1.0,
            "dictionary_hit_ratio": float("nan"),
            "avg_chars_per_line": 0.0,
            "whitespace_ratio": 0.0,
        }

    # --- Basis ---
    chars = len(text)
    whitespace_chars = sum(1 for c in text if c.isspace())

    lines = [line for line in text.splitlines() if line.strip()]
    avg_chars_per_line = chars / max(len(lines), 1)

    # --- Woorden ---
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    total_words = len(words)

    if total_words == 0:
        avg_word_length = 0.0
        short_word_ratio = 1.0
        dictionary_hit_ratio = float("nan")
    else:
        word_lengths = [len(w) for w in words]
        avg_word_length = sum(word_lengths) / total_words

        short_words = sum(1 for w in words if len(w) <= 2)
        short_word_ratio = short_words / total_words

        if dictionary is not None:
            valid_words = sum(1 for w in words if w.lower() in dictionary)
            dictionary_hit_ratio = valid_words / total_words
        else:
            dictionary_hit_ratio = float("nan")

    whitespace_ratio = whitespace_chars / chars

    return {
        "avg_word_length": round(avg_word_length, 3),
        "short_word_ratio": round(short_word_ratio, 3),
        "dictionary_hit_ratio": (
            round(dictionary_hit_ratio, 3) if dictionary is not None else float("nan")
        ),
        "avg_chars_per_line": round(avg_chars_per_line, 1),
        "whitespace_ratio": round(whitespace_ratio, 3),
    }


def is_llm_appropiate(metrics: Dict[str, float]) -> bool:
    """Verify that the extraction is appropiate

    Args:
        metrics (Dict[str, float]): _description_

    Returns:
        bool: _description_
    """
    return not (
        metrics["avg_word_length"] < 3.0
        or metrics["short_word_ratio"] > 0.4
        or metrics["avg_chars_per_line"] < 30
        or metrics["whitespace_ratio"] > 0.30
        or (
            not isinstance(metrics["dictionary_hit_ratio"], float)
            or metrics["dictionary_hit_ratio"] < 0.6
        )
    )


def clean_for_llm(text: str) -> str:
    """
    Clean a string for downstream use in Large Language Models (LLMs).

    This function normalizes text by standardizing line endings, removing extraneous whitespace,
    collapsing multiple spaces and tabs within lines, and reducing long sequences of blank lines.
    It is intended to produce concise, consistently formatted input appropriate for LLM retrieval or inference.

    Args:
        text (str): The input string to be cleaned.

    Returns:
        str: The cleaned and normalized text.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces on each line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Collapse multiple spaces/tabs inside lines
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines into just 2 (1 blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_concluding_chunk(chunks: list[str]) -> list[str]:
    """
    Detect chunks (strings) that appear to contain conclusions, advice, or results.

    Args:
        chunks (list[str]): List of text chunks (strings) to be inspected.

    Returns:
        list[str]: List of chunks which contain words indicating conclusions or results (case-insensitive),
                   specifically those containing "conclusie", "advies", or "resultaat".
    """
    patterns = ["conclusie", "advies", "resultaat"]
    regex = re.compile(r"|".join(patterns), re.IGNORECASE)
    matching_chunks = [chunk for chunk in chunks if regex.search(chunk)]
    return matching_chunks


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
        gevonden_sleutels = create_metadata_filter(
            vraag,
            COMPONENTS,
        )
        print(f"Gevonden component sleutels: {gevonden_sleutels}")
        print("-" * 40)
