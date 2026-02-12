"""
This module provides helper functions for the ingestion pipeline of the Ksandr platform.

Key functionalities include:
- Basic text cleaning and structure analysis for use with vectorstores.
- Extraction of document metadata such as type, permissions, and file details from paths.
- Utilities for HTML content cleaning and safe parsing.

These helpers are intended to be imported in ingestion and database population scripts such as `populate_database.py` and `verwijder_duplicaten.py`.
"""

import os
import re
from typing import Dict, Union

from bs4 import BeautifulSoup


def looks_like_clean_text(text: str) -> bool:
    """
    Heuristic filter to determine if a text chunk is "clean" (i.e., usable) for vectorstore ingestion.

    Evaluates a text chunk for minimal quality and noise by checking, for example:
      - If it is non-empty after stripping whitespace.
      - The proportion of alphabetic words versus numerics and symbols.
      - If the chunk contains certain "header" or "irrelevant" markers (such as "Dossier" or "Populatiegegevens").
      - Presence of symbols (e.g., '>' may indicate markup or structure, which are considered unclean).

    Returns:
        bool: True if the chunk meets heuristic criteria for usefulness, False if considered noise/irrelevant.
    """
    # Remove newlines for analysis
    text = text.strip()
    if not text:
        return False
    # Count alphabetic words vs symbols/numbers
    words = re.findall(r"[a-zA-Z]+", text)
    numbers = re.findall(r"\d+", text)
    word_ratio = len(words) / max(len(text.split()), 1)
    number_ratio = len(numbers) / max(len(text.split()), 1)
    if word_ratio > 0.6 and number_ratio < 0.3:
        return True
    if ">" in text:
        return False
    if "Dossier" in text:
        return True
    if "Populatiegegevens" in text:
        return True
    return False


def extract_file_data(file_path: str) -> Dict[str, Union[int, str]]:
    """
    Extracts and infers metadata from a file path string for downstream document processing.

    Analyzes the provided file path to determine:
        - 'type': Main group or category (e.g., 'aads', 'documents', etc.)
        - 'type_id': Secondary type or identifier (if present)
        - 'permission': Access permission class or flag (parsed from position or inferred)
        - 'filename': Basename of the file (without extension)
        - 'permission_and_type': Combined string for grouped queries or filtering

    Supports analytic rules for a variety of expected directory structures (e.g., aads, general, documents, etc.)
    Returns default values if pattern is not recognized.

    Args:
        file_path (str): Full path string to the file.

    Returns:
        Dict[str, Union[int, str]]: Dictionary with extracted metadata fields.
    """
    parts = file_path.strip("/").split("/")
    result = {
        "type": "",
        "type_id": "na",
        "permission": "",
        "filename": "",
        "permission_and_type": "",
    }

    data_groups = ["aads", "general", "documents", "groups", "ese", "esg", "rmd", "dga"]

    if len(parts) >= 3:
        derde_deel = parts[2]
        if derde_deel in data_groups:
            result["type"] = derde_deel
            try:
                if derde_deel == "aads":
                    aads_index = parts.index("aads")
                    result["permission"] = (
                        parts[aads_index + 2]
                        if len(parts) > aads_index + 2
                        else "cat-3"
                    )
                    result["type_id"] = parts[aads_index + 1]
                    result["permission_and_type"] = (
                        f"{result['permission']}_{result['type_id']}"
                    )
                elif derde_deel in ["documents", "groups", "rmd", "dga"]:
                    result["permission"] = parts[parts.index(derde_deel) + 1]
                    result["permission_and_type"] = (
                        f"{result['permission']}_{result['type']}"
                    )
                elif derde_deel in ["ese", "esg", "general"]:
                    result["permission"] = "true"
                    result["permission_and_type"] = (
                        f"{result['permission']}_{result['type']}"
                    )
            except (IndexError, ValueError):
                pass

    if not result["type"]:
        result["type"] = "unknown"
    if not result["permission"]:
        result["permission"] = "undefined"

    result["filename"] = os.path.splitext(os.path.basename(file_path))[0]
    return result


def clean_html(value: str | dict | list):
    """
    Cleans a string, list, or dictionary by removing HTML tags and normalizing whitespace.

    For string input, all HTML tags are stripped and excess whitespace is collapsed.
    For dictionaries and lists, all elements/values are recursively cleaned.
    Removes 'tags' key from dictionaries and omits empty/null values.

    Args:
        value (str | dict | list): The value to clean.

    Returns:
        Cleaned version of the original value with HTML stripped and extraneous
        whitespace removed. Returns None for empty values.
    """
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        # Verwijder HTML-tags
        cleaned = BeautifulSoup(value, "html.parser").get_text(separator=" ")
        # Verwijder overtollige witruimte
        return re.sub(r"\s+", " ", cleaned).strip()
    if isinstance(value, dict):
        value.pop("tags", None)
        return {k: clean_html(v) for k, v in value.items() if clean_html(v) is not None}
    if isinstance(value, list):
        lijst = [clean_html(v) for v in value if clean_html(v) is not None]
        return lijst if lijst else None
    return value


def prepare_text_for_vector_store(text: str) -> str:
    """
    Cleans and normalizes a text string for insertion into the vector store.

    Operations performed:
        - Converts all escaped newlines ("\\n") and different newline formats ("\r\n", "\r") to standard "\n".
        - Collapses runs of three or more newlines down to two (to preserve paragraph breaks).
        - Removes trailing spaces and tabs from each line.
        - Collapses multiple consecutive spaces within lines into a single space.
        - Trims leading and trailing whitespace from the whole string.

    Args:
        text (str): The input string to be cleaned and normalized.

    Returns:
        str: The cleaned and normalized string, suitable for vector-store ingestion.
    """
    # Normalize escaped and real newlines
    text = text.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    # Replace 3+ newlines with just 2 (keeps paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces on each line
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse multiple spaces inside lines
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
