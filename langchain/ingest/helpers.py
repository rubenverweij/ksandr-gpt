import os
import re
from typing import Dict, Union

from bs4 import BeautifulSoup


VALID_PERMISSIONS = {"cat-1", "cat-2"}


def looks_like_clean_text(text):
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
    """Extraheer type, permissie en bestandsnaam uit het pad."""
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


def clean_html(value):
    """Maak HTML-schoon en verwijder tags/lege velden."""
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
    # Normalize escaped and real newlines
    text = text.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    # Replace 3+ newlines with just 2 (keeps paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces on each line
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse multiple spaces inside lines
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
