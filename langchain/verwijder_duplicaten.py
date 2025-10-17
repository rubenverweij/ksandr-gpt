import hashlib
import json
from langchain_chroma import Chroma
from helpers import get_embedding_function

CHROMA_PATH = "/home/ubuntu/onprem_data/chroma"
OUTPUT_JSON_PATH = "/home/ubuntu/onprem_data/duplicates.json"


def hash_doc(text: str) -> str:
    """Maak een MD5-hash van de tekst (na strippen van witruimte)"""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def dedup_docs_in_chroma(
    chroma_path: str = CHROMA_PATH, output_json_path: str = OUTPUT_JSON_PATH
):
    """
    Verwijder dubbele documenten uit een Chroma-database op basis van hash van de tekst.
    Slaat metadata van duplicaten op in een JSON-bestand.
    """
    print("🔍 Start deduplicatieproces...")

    # Stap 1: Laad embedding functie
    embedding_function = get_embedding_function()

    # Stap 2: Verbind met Chroma database
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Stap 3: Haal alle documenten op
    results = db.get()

    seen_hashes = {}  # Hou bij welke hashes we al hebben gezien
    to_delete = []  # Verzamel ID's van duplicaten om te verwijderen
    hash_to_metadata = {}  # Verzamel metadata van duplicaten

    for i, (doc, doc_id, metadata) in enumerate(
        zip(results["documents"], results["ids"], results["metadatas"])
    ):
        doc_hash = hash_doc(doc)

        # Beperk metadata tot enkele relevante keys
        keys_to_extract = ["source", "chunk"]
        subset = {k: metadata[k] for k in keys_to_extract if k in metadata}
        subset["text"] = (
            doc[:20] + " ... " + doc[-20:]
        )  # Voeg korte tekstsamenvatting toe

        if doc_hash not in seen_hashes:
            seen_hashes[doc_hash] = doc_id  # Bewaar eerste document met deze hash
            hash_to_metadata[doc_hash] = [subset]
        else:
            to_delete.append(doc_id)  # Markeer duplicaat voor verwijdering
            hash_to_metadata[doc_hash].append(subset)

        if i % 1000 == 0:
            print(
                f"➡️  Gecontroleerd: {i} documenten... {len(to_delete)} duplicaten gevonden"
            )

    # Filter alleen duplicaten (meer dan 1 keer voorgekomen)
    hash_to_metadata = {k: v for k, v in hash_to_metadata.items() if len(v) >= 2}

    # Stap 4: Sla duplicaten-metadata op in JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(hash_to_metadata, f, ensure_ascii=False, indent=2)

    print(f"💾 Duplicaten-metadata opgeslagen in: {output_json_path}")

    # Stap 5: Verwijder duplicaten uit Chroma database
    if to_delete:
        print(f"🗑️  {len(to_delete)} duplicaten gevonden. Verwijderen...")
        db.delete(ids=to_delete)
    else:
        print("✅ Geen duplicaten gevonden.")

    print("✅ Deduplicatie voltooid.")


if __name__ == "__main__":
    dedup_docs_in_chroma()
