"""
Configuration settings and constants for the Ksandr vectorstore pipeline.

This module defines shared paths, valid permission categories, and
other global options for ingestion and database processing scripts.

Constants:
    LOCAL_DIR (str): Base local directory for data.
    INVALID_DATA_LOCATION (str): Path to file containing unusable text examples.
    VALID_PERMISSIONS (set): Accepted permission categories for tracked documents.
    CHROMA_PATH (str): Default persistent directory for Chroma vector store.
    OUTPUT_JSON_PATH (str): Output file path for duplicate document metadata.

Intended usage:
    Import these constants in scripts such as `populate_database.py`,
    `verwijder_duplicaten.py`, and `dump_onbruikbare_data.py` for centralized configuration.
"""

LOCAL_DIR_GPU_SERVER = "/home/ubuntu"
INVALID_DATA_LOCATION = "onprem_data/voorbeelden_onbruikbare_teksten.json"
VALID_PERMISSIONS = {"cat-1", "cat-2"}
CHROMA_PATH = "/home/ubuntu/onprem_data/chroma"
OUTPUT_JSON_PATH = "/home/ubuntu/onprem_data/duplicates.json"
