"""
Ksandr Vectorstore Configuration

Defines global constants and path mappings for use throughout
the Ksandr vectorstore ingestion and data management scripts.

Constants:
    - LOCAL_DIR_GPU_SERVER (str): Root directory for persistent storage on the GPU server.
    - DOCKER_DIR (str): Root directory used inside Docker containers.
    - INVALID_DATA_FILE_LOCATION (dict): Paths (by environment) for JSON files recording unusable/invalid document chunks.
    - VALID_PERMISSIONS (set): Permitted permission categories for document filtering.
    - CHROMA_DB_PATH (dict): Chroma vectorstore database directory (by environment).
    - DUPLICATES_DATA_PATH (dict): File locations (by environment) for duplicate metadata JSON.
    - RAW_DATA_SOURCES (dict): Document data source root directories (by environment).

Usage:
    Import these constants directly to ensure consistent file paths, directory structure,
    and category values across scripts (e.g., populate_database.py, verwijder_duplicaten.py,
    dump_onbruikbare_data.py).
"""

import os

LOCAL_DIR_GPU_SERVER = "/home/ubuntu"
DOCKER_DIR = "/root"

INVALID_DATA_FILE_LOCATION = {
    "production": {
        "server": f"{LOCAL_DIR_GPU_SERVER}/da_data/production/invalid_docs.json",
        "docker": f"{DOCKER_DIR}/da_data/production/invalid_docs.json",
    },
    "staging": {
        "server": f"{LOCAL_DIR_GPU_SERVER}/da_data/staging/invalid_docs.json",
        "docker": f"{DOCKER_DIR}/da_data/staging/invalid_docs.json",
    },
}

VALID_PERMISSIONS = {"cat-1", "cat-2"}

CHROMA_DB_PATH = {
    "production": {
        "server": f"{LOCAL_DIR_GPU_SERVER}/da_data/production/chroma",
        "docker": f"{DOCKER_DIR}/da_data/production/chroma",
    },
    "staging": {
        "server": f"{LOCAL_DIR_GPU_SERVER}/da_data/staging/chroma",
        "docker": f"{DOCKER_DIR}/da_data/staging/chroma",
    },
}

DUPLICATES_DATA_PATH = {
    "production": {
        "server": f"{LOCAL_DIR_GPU_SERVER}/da_data/production/duplicates.json",
        "docker": f"{DOCKER_DIR}/da_data/production/duplicates.json",
    },
    "staging": {
        "server": f"{LOCAL_DIR_GPU_SERVER}/da_data/staging/duplicates.json",
        "docker": f"{DOCKER_DIR}/da_data/staging/duplicates.json",
    },
}

RAW_DATA_SOURCES = {
    "production": {
        "server": f"{LOCAL_DIR_GPU_SERVER}/ksandr_files_production/",
        "docker": f"{DOCKER_DIR}/ksandr_files_production/",
    },
    "staging": {
        "server": f"{LOCAL_DIR_GPU_SERVER}/ksandr_files_staging",
        "docker": f"{DOCKER_DIR}/ksandr_files_staging",
    },
}


def running_inside_docker() -> str:
    """
    Returns True if the code is running inside a Docker container.
    """
    if os.path.exists("/.dockerenv"):
        return "docker"
    else:
        return "server"
