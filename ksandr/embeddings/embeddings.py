"""
This module provides the embedding function for use within graph database-related workflows.

Features:
- Sets up the HuggingFace Dutch sentence transformer (RobBERT-2022) for use as a vector embedding function.
- Configures the device (defaults to GPU if available) and ensures embedding normalization.
- Supplies `get_embedding_function`, which returns an embedding function compatible with vectorstores like Chroma.

Typical usage:
    - Import and call `get_embedding_function()` when creating or querying Chroma or other vector databases in the graph DB context.
    - Used by scripts and pipelines under the graphdb package that require robust Dutch sentence embeddings.
"""

import torch
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    embedding_encode_kwargs: dict = {"normalize_embeddings": True}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name="NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs,
    )
    return embeddings
