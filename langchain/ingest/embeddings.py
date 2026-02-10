"""
This module initializes and provides the embedding function for the ingestion pipeline.

It is responsible for:
- Instantiating a HuggingFace Dutch sentence transformer (RobBERT-2022) as the vector embedding function.
- Configuring the device and embedding normalization for use with vectorstores such as Chroma.
- Offering a standardized `get_embedding_function` method to supply the embedding function to database
  population and deduplication scripts.

Intended usage:
- Import and call `get_embedding_function()` when loading or writing to a Chroma vector database.
- Used by ingestion scripts (e.g., `populate_database.py`, `verwijder_duplicaten.py`).
"""

import torch
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    embedding_encode_kwargs: dict = {"normalize_embeddings": True}
    device = torch.device("cpu")
    embedding_model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name="NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs,
    )
    return embeddings
