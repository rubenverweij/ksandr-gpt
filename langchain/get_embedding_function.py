from langchain_huggingface import HuggingFaceEmbeddings
import torch


def get_embedding_function():
    embedding_encode_kwargs: dict = {"normalize_embeddings": False}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs,
    )
    return embeddings
