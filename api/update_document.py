from onprem import LLM
from langchain_core.documents import Document


def update_document(page_content: str, metadata: dict, doc_id: str):
    """
    Updates a document in the vector database.

    Args:
        page_content (str): The new content for the document.
        metadata (dict): The metadata associated with the document.
        doc_id (str): The ID of the document to be updated.

    Returns:
        None
    """
    try:
        # Initialize and load the vector database
        llm = LLM(n_gpu_layers=-1, embedding_model_kwargs={"device": "cuda"})
        vector_db = llm.load_vectordb()

        # Get all document IDs from the database
        document_ids = vector_db.get()["ids"]

        if doc_id not in document_ids:
            print(f"Document with ID {doc_id} not found in the database.")
            return

        # Prepare the updated document
        updated_document = Document(
            page_content=page_content, metadata=metadata, id=doc_id
        )

        # Update the document in the vector database
        vector_db.update_document(document_id=doc_id, document=updated_document)
        print(f"Document with ID {doc_id} successfully updated.")

    except Exception as e:
        print(f"An error occurred while updating the document: {e}")
