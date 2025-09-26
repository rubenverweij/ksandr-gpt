from langchain_huggingface import HuggingFaceEmbeddings
import torch
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentence_transformer = HuggingFaceEmbeddings(
    model_name="NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
reranker = CrossEncoder(
    "NetherlandsForensicInstitute/robbert-2023-dutch-base-cross-encoder", device=device
)


def compare_answers_with_cross_encoder(query, answer_1, answer_2):
    # Generate scores using the cross-encoder model
    scores = reranker.predict([(query, answer_1), (query, answer_2)])

    # Compare the two answers: higher score is better
    if scores[0] > scores[1]:
        best_answer = "Answer 1"
        score_difference = scores[0] - scores[1]
    else:
        best_answer = "Answer 2"
        score_difference = scores[1] - scores[0]

    return best_answer, score_difference


# Example
query = "What is the capital of France?"
answer_1 = "The capital of France is Paris."
answer_2 = "Paris is the capital city of France."

best_answer, score_diff = compare_answers_with_cross_encoder(query, answer_1, answer_2)
print(f"The better answer is {best_answer} with a score difference of {score_diff:.2f}")


def get_answer_quality(sentence_transformer, answer_1, answer_2):
    # Generate embeddings for both answers
    embeddings = sentence_transformer.encode([answer_1, answer_2])
    # Compute cosine similarity between the two embeddings
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    # Convert similarity score to a quality scale (optional, can fine-tune thresholds)
    quality_score = similarity_score * 100  # scale to percentage
    return quality_score


# Example answers to compare
answer_1 = "The capital of France is Paris."
answer_2 = "Paris is the capital city of France."

# Get the similarity score
quality_score = get_answer_quality(answer_1, answer_2)
print(f"Answer Quality Score: {quality_score}")
