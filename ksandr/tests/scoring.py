from langchain_huggingface import HuggingFaceEmbeddings
import torch
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTENCE_TRANSFORMERS = {
    "robbert-2022": HuggingFaceEmbeddings(
        model_name="NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    ),
    "mini-lm-l6": HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    ),
}


RERANKER = CrossEncoder(
    "NetherlandsForensicInstitute/robbert-2023-dutch-base-cross-encoder", device=DEVICE
)


def compare_answers_with_cross_encoder(query, answer_1, answer_2):
    # Generate scores using the cross-encoder model
    scores = RERANKER.predict([(query, answer_1), (query, answer_2)])
    # Compare the two answers: higher score is better
    if scores[0] > scores[1]:
        best_answer = "referentie"
        score_difference = scores[0] - scores[1]
    else:
        best_answer = "nieuw"
        score_difference = scores[1] - scores[0]
    return best_answer, score_difference, scores


def get_answer_quality(transformer, answer_1, answer_2):
    # Generate embeddings for both answers
    embeddings_1 = transformer.embed_query(answer_1)
    embeddings_2 = transformer.embed_query(answer_2)
    # Compute cosine similarity between the two embeddings
    similarity_score = cosine_similarity([embeddings_1], [embeddings_2])[0][0]
    # Convert similarity score to a quality scale (optional, can fine-tune thresholds)
    quality_score = similarity_score * 100  # scale to percentage
    return round(float(quality_score), 1)


def tfidf_cosine_sim(s1, s2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([s1, s2])
    return round(float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]), 2)
