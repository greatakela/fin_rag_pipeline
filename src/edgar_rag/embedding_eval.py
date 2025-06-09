# src/edgar_rag/embedding_eval.py

# This module contains the embedding evaluation functions.
# It includes functions for recall at k, NDCG at k, and placeholder for mean pairwise similarity.

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def embedding_recall_at_k(embeddings, queries, references, k=5):
    """
    embeddings: np.array, shape [n_chunks, d]
    queries: np.array, shape [n_queries, d]
    references: list of list, each sublist contains indices of relevant chunks for the query
    """
    scores = cosine_similarity(queries, embeddings)
    recall_scores = []
    for i, rel in enumerate(references):
        topk = np.argsort(scores[i])[::-1][:k]
        hit = any(idx in topk for idx in rel)
        recall_scores.append(int(hit))
    return np.mean(recall_scores)

# Add NDCG@k, etc. as needed
def cosine_similarity_matrix(x, y=None):
    """
    Compute cosine similarity between x and y.
    If y is None, computes pairwise within x (square matrix).
    """
    return cosine_similarity(x, y)

def recall_at_k(sim_matrix, gold_indices, k=5):
    """
    sim_matrix: [n_queries, n_chunks]
    gold_indices: list of lists, gold_indices[i] = [indices of relevant docs for query i]
    """
    recalls = []
    for i, rel in enumerate(gold_indices):
        top_k = np.argsort(sim_matrix[i])[::-1][:k]
        recall = any(idx in top_k for idx in rel)
        recalls.append(int(recall))
    return float(np.mean(recalls))

def ndcg_at_k(sim_matrix, gold_indices, k=10):
    """Compute NDCG@k for each query."""
    def dcg(rels):
        return sum(rel / np.log2(i+2) for i, rel in enumerate(rels))
    scores = []
    for i, rels in enumerate(gold_indices):
        scores_for_query = np.zeros(sim_matrix.shape[1])
        scores_for_query[rels] = 1
        top_k_idx = np.argsort(sim_matrix[i])[::-1][:k]
        gains = scores_for_query[top_k_idx]
        ideal = np.sort(scores_for_query)[::-1][:k]
        ndcg = dcg(gains) / (dcg(ideal) or 1)
        scores.append(ndcg)
    return float(np.mean(scores))

# Placeholder for pairwise annotated evaluation if needed
def mean_pairwise_similarity(embeddings, pairs):
    sims = []
    for i, j in pairs:
        sims.append(float(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]))
    return float(np.mean(sims)) if sims else 0.0