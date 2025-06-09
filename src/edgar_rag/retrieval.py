# src/edgar_rag/retrieval.py

# This module contains the Retriever class, which is used to retrieve the most relevant context chunks from the vector store.
# It includes dense, sparse, and hybrid retrieval methods.

import numpy as np

class Retriever:
    def __init__(self, vector_store, bm25_store=None, hybrid_alpha=0.5):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.alpha = hybrid_alpha

    def dense(self, query_embedding, top_k=5):
        return self.vector_store.query(query_embedding, top_k=top_k)

    def sparse(self, query, top_k=5):
        return self.bm25_store.query(query, top_k=top_k) if self.bm25_store else []

    def hybrid(self, query, query_embedding, top_k=5):
        # Returns indices and fused scores
        dense_res = self.dense(query_embedding, top_k=top_k)
        sparse_res = self.sparse(query, top_k=top_k)
        dense_map = {int(item["id"]): 1 - float(item["distance"]) for item in dense_res}
        bm25_scores = np.array([item["score"] for item in sparse_res]) if sparse_res else np.array([])
        sparse_map = {item["idx"]: bm25_scores[i] if len(bm25_scores) > 0 else 0 for i, item in enumerate(sparse_res)}
        # Normalize and fuse
        if len(bm25_scores) > 0 and np.ptp(bm25_scores) != 0:
            norm_bm25 = (bm25_scores - np.min(bm25_scores)) / np.ptp(bm25_scores)
            for i, item in enumerate(sparse_res):
                sparse_map[item["idx"]] = norm_bm25[i]
        merged = {}
        for idx, score in dense_map.items():
            merged[idx] = merged.get(idx, 0) + self.alpha * score
        for idx, score in sparse_map.items():
            merged[idx] = merged.get(idx, 0) + (1 - self.alpha) * score
        # Top results by merged score
        merged_top = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return merged_top  # [(idx, fused_score), ...]
