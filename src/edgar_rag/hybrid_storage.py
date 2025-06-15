# src/edgar_rag/hybrid_storage.py

# This module contains the HybridRetriever class, which is used to retrieve the most relevant context chunks from the vector store and BM25 store.
# It includes functions for dense, sparse, and hybrid retrieval.

import numpy as np
from rank_bm25 import BM25Okapi

class FaissVectorStore:
    def __init__(self, embedding_dim=None, index=None):
        import faiss
        if index is not None:
            self.index = index
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
        self.embedding_dim = embedding_dim
        self.ids = []
        self.metadata = []

    def add_embeddings(self, ids, embeddings, metadatas):
        embeddings = np.vstack(embeddings).astype(np.float32)
        self.index.add(embeddings)
        self.ids.extend(ids)
        self.metadata.extend(metadatas)

    def query(self, query_embedding, top_k=5):
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(query_vec, top_k)
        # D: distances, I: indices
        results = []
        for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
            if idx < 0 or idx >= len(self.ids):
                continue
            result = {
                "id": self.ids[idx],
                "distance": dist,
                "metadata": self.metadata[idx]
            }
            results.append(result)
        return results

class BM25Store:
    def __init__(self, corpus_texts):
        self.corpus = [t.split() for t in corpus_texts]
        self.bm25 = BM25Okapi(self.corpus)

    def query(self, query, top_k=5):
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append({"idx": idx, "score": scores[idx]})
        return results

class HybridRetriever:
    def __init__(self, vector_store, bm25_store, alpha=0.5):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.alpha = alpha  # fusion weight

    def query(self, query, query_embedding, top_k=5):
        # Dense retrieval
        vector_res = self.vector_store.query(query_embedding, top_k=top_k)
        dense_map = {int(item["id"]): 1 - float(item["distance"]) for item in vector_res}  # FAISS: lower distance = better, invert for score

        # Sparse retrieval
        sparse_res = self.bm25_store.query(query, top_k=top_k)

        bm25_scores = np.array([item["score"] for item in sparse_res]) if sparse_res else np.array([])

        if len(bm25_scores) > 0:
            if np.ptp(bm25_scores) != 0:
                bm25_scores = (bm25_scores - np.min(bm25_scores)) / np.ptp(bm25_scores)
            else:
                bm25_scores = np.ones_like(bm25_scores)
        else:
            bm25_scores = np.array([])

        sparse_map = {item["idx"]: bm25_scores[i] for i, item in enumerate(sparse_res)}

        # Weighted fusion
        merged = {}
        for idx, score in dense_map.items():
            merged[idx] = merged.get(idx, 0) + self.alpha * score
        for idx, score in sparse_map.items():
            merged[idx] = merged.get(idx, 0) + (1 - self.alpha) * score

        # Top results by merged score
        merged_top = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return merged_top  # List of (chunk idx, fused score)
