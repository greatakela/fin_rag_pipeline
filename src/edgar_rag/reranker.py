# src/edgar_rag/reranker.py
# This module contains the reranking functions.
# It includes functions for reranking using Cohere and BGE.
#
# Selection of a model reranking is required here, in the future change to CONFIG-defined models

import numpy as np
import os

def cohere_rerank(query, docs, top_k=5, api_key=None, model="rerank-v3.5"):
    import cohere
    co = cohere.ClientV2(api_key or os.getenv("COHERE_API_KEY"))
    resp = co.rerank(query=query, documents=docs, model=model, top_n=top_k)
    reranked = [(r.index, r.relevance_score) for r in resp.results]
    return reranked

def bge_rerank(query, docs, top_k=5, model_name="BAAI/bge-reranker-base"):
    # Using SentenceTransformers or HuggingFace pipeline
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder(model_name)
    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)
    idxs = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in idxs]

# Plug-and-play: you can add VoyageAI, local LLMs, etc.
