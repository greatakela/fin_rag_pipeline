# src/edgar_rag/embedding.py
# This module contains the embedding functions.
# It includes functions for embedding text data using OpenAI and Sentence Transformers.
# It also includes a main embedding router that selects the appropriate embedding method based on the configuration.

# Contains Embedding models that will need to be CONFIG-defined 

import numpy as np

# For OpenAI embeddings
def openai_embed(texts, model="text-embedding-3-small", batch_size=32):
    import openai
    import os
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = openai.embeddings.create(input=batch, model=model)
        # OpenAI returns a list of dicts
        batch_embeds = [np.array(d.embedding) for d in resp.data]
        embeddings.extend(batch_embeds)
    return np.stack(embeddings)

# For Sentence Transformers
def sbert_embed(texts, model_name="all-MiniLM-L6-v2", batch_size=32):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)

# Dummy BGE (add your actual BGE code if needed)
def bge_embed(texts, model_name="BAAI/bge-base-en", batch_size=32):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)

# Main embedding router
def embed_texts(texts, config):
    method = config.get("embedder", "openai")
    model = config.get("embedding_model", None)
    batch_size = config.get("embedding_batch_size", 32)
    if method == "openai":
        return openai_embed(texts, model=model, batch_size=batch_size)
    elif method == "sentence_transformers":
        return sbert_embed(texts, model_name=model, batch_size=batch_size)
    elif method == "bge":
        return bge_embed(texts, model_name=model, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown embedder: {method}")
