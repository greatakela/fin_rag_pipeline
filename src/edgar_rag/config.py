# src/edgar_rag/config.py
# This module contains the configuration for the RAG pipeline.


import os

# Load from .env or environment
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "fin-rag-pipeline")

# Example config for experimentation
EXPERIMENT_CONFIG = {
    "filing_type": "10-K",          # 10-Q, 10_K
    "ticker": "NVDA",               # AAPL, GOOGL, MSFT, AMZN, META, TSLA, NVDA, etc.
    "num_filings": 3,
    "chunker": "markdown_heading",  # Or: "recursive", "sentence", etc.
    "chunk_size": 1000,             # 1000
    "max_chunk_chars": 1000,        # 4000
    "chunk_overlap": 100,           # 100
    "embedder": "openai",           # Or: "sentence_transformers", etc.
    "retriever": "hybrid",          # Or: "dense", "sparse"
    "reranker": "bge",              # Or: "cohere", "voyageai"
    "llm": "openai_gpt4o",          # Or: "llama2", etc.
    "embedder": "openai",           # Or "sentence_transformers", "bge", etc.
    "embedding_model": "text-embedding-3-small",  # for OpenAI; else model name/path
    "embedding_batch_size": 32,
    "retrieval_query": "What was the aggregate voting stock value in March 2002?", # What was the aggregate voting stock value in 2002?  What were the terms of Microsoft Agreement? How much was paid in advance by Microsoft in 2000 per terms of Microsoft Agreement?
    "query_processing": "rewrite_llm",  # or "extract_keywords", "rewrite", "chain", "none", "expand"
    "llm_model_name": "gpt-4o",
    "hybrid_alpha": 0.5,
    "top_k": 10,
    "rerank_method": "cohere",          # or "bge"
    "rerank_top_k": 5,
    "use_llm_selector": True,
    "llm_selector_model": "gpt-4o",
    "llm_method": "openai",         # or "hf" for HuggingFace/local
    "llm_model": "gpt-4o",          # or your HF model, or gpt-4o
    "llm_max_tokens": 400,
    "context_limit": 10000,         # 3500
    "max_validation_iters": 1,          # 3
    "validation_llm_method": "openai",  # or "hf"
    "validation_llm_model": "gpt-4o",   # or your HF model
    "validation_llm_max_tokens": 256,
    "prompt_template": ('''
        You are a professional financial data analyst. Use ONLY the context provided below to answer the user's question. 

        --- 
        Context:
        {context}

        --- 
        Question: {query}

        Instructions:
        - If you find a direct answer, quote the relevant number or value VERBATIM, and copy the ENTIRE sentence from the context that contains it.
        - If there are multiple relevant figures, list each number and its full source sentence.
        - If no answer is present, reply EXACTLY: "Not found in the provided context."
        - Do NOT make up, infer, or summarize. Only copy sentences or values present in the context.
        - After your answer, briefly explain (in 1-2 sentences) *why* you gave this answer (e.g., 'The context contains a dollar value as of the requested date,' or 'No relevant values were found.').
        '''    
    )
    # ... add other params as needed
}
