# tests/test_pipeline.py
# This module contains the test for the RAG pipeline.

import os
import pytest
import numpy as np

from src.edgar_rag.hybrid_storage import FaissVectorStore, BM25Store
from src.edgar_rag.rag_pipeline import full_rag_pipeline  # Or wherever you placed the function
from src.edgar_rag.config import EXPERIMENT_CONFIG
from src.edgar_rag.langgraph_pipeline import PipelineState

@pytest.mark.integration
def test_full_rag_pipeline():
    # Set up minimal pipeline state with necessary pre-loaded chunks, embeddings, etc.
    state = PipelineState()
    # For a real pipeline, you'll want to load or simulate the ingestion, chunking, and embedding steps first.
    # For demo, assume state is prepped (or load from a fixture).
    # (Optionally: mock embedding/vectorstore, or use real data for more robust test.)

    # Example query
    user_query = "What was NVIDIA's net income in 2022?"

    # Update experiment config if needed for testing
    config = EXPERIMENT_CONFIG.copy()
    config.update({
        "retrieval_query": user_query,
        "llm_method": "openai",         # Or "hf" for HuggingFace local model
        "llm_model": "gpt-4o-mini",          # Or your local model name
        "llm_max_tokens": 128,
        "top_k": 5,
        "rerank_method": "cohere",      # Or "bge" or None
        "use_advanced_selector": False, # To test simple flow first
    })

    # Ensure API keys are set (skip test if missing)
    if config["llm_method"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    if config["rerank_method"] == "cohere" and not os.getenv("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY not set")

    state.chunks = [{"chunk": "NVIDIA reported net income of $9.75 billion for the year 2022.", "metadata": {"year": 2022}}]
    dummy_emb = np.random.randn(1, 1536).astype(np.float32)
    state.embeddings = dummy_emb
    state.vector_store = FaissVectorStore(embedding_dim=1536)
    state.vector_store.add_embeddings(ids=["0"], embeddings=dummy_emb, metadatas=state.chunks)
    state.bm25_store = BM25Store([state.chunks[0]["chunk"]])
    state.selected_context = [state.chunks[0]["chunk"]]
    # Run the pipeline!

    answer, new_state = full_rag_pipeline(user_query, state, config)

    print("\n[Test Output] LLM Answer:", answer)
    # Assert a non-empty answer
    assert isinstance(answer, str)
    assert len(answer) > 10
    # Assert context and answer exist in state
    assert hasattr(new_state, "llm_answer")
    assert new_state.llm_answer == answer
    assert hasattr(new_state, "history")
    assert len(new_state.history) >= 1
