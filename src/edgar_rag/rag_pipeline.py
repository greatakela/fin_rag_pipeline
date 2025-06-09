# src/edgar_rag/rag_pipeline.py
# This module contains the stand-alone RAG pipeline.
# It includes functions for query processing, retrieval, reranking, and LLM generation.

import os

from src.edgar_rag.embedding import embed_texts
from src.edgar_rag.utils import format_chat_history
from src.edgar_rag.query_processing import QueryProcessor
from src.edgar_rag.retrieval import Retriever
from src.edgar_rag.reranker import cohere_rerank, bge_rerank
from src.edgar_rag.doc_selector import advanced_context_window_selection

def full_rag_pipeline(user_query, pipeline_state, config):
    # 1. Query Processing (rewrite/expand)
    processor = QueryProcessor(method=config.get("query_processing", "none"), llm_model_name=config.get("llm_query_model"))
    proc_result = processor.process(user_query)
    processed_query = proc_result["processed_query"]

    # 2. Retrieval (Hybrid/Dense/Sparse)
    retriever = Retriever(pipeline_state.vector_store, pipeline_state.bm25_store, hybrid_alpha=config.get("hybrid_alpha", 0.5))
    query_emb = embed_texts([processed_query], config)[0]
    hybrid_results = retriever.hybrid(processed_query, query_emb, top_k=config.get("top_k", 10))
    candidate_indices = [idx for idx, _ in hybrid_results]
    candidate_chunks = [pipeline_state.chunks[idx]["chunk"] for idx in candidate_indices]

    # 3. Reranking
    rerank_method = config.get("rerank_method", "cohere")
    if rerank_method == "cohere":
        reranked = cohere_rerank(processed_query, candidate_chunks, top_k=config.get("rerank_top_k", 5))
    elif rerank_method == "bge":
        reranked = bge_rerank(processed_query, candidate_chunks, top_k=config.get("rerank_top_k", 5))
    else:
        reranked = list(enumerate([1.0]*len(candidate_chunks)))
    reranked_indices = [candidate_indices[i] for i, _ in reranked]
    reranked_chunks = [pipeline_state.chunks[i]["chunk"] for i in reranked_indices]

    # 4. Advanced Context Selection (LLM-based)
    if config.get("use_advanced_selector", True):
        context_chunks = advanced_context_window_selection(
            processed_query,
            reranked_chunks,
            max_context_chars=config.get("context_limit", 3500),
            model=config.get("llm_model", "gpt-4o-mini"),
        )
    else:
        context_chunks = reranked_chunks

    # 5. Prompt Construction (Multi-turn if enabled)
    prompt_template = config.get("prompt_template", None)
    history = getattr(pipeline_state, "history", [])
    messages = format_chat_history(history, "\n\n".join(context_chunks), processed_query, prompt_template)

    # 6. LLM Generation
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=config.get("llm_model", "gpt-4o-mini"),
        messages=messages,
        max_tokens=config.get("llm_max_tokens", 400),
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()
    pipeline_state.llm_answer = answer
    # 7. Post-processing (optional: cleanup, reformat, etc.)
    # e.g., answer = clean_answer(answer)
    print("\n[Final Answer]:\n", answer)
    # Optionally update pipeline_state.history for multi-turn
    pipeline_state.history = history + [{"user": user_query, "assistant": answer}]
    return answer, pipeline_state
