# src/edgar_rag/langgraph_pipeline.py
# This module contains the LangGraph pipeline for the RAG pipeline.
# It includes functions for loading filings, chunking, embedding, and retrieval.
# It also includes functions for query processing, reranking, and LLM generation.

import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph
from langsmith.run_helpers import traceable
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.edgar_rag.config import EXPERIMENT_CONFIG
from src.edgar_rag.data_loader import EdgarIngestor
from src.edgar_rag.chunker import Chunker, chunk_coherence, chunk_redundancy
from src.edgar_rag.utils import extract_clean_text_and_tables, print_pipeline_summary, print_metrics_block
from src.edgar_rag.embedding import embed_texts
from src.edgar_rag.utils import count_tokens, split_chunk_on_token_limit, format_chat_history
from src.edgar_rag.utils import debug_print_chunks, debug_print_metrics, debug_print_chunks, debug_print_reranked, debug_state
from src.edgar_rag.embedding_eval import cosine_similarity_matrix, recall_at_k, ndcg_at_k
from src.edgar_rag.hybrid_storage import FaissVectorStore, BM25Store, HybridRetriever
from src.edgar_rag.query_processing import QueryProcessor
from src.edgar_rag.retrieval import Retriever
from src.edgar_rag.reranker import cohere_rerank, bge_rerank
from src.edgar_rag.doc_selector import select_context_with_llm
from src.edgar_rag.postprocessing import deduplicate_chunks, remove_noise
from src.edgar_rag.llm import llm_generate_answer
from src.edgar_rag.doc_selector import advanced_context_window_selection
from src.edgar_rag.validation import validate_answer
from src.edgar_rag.utils import clean_10k_text, chunk_by_sec_headings, sub_chunk_section
from src.edgar_rag.faiss_persistence import save_faiss_and_metadata, load_faiss_and_metadata, maybe_load_embeddings
from src.edgar_rag.eval import chunk_coverage, retrieval_metrics, evaluate_generation_with_ragas, json_safe

# ================================================================================
#
# TODO: add imports for embedder, retriever, reranker, LLM, eval as you implement
#
#=================================================================================

# 1. Set LangSmith API keys for full observability
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "fin-rag-pipeline")

# 2. Define the pipeline state as a dataclass (no base class needed)
@dataclass
class PipelineState:
    filings: List[Dict[str, Any]] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunk_metrics: Dict[str, Any] = field(default_factory=dict)
    embeddings: Any = None
    # embeddings: List[Any] = field(default_factory=list)
    embedding_metrics: Dict[str, Any] = field(default_factory=dict)
    tables: List[Any] = field(default_factory=list)    # New: list of DataFrames
    vector_store: Any = None
    chunk_metadata: list = field(default_factory=list)  # List of dicts loaded from JSON
    bm25_store: Any = None
    query: str = ""
    query_keywords: list = field(default_factory=list)
    hybrid_results: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # List of {"user": ..., "assistant": ...}
    selected_context: List[str] = field(default_factory=list)
    llm_answer: Optional[str] = None
    # --- For evaluation ---
    eval_logs: List[Dict[str, Any]] = field(default_factory=list)
    ragas_metrics: Dict[str, Any] = field(default_factory=dict)  # One-off or per-query
    langsmith_metrics: Dict[str, Any] = field(default_factory=dict)
    gold_answers: List[Dict[str, Any]] = field(default_factory=list)  # For eval
    # Add other fields as your pipeline grows

# 3. Pipeline stage nodes with tracing
@traceable
def load_filings_node(state: PipelineState, config):
    ingestor = EdgarIngestor()
    ticker = config.get("ticker", "NVDA")
    filing_type = config.get("filing_type", "10-K")
    num_filings = config.get("num_filings", 2)
    filings = ingestor.load_filings(
        ticker,
        filing_type,
        num_filings=num_filings
    )
    print(f"[Data Loading Node] Loaded {len(filings)} filings for {ticker} ({filing_type})")
    state.filings = filings
    state.metrics['filings_loaded'] = len(state.filings)
    # Example: log missing/parse errors if applicable
    # state.metrics['filings_parse_errors'] = ...
    # Add to eval_logs:
    state.eval_logs.append({
        "stage": "loading",
        "filings_loaded": len(state.filings),
        "timestamp": time.time(),
    })
    print('=================================')
    print(' ')
    # print('Loader Module Summary:')
    # print_pipeline_summary(state)
    return state

@traceable
def chunker_node(state: PipelineState, config):
    print(f"Chunking method: {config['chunker']} size: {config['chunk_size']} overlap: {config['chunk_overlap']}")
    chunker = Chunker(method=config["chunker"], chunk_size=config["chunk_size"], overlap=config["chunk_overlap"])

    all_chunks = []
    all_tables = []
    for filing in state.filings:
        # Step 1: Clean the raw text and extract tables
        clean_text, tables = clean_10k_text(filing["text"])
        # Add tables for downstream processing
        filing["clean_text"] = clean_text
        filing["tables"] = tables
        all_tables.extend(tables)

        # Chunk cleaned text
         
        # Step 2: Chunk the text by SEC section headings
        section_chunks = chunk_by_sec_headings(clean_text)

        # Step 3: For each section, optionally sub-chunk if too large
        for sec in section_chunks:
            section_text = sec["chunk"]
            if len(section_text) > config.get("max_chunk_chars", 4000):
                sub_chunks = sub_chunk_section(
                    section_text,
                    max_chars=config.get("max_chunk_chars", 4000),
                    overlap=config.get("chunk_overlap", 500)
                )
                # Add each sub-chunk as a new chunk, with section metadata
                for i, sub in enumerate(sub_chunks):
                    all_chunks.append({
                        "section": sec["section"],
                        "title": sec["title"],
                        "chunk": sub,
                        "parent_start": sec["start"],
                        "parent_end": sec["end"],
                        "sub_idx": i,
                        "source": filing.get("source_filename", "")
                    })
            else:
                # Section is small enough; add as-is
                sec["source"] = filing.get("source_filename", "")
                all_chunks.append(sec)
    # debug_print_chunks("Section-Aware Chunks (Post Chunker)", all_chunks, n=5)
    print(f"[Chunker Node] Created {len(all_chunks)} section-aware sub-chunks from {len(section_chunks)} SEC sections. Extracted {len(all_tables)} tables.")
    state.chunks = all_chunks
    state.tables = all_tables
    state.metrics['chunks_created'] = len(all_chunks)
    state.metrics['avg_chunk_size'] = np.mean([len(c['chunk']) for c in all_chunks]) if all_chunks else 0
    # Add to eval_logs:
    state.eval_logs.append({
        "stage": "chunking",
        "chunks_created": len(all_chunks),
        "avg_chunk_size": np.mean([len(c['chunk']) for c in all_chunks]) if all_chunks else 0,
        "timestamp": time.time(),
    })
    print('=================================')
    print(' ')
    # print('Chunker Module Summary:')
    # print_pipeline_summary(state)
    return state

@traceable
def chunk_eval_node(state: PipelineState, config):
    coh = chunk_coherence(state.chunks)
    red = chunk_redundancy(state.chunks)
    print(f"[Chunker Eval Node] Chunk coherence={coh:.2f}, redundancy={red:.2f}")
    state.chunk_metrics = {"coherence": coh, "redundancy": red}
    coverage = chunk_coverage(state.filings, state.chunks)
    state.chunk_metrics['coverage'] = coverage
    debug_print_metrics("Chunker Metrics", state.chunk_metrics)
    print('=================================')
    print(' ')
    # print('Chunk Evaluation Module Summary:')
    # print_pipeline_summary(state)
    return state


@traceable
def embedding_node(state: PipelineState, config):
    faiss_path = config.get("faiss_path", "data/fin_rag_faiss.index")
    metadata_path = config.get("metadata_path", "data/fin_rag_metadata.json")
    embeddings_path = config.get("embeddings_path", "data/fin_rag_embeddings.npy")
    
    # Try loading
    index, metadata, embeddings = maybe_load_embeddings(faiss_path, metadata_path, embeddings_path)
    if index is not None and metadata is not None and embeddings is not None:
        print(f"[Embedding Node] Loaded FAISS, metadata, embeddings from disk.")
        state.embeddings = embeddings
        state.chunk_metadata = metadata
        # Re-wrap index in your FaissVectorStore if needed
        embedding_dim = embeddings.shape[1]
        faiss_store = FaissVectorStore(embedding_dim=embedding_dim, index=index)
        state.vector_store = faiss_store
        return state  # SKIP rest of embedding!

    MAX_TOKENS = 8192  # For OpenAI text-embedding-3-small
    filtered_texts = []
    for chunk in state.chunks:
        chunk_text = chunk['chunk']
        tokens = count_tokens(chunk_text, model=config.get("embedding_model", "text-embedding-3-small"))
        if tokens <= MAX_TOKENS:
            filtered_texts.append(chunk_text)
        else:
            print(f"[Embedding Node] Chunk too large ({tokens} tokens). Splitting.")
            split_chunks = split_chunk_on_token_limit(chunk_text, model=config.get("embedding_model", "text-embedding-3-small"), max_tokens=MAX_TOKENS)
            print(f"[Embedding Node] Split into {len(split_chunks)} chunks.")
            filtered_texts.extend(split_chunks)
    if not filtered_texts:
        print("[Embedding Node] No chunks to embed! Check chunking and cleaning logic.")
        state.embeddings = None
        return state
    
    print(f"[Embedding Node] Embedding {len(filtered_texts)} chunks using {config['embedder']} ({config.get('embedding_model')})...")
    embeddings = embed_texts(filtered_texts, config)
    state.embeddings = embeddings
    state.chunk_metadata = state.chunks  # Or add any additional info you want in metadata

    # Now persist (save) everything so next run can reuse!
    t0 = time.time()
    faiss_store = FaissVectorStore(embedding_dim=embeddings.shape[1])
    chunk_ids = [str(i) for i in range(len(state.chunks))]
    faiss_store.add_embeddings(chunk_ids, embeddings, state.chunk_metadata)
    # Save FAISS, metadata, and embeddings to disk
    save_faiss_and_metadata(faiss_store, state.chunk_metadata, faiss_path, metadata_path)
    np.save(embeddings_path, embeddings)
    state.vector_store = faiss_store
    t1 = time.time()
    print(f"[Embedding Node] Saved FAISS, metadata, and embeddings to disk.")

    state.metrics['storage_latency'] = t1 - t0
    state.metrics['embedding_shape'] = state.embeddings.shape if state.embeddings is not None else None
    state.eval_logs.append({
        "stage": "embedding",
        "n_chunks": len(filtered_texts),
        "embedding_shape": str(getattr(state.embeddings, 'shape', None)),
        "timestamp": time.time(),
    })
    state.eval_logs.append({    
        "stage": "storage_latency",
        "latency": t1 - t0,
        "timestamp": t1,
    })
    print(f"[Embedding Node] Embedding shape: {embeddings.shape}")
    print('=================================')
    print(' ')
    # print('Embedding Module Summary:')
    # print_pipeline_summary(state)
    return state

@traceable
def embedding_eval_node(state: PipelineState, config):
    # Fake queries: for demo, use a few chunk embeddings as queries (real use: load/query/label pairs)
    if state.embeddings is None or len(state.embeddings) == 0:
        print("[Embedding Eval Node] No embeddings to evaluate.")
        state.embedding_metrics = {}
        return state

    embeddings = state.embeddings
    num_queries = min(5, embeddings.shape[0])
    sim_matrix = cosine_similarity_matrix(embeddings[:num_queries], embeddings)

    # Fake gold: for each "query", the relevant chunk is just itself
    gold = [[i] for i in range(num_queries)]

    recall = recall_at_k(sim_matrix, gold, k=5)
    ndcg = ndcg_at_k(sim_matrix, gold, k=10)

    metrics = {
        "recall@5": recall,
        "ndcg@10": ndcg,
    }
    print(f"[Embedding Eval Node] recall@5={recall:.2f}, ndcg@10={ndcg:.2f}")
    state.embedding_metrics = metrics
    state.ragas_metrics['embedding'] = metrics
    state.eval_logs.append({
        "stage": "embedding_eval",
        "recall@5": recall,
        "ndcg@10": ndcg,
        "timestamp": time.time(),
    })
    print('=================================')
    print(' ')
    # print('Embedding Eval Module Summary:')
    # print_pipeline_summary(state)
    return state

@traceable
def hybrid_storage_node(state: PipelineState, config):
    chunk_texts = [c["chunk"] for c in state.chunks]
    bm25_store = BM25Store(chunk_texts)
    state.bm25_store = bm25_store
    print("[Hybrid Storage Node] FAISS vector store and BM25 store ready.")
    print('=================================')
    print(' ')
    # print('Hybrid Storage Module Summary:')
    # print_pipeline_summary(state)
    return state

#===========================================================================
#    will need to work on the interface for queries - streamlit chat?
#===========================================================================

@traceable
def query_processing_node(state: PipelineState, config):
    method = config.get("query_processing", "none")
    llm_model_name = config.get("llm_model_name", "gpt-4o-mini")
    query = config.get("retrieval_query", "What was the aggregate voting stock value in 2002?")
    processor = QueryProcessor(method=method, llm_model_name=llm_model_name)
    result = processor.process(query)
    processed_query = result["processed_query"]
    print(f"[Query Processing] {method}: '{query}' -> '{processed_query}'")
    # Store for downstream nodes
    state.query = processed_query
    if "keywords" in result:
        state.query_keywords = result["keywords"]
        print(f"  Extracted keywords: {result['keywords']}")
    print('=================================')
    print(' ')
    # print('Query Processing Module Summary:')
    # print_pipeline_summary(state)    
    return state

@traceable
def hybrid_retrieval_node(state: PipelineState, config):
    query = getattr(state, "query", config.get("retrieval_query", "What was the aggregate voting stock value in 2002?"))
    alpha = config.get("hybrid_alpha", 0.5)
    top_k = config.get("top_k", 5)

    print(f"\n[HybridRetrieval] Query: '{query}' | top_k={top_k} | alpha={alpha}")

    # Compute query embedding
    t0 = time.time()
    query_emb = embed_texts([query], config)[0]
    embedding_time = time.time() - t0
    print(f"[HybridRetrieval] Query embedding time: {embedding_time:.4f}s")

    # Hybrid retrieval
    retriever = HybridRetriever(state.vector_store, state.bm25_store, alpha=alpha)

    t0 = time.time()
    hybrid_results = retriever.query(query, query_emb, top_k=top_k)
    hybrid_time = time.time() - t0
    
    # Display results (ids and scores)
    print(f"[HybridRetrieval] Top {top_k} fused results (chunk_idx, score):")
    for idx, score in hybrid_results:
        print(f"  {idx:>3} | {score:.4f}")
        # Optionally print chunk preview:
        chunk_txt = state.chunks[idx]["chunk"]
        print("      Chunk size:", len(chunk_txt))
        print("      Preview:", repr(chunk_txt[:150]), "\n")

    # Store for downstream stages/metrics
    state.hybrid_results = hybrid_results

    # Store metrics
    if not hasattr(state, "metrics") or state.metrics is None:
        state.metrics = {}
    state.metrics.update({
        "query_embedding_time": embedding_time,
        "hybrid_retrieval_time": hybrid_time,
        "retrieval_top_k": top_k,
        "retrieval_alpha": alpha,
        "retrieval_query": query,
    })
    print(f"[HybridRetrieval] Timing: embedding={embedding_time:.4f}s, hybrid={hybrid_time:.4f}s")
    print('=================================')
    print(' ')
    # print('Hybrid Retrieval Module Summary:')
    # print_metrics_block(state.metrics)
    # print_pipeline_summary(state) 
    return state
    # return {"final_state": state}

@traceable
def retrieval_and_rerank_node(state: PipelineState, config):
    # print(">>> ENTRY", __name__, "id(state):", id(state), "state.selected_context:", getattr(state, "selected_context", None))
    # Use processed query
    query = getattr(state, "query", config.get("retrieval_query", ""))
    query_emb = embed_texts([query], config)[0]
    retriever = Retriever(state.vector_store, state.bm25_store, hybrid_alpha=config.get("hybrid_alpha", 0.5))
    top_k = config.get("top_k", 10)
    
    # Get hybrid retrieval candidates
    hybrid_results = retriever.hybrid(query, query_emb, top_k=top_k)
    chunk_indices = [idx for idx, _ in hybrid_results]
    candidate_chunks = [state.chunks[idx]["chunk"] for idx in chunk_indices]

    # debug_print_reranked("Initial Hybrid Retrieval Results", hybrid_results, candidate_chunks, chunk_indices, top_k)

    # Reranking (choose model based on config)
    rerank_method = config.get("rerank_method", "cohere")  # or "bge"
    if rerank_method == "cohere":
        api_key = os.getenv("COHERE_API_KEY")
        reranked = cohere_rerank(query, candidate_chunks, top_k=config.get("rerank_top_k", 5), api_key=api_key) # returns [(idx, score), ...]
    elif rerank_method == "bge":
        reranked = bge_rerank(query, candidate_chunks, top_k=config.get("rerank_top_k", 5)) # returns [(idx, score), ...]
    else:
        reranked = list(enumerate([1.0] * len(candidate_chunks)))  # no reranking

    '''    
    # Always sort by score descending
    reranked_sorted = sorted(reranked, key=lambda x: x[1], reverse=True)
    
    # Pick top N indices
    reranked_indices = [chunk_indices[idx] for idx, _ in reranked_sorted[:config.get("rerank_top_k", 5)]]
    context_chunks = [state.chunks[i]["chunk"] for i in reranked_indices]

    # DEBUGGING
    # Suppose reranked = [(idx, score), ...]
    print("\n=== [DEBUG] Reranked Chunks and Scores ===")
    for i, (orig_idx, score) in enumerate(reranked_sorted[:5]):
        print(f"[{i}] idx={orig_idx}, score={score:.3f}, preview={candidate_chunks[orig_idx][:120]!r}")

    print("=== [DEBUG] Candidate Chunks after Retrieval/Rerank ===")
    for i, chunk in enumerate(candidate_chunks):
        print(f"[{i}] len={len(chunk)}\n{chunk[:350]}\n")
    print("=== [DEBUG] END of DEBUG ===")
    '''

    # DEBUGGING

    # Agentic doc selector (LLM-based, optional)
    # if config.get("use_llm_selector", False):
    #     context_chunks = select_context_with_llm(query, context_chunks, max_context_chars=3500, model=config.get("llm_selector_model", "gpt-4o-mini"))

    # context_chunks = candidate_chunks[:5]
    
    # debug_print_reranked("After Reranking", reranked, candidate_chunks, chunk_indices, top_k)

    reranked_indices = [chunk_indices[i] for i, _ in reranked]
    context_chunks = [state.chunks[i]["chunk"] for i in reranked_indices]

    # debug_print_reranked("Final Context Chunks", reranked, context_chunks, reranked_indices, top_k)

    # Post-processing
    context_chunks = deduplicate_chunks(context_chunks)
    context_chunks = remove_noise(context_chunks)

    # debug_print_reranked("Final Selected Context", reranked, state.chunks, chunk_indices, top_k)
    # context_chunks = [state.chunks[idx]['chunk'] for idx, _ in reranked[:top_k]]

    print("=== [DEBUG] Final Selected Context ===")
    for i, (idx, score) in enumerate(reranked[:top_k]):
        chunk = context_chunks[i]
        print(f"[{i}] state.chunks idx={idx}, score={score:.3f}")
        print(f"Content preview: {chunk!r}")
    
    # Calculate top-k recall for a labeled QA pair, if available
    # (You'd need gold indices for real eval)
    # Example:
    if hasattr(state, "gold_answers"):
        metrics = retrieval_metrics(state.query, context_chunks, state.gold_answers)
        state.ragas_metrics['retrieval'] = metrics
    
    # state.ragas_metrics['retrieval'] = {
        # "retrieved_indices": reranked_indices,
        # ...add recall@k, ndcg, etc if available
    # }
    state.eval_logs.append({
        "stage": "retrieval_rerank",
        "retrieved_indices": reranked_indices,
        "timestamp": time.time(),
    })
    # Store for downstream
    state.selected_context = context_chunks
    # print(context_chunks) #[0])

    print(f"[Retrieval+Rerank] {len(context_chunks)} final context chunks selected.")
    print('=================================')
    print(' ')
    # print('Retrieval and Reranking Module Summary:')
    # print_metrics_block(state.metrics)
    # print_pipeline_summary(state) 
    # debug_state(state)
    # print(">>> retrieval_and_rerank_node sets selected_context with", len(state.selected_context), "chunks")
    '''print("=== STATE DUMP ===")
    for k in state.__dict__:
        v = getattr(state, k)
        if isinstance(v, list):
            print(f"{k}: list with {len(v)} items")
        elif isinstance(v, dict):
            print(f"{k}: dict with {len(v)} keys")
        else:
            print(f"{k}: {type(v)}")'''

    return state

@traceable
def llm_generation_node(state: PipelineState, config):
    query = getattr(state, "query", config.get("retrieval_query", ""))
    candidate_chunks = getattr(state, "selected_context", [])
    
    print("\n=== [DEBUG] LLM Generation Input ===")
    print(f"Query: {query}")
    print(f"Number of candidate chunks: {len(candidate_chunks)}")
    
    if config.get("use_advanced_selector", True):
        print("\nUsing advanced context selection...")
        context_chunks = advanced_context_window_selection(
            query, candidate_chunks, 
            max_context_chars=config.get("context_limit", 8000),
            model=config.get("llm_model", "gpt-4o")
        )
        print(f"After advanced selection: {len(context_chunks)} chunks")
        
        # If advanced selection returns too few chunks or seems to have selected irrelevant content,
        # fall back to using the top N chunks from retrieval/reranking
        if len(context_chunks) < 2 or not any(query.lower() in chunk.lower() for chunk in context_chunks):
            print("\nAdvanced selection may have missed relevant content. Falling back to top chunks...")
            context_chunks = candidate_chunks[:3]  # Use top 3 chunks from retrieval/reranking
            print(f"Using top {len(context_chunks)} chunks from retrieval")
    else:
        context_chunks = candidate_chunks
        print("\nUsing all candidate chunks without advanced selection")
    
    # debug_print_chunks("Final Context for LLLM", context_chunks, len(context_chunks))

    context = "\n\n".join(context_chunks)[:config.get("context_limit", 8000)]
    prompt_template = config.get("prompt_template", None)
    history = getattr(state, "history", [])
    messages = format_chat_history(history, context, query, prompt_template)

    # DEBUGGING
    print("===== CONTEXT SENT TO LLM =====")
    print(context[:2000] + ("..." if len(context) > 2000 else ""))  # show first 2000 chars    
    print("\n=== [DEBUG] Final Context Length ===")
    print(f"Total context length: {len(context)} chars")
    print(f"Context limit: {config.get('context_limit', 8000)} chars")
    print(f"Truncated: {len(context) > config.get('context_limit', 8000)}")
    print("===== END CONTEXT =====")
    # DEBUGGING
 
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=config.get("llm_model", "gpt-4o"),
        messages=messages,
        max_tokens=config.get("llm_max_tokens", 400),
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()
    state.llm_answer = answer
    # Append to history
    state.history = history + [{"user": query, "assistant": answer}]
    
    # After answer generated:
    # Call RAGAS (or your validation code) here:
    # e.g., ragas_metrics = ragas.evaluate(...)
    # state.ragas_metrics['llm_generation'] = ragas_metrics
    ragas_scores = evaluate_generation_with_ragas(state.llm_answer, "\n\n".join(context_chunks), state.query)
    state.ragas_metrics['generation'] = ragas_scores
    state.eval_logs.append({
        "stage": "generation_eval",
        **ragas_scores,
        "timestamp": time.time(),
    })
    
    state.eval_logs.append({
        "stage": "generation",
        "llm_answer": state.llm_answer,
        "timestamp": time.time(),
    })

    print(f"\n[LLM Generation Node] Generated Answer:\n{answer}\n")
    print('=================================')
    print(' ')
    # print('LLM Generation Module Summary:')
    # print_pipeline_summary(state) 
    return state

@traceable
def answer_validation_node(state: PipelineState, config):
    query = getattr(state, "query", "")
    answer = getattr(state, "llm_answer", "")
    context_chunks = getattr(state, "selected_context", [])
    context = "\n\n".join(context_chunks)
    # Optionally keep a counter to limit iterations
    state.validation_iter = getattr(state, "validation_iter", 0) + 1
    max_iters = config.get("max_validation_iters", 3)

    val_result = validate_answer(answer, context, query, config=config)

    state.eval_logs.append({
        "stage": "validation",
        "answer": answer,
        "validation_result": val_result,
        "timestamp": time.time(),
    })

    print("[Answer Validation]")
    print("Hallucination:", val_result["hallucination"])
    print("Grounded:", val_result["grounded"])
    print("Answers Question:", val_result["answers_question"])
    print("Explanation:", val_result.get("explanation", ""))

    # Routing
    if val_result["hallucination"].lower() == "yes":
        state.validation_status = "hallucination"
        if state.validation_iter < max_iters:
            return {"__output__": "regenerate"}
        else:
            state.final_status = "max_iterations"
            return {"__output__": "stop"}
    elif val_result["grounded"].lower() != "yes":
        state.validation_status = "not_grounded"
        if state.validation_iter < max_iters:
            return {"__output__": "rewrite_query"}
        else:
            state.final_status = "max_iterations"
            return {"__output__": "stop"}
    elif val_result["answers_question"].lower() != "yes":
        state.validation_status = "not_relevant"
        if state.validation_iter < max_iters:
            return {"__output__": "rewrite_query"}
        else:
            state.final_status = "max_iterations"
            return {"__output__": "stop"}
    else:
        state.validation_status = "success"
        state.final_status = "success"
        return {"__output__": "final_answer"}

@traceable
def finish_node(state: "PipelineState", config):
    """
    Final node: logs or returns the final answer, summaries, and metrics.
    You can extend this to write to disk, send to a UI, etc.
    """
    answer = getattr(state, "llm_answer", "[NO ANSWER]")
    query = getattr(state, "query", "")
    status = getattr(state, "final_status", "completed")
    print('\n===== FINAL ANSWER =====')
    print(f"Query: {query}")
    print(f"Final Answer: {answer}")
    print(f"Status: {status}")
    # Optionally print key metrics or history
    metrics = getattr(state, "metrics", {})
    debug_print_metrics("Pipeline Metrics", metrics)
    # Optionally print chat history
    history = getattr(state, "history", [])
    if history:
        print("\n[Conversation History]")
        for i, turn in enumerate(history, 1):
            print(f"{i:02d}. User: {turn['user']}\n    Assistant: {turn['assistant']}")

    import json
    with open("data/eval_logs.json", "w") as f:
        json.dump(json_safe(state.eval_logs), f, indent=2)
    with open("data/ragas_metrics.json", "w") as f:
        json.dump(state.ragas_metrics, f, indent=2)

    # Return state so LangGraph finishes cleanly
    return state


# --- Add more nodes for embedding, embedding_eval, storage, etc. as you build them ---

# 4. Build the pipeline graph
def build_pipeline_graph(config):
    graph = StateGraph(PipelineState)
    graph.add_node("LoadFilings", lambda s: load_filings_node(s, config))
    graph.add_node("Chunking", lambda s: chunker_node(s, config))
    graph.add_node("ChunkEval", lambda s: chunk_eval_node(s, config))
    graph.add_node("Embedding", lambda s: embedding_node(s, config))
    graph.add_node("EmbeddingEval", lambda s: embedding_eval_node(s, config))
    graph.add_node("HybridStorage", lambda s: hybrid_storage_node(s, config))
    graph.add_node("QueryProcessing", lambda s: query_processing_node(s, config))
    graph.add_node("HybridRetrieval", lambda s: hybrid_retrieval_node(s, config))
    graph.add_node("RetrievalAndRerank", lambda s: retrieval_and_rerank_node(s, config))
    graph.add_node("LLMGeneration", lambda s: llm_generation_node(s, config))
    graph.add_node("AnswerValidation", lambda s: answer_validation_node(s, config))
    graph.add_node("Finish", lambda s: finish_node(s, config))  # or a function that logs/returns result

    # add here as you add more stages...

    # TODO: Add more stages as you build
    graph.set_entry_point("LoadFilings")
    graph.add_edge("LoadFilings", "Chunking")
    graph.add_edge("Chunking", "ChunkEval")
    graph.add_edge("ChunkEval", "Embedding")
    graph.add_edge("Embedding", "EmbeddingEval")
    graph.add_edge("EmbeddingEval", "HybridStorage")
    graph.add_edge("HybridStorage", "QueryProcessing")
    graph.add_edge("QueryProcessing", "HybridRetrieval")
    graph.add_edge("HybridRetrieval", "RetrievalAndRerank")
    graph.add_edge("RetrievalAndRerank", "LLMGeneration")
    graph.add_edge("LLMGeneration", "AnswerValidation")
    graph.add_conditional_edges(
        "AnswerValidation",
        {
            "regenerate": lambda s: llm_generation_node(s, config),
            "rewrite_query": lambda s: query_processing_node(s, config),
            "final_answer": lambda s: finish_node(s, config),
            "stop": lambda s: finish_node(s, config),
        }
    )
    # add here as you add more stages...

    return graph

# 5. Main runner

def run_pipeline(config=EXPERIMENT_CONFIG):
    graph = build_pipeline_graph(config)
    state = PipelineState()
    print("Running pipeline with LangGraph + LangSmith tracing...")
    # This will trace all @traceable nodes in LangSmith automatically!
    compiled = graph.compile()
    result_state = compiled.invoke(state)
    print('=================================')
    print(' ')
    print("Pipeline complete. Results:")

    # print(result_state.keys())
    for k, v in result_state.items():
        print(f"Key: {k} | Value type: {type(v)}")

    return result_state

if __name__ == "__main__":
    run_pipeline(config=EXPERIMENT_CONFIG)
    
