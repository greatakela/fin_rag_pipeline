# src/edgar_rag/langgraph_pipeline.py
# This module contains the LangGraph pipeline for the RAG pipeline.
# It includes functions for loading filings, chunking, embedding, and retrieval.
# It also includes functions for query processing, reranking, and LLM generation.

import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph
from langsmith.run_helpers import traceable

from src.edgar_rag.config import EXPERIMENT_CONFIG
from src.edgar_rag.data_loader import EdgarIngestor
from src.edgar_rag.chunker import Chunker, chunk_coherence, chunk_redundancy
from src.edgar_rag.utils import extract_clean_text_and_tables, print_pipeline_summary, print_metrics_block
from src.edgar_rag.embedding import embed_texts
from src.edgar_rag.utils import count_tokens, split_chunk_on_token_limit, format_chat_history
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
    bm25_store: Any = None
    query: str = ""
    query_keywords: list = field(default_factory=list)
    hybrid_results: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # List of {"user": ..., "assistant": ...}
    selected_context: List[str] = field(default_factory=list)
    llm_answer: Optional[str] = None
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
    print(f"[Node] Loaded {len(filings)} filings for {ticker} ({filing_type})")
    state.filings = filings
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

# DEBUGGGGING
#        headings = find_sec_item_headings(clean_text)
#        print(f"[DEBUG] Found {len(headings)} ITEM headings")
#        if headings:
#            for item_num, sec_title, start, end in headings:
#                print(f"  ITEM {item_num} | {sec_title} | starts at {start}")
#        
#        for line in clean_text.splitlines():
#            if "ITEM" in line.upper():
#                print("[DEBUG] ITEM candidate line:", line)
# DEBUGGGGING
        # Chunk cleaned text
         
        # Step 2: Chunk the text by SEC section headings
        section_chunks = chunk_by_sec_headings(clean_text)

        # Optionally, further split big section chunks by tokens, etc.
        # (You can add this as needed for very large sections)

        # --- Add section chunks to main chunk list ---
        '''
        for chunk in section_chunks:
            chunk['metadata'] = {
                "section": chunk["section"],
                "title": chunk["title"],
                "accession": filing.get("accession_number", ""),
                "company": filing.get("company", ""),
                "filing_type": filing.get("filing_type", "")
            }
            all_chunks.append(chunk)
        '''

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

    print(f"[Node] Created {len(all_chunks)} section-aware sub-chunks from {len(section_chunks)} SEC sections. Extracted {len(all_tables)} tables.")
    state.chunks = all_chunks
    state.tables = all_tables
    print('=================================')
    print(' ')
    # print('Chunker Module Summary:')
    # print_pipeline_summary(state)
    return state

@traceable
def chunk_eval_node(state: PipelineState, config):
    coh = chunk_coherence(state.chunks)
    red = chunk_redundancy(state.chunks)
    print(f"[Node] Chunk coherence={coh:.2f}, redundancy={red:.2f}")
    state.chunk_metrics = {"coherence": coh, "redundancy": red}
    print('=================================')
    print(' ')
    # print('Chunk Evaluation Module Summary:')
    # print_pipeline_summary(state)
    return state


@traceable
def embedding_node(state: PipelineState, config):
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
    print('=================================')
    print(' ')
    # print('Embedding Eval Module Summary:')
    # print_pipeline_summary(state)
    return state

@traceable
def hybrid_storage_node(state: PipelineState, config):
    embedding_dim = state.embeddings.shape[1]
    vec_store = FaissVectorStore(embedding_dim=embedding_dim)
    chunk_ids = [str(i) for i in range(len(state.chunks))]
    chunk_texts = [c["chunk"] for c in state.chunks]
    metadatas = state.chunks
    vec_store.add_embeddings(chunk_ids, state.embeddings, metadatas)

    bm25_store = BM25Store(chunk_texts)
    state.vector_store = vec_store
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

    print("\n=== [DEBUG] Initial Hybrid Retrieval Results ===")
    for i, (idx, score) in enumerate(hybrid_results):
        print(f"[{i}] idx={idx}, score={score:.3f}")
        print(f"Content preview: {candidate_chunks[i][:200]}\n")

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
    print("\n=== [DEBUG] After Reranking ===")
    for i, (idx, score) in enumerate(reranked):
        print(f"[{i}] idx={idx}, score={score:.3f}")
        print(f"Content preview: {candidate_chunks[idx][:200]}\n")

    reranked_indices = [chunk_indices[i] for i, _ in reranked]
    
    context_chunks = [state.chunks[i]["chunk"] for i in reranked_indices]


    # DEBUGGING

    print("\n=== [DEBUG] Final Context Chunks ===")
    for i, chunk in enumerate(context_chunks):
        print(f"[{i}] len={len(chunk)} preview={chunk[:120]!r}")

    # DEBUGGING

    # Post-processing
    context_chunks = deduplicate_chunks(context_chunks)
    context_chunks = remove_noise(context_chunks)

    print("\n=== [DEBUG] Final Selected Context ===")
    for i, chunk in enumerate(context_chunks):
        print(f"[{i}] len={len(chunk)}")
        print(f"Content preview: {chunk[:200]}\n")

    # Store for downstream
    state.selected_context = context_chunks
    print(context_chunks) #[0])
    print(f"[Retrieval+Rerank] {len(context_chunks)} final context chunks selected.")
    print('=================================')
    print(' ')
    # print('Retrieval and Reranking Module Summary:')
    # print_metrics_block(state.metrics)
    # print_pipeline_summary(state) 
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
    
    print("\n=== [DEBUG] Final Context for LLM ===")
    for i, chunk in enumerate(context_chunks):
        print(f"[{i}] len={len(chunk)}")
        print(f"Content preview: {chunk[:200]}\n")

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
    
    print(f"\n[LLM Generation Node] Generated Answer:\n{answer}\n")
    print('=================================')
    print(' ')
    # print('LLM Generation Module Summary:')
    # print_pipeline_summary(state) 
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
def answer_validation_node(state: PipelineState, config):
    query = getattr(state, "query", "")
    answer = getattr(state, "llm_answer", "")
    context_chunks = getattr(state, "selected_context", [])
    context = "\n\n".join(context_chunks)
    # Optionally keep a counter to limit iterations
    state.validation_iter = getattr(state, "validation_iter", 0) + 1
    max_iters = config.get("max_validation_iters", 3)

    val_result = validate_answer(answer, context, query, config=config)

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
    if metrics:
        print("\n[Pipeline Metrics]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    # Optionally print chat history
    history = getattr(state, "history", [])
    if history:
        print("\n[Conversation History]")
        for i, turn in enumerate(history, 1):
            print(f"{i:02d}. User: {turn['user']}\n    Assistant: {turn['assistant']}")
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

    '''
    if "final_state" in result_state:
        print_pipeline_summary(result_state["final_state"])
    else:
        print_pipeline_summary(result_state)

    print(f"Type of result_state: {type(result_state)}")
    '''
    print(result_state.keys())
    for k, v in result_state.items():
        print(f"Key: {k} | Value type: {type(v)}")

    return result_state

if __name__ == "__main__":
    run_pipeline(config=EXPERIMENT_CONFIG)
    
