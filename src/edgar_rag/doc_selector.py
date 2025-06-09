# src/edgar_rag/doc_selector.py
# This module contains functions for selecting the most relevant context chunks from a list of candidate chunks.
# It includes a simple greedy selection method and an advanced method that uses an LLM to select the most relevant chunks.
# The advanced method is used in the langgraph pipeline.
#
# LLM selection is required here, in the future change to CONFIG-defined models

import os, openai

def select_context_with_llm(query, candidate_chunks, max_context_chars=3500, model="gpt-4o-mini"):
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key) 
    prompt = (
        f"Given the following candidate context chunks from a financial report, select the set of chunks (in original order) that, "
        f"when concatenated, will best allow an LLM to answer the query below within a {max_context_chars}-character context window. "
        f"List only the indices of the selected chunks (0-based), separated by commas.\n\n"
        f"Query: {query}\n\n"
        f"Chunks:\n"
        + "\n".join([f"{i}: {chunk[:200]}..." for i, chunk in enumerate(candidate_chunks)]) +
        "\n\nSelected chunk indices:"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "Select best context chunks."}, {"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.0,
    )
    indices_str = response.choices[0].message.content.strip() 
    idxs = [int(i) for i in indices_str.split(",") if i.strip().isdigit()]
    selected_chunks = [candidate_chunks[i] for i in idxs]
    
    print("\n=== [DEBUG] Candidate Chunks ===")
    for i, chunk in enumerate(candidate_chunks):
        print(f"[{i}] {chunk[:120]!r}")
    print(f"\n[DEBUG] LLM selected indices string: {indices_str!r}")
    print(f"[DEBUG] Final chosen idxs: {idxs!r}")

    return selected_chunks

def advanced_context_window_selection(query, candidate_chunks, max_context_chars=10000, model="gpt-4o"):
    """
    Advanced context selection that considers:
    1. Semantic similarity to query
    2. Presence of temporal context matching query
    3. Presence of numerical values and dates
    4. Query term coverage
    """
    import re
    from datetime import datetime
    import openai
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Extract temporal context from query
    date_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
    query_dates = re.findall(date_pattern, query, re.IGNORECASE)
    
    # Score each chunk
    chunk_scores = []
    for i, chunk in enumerate(candidate_chunks):
        score = 0.0
        
        # 1. Check for temporal context match
        chunk_dates = re.findall(date_pattern, chunk, re.IGNORECASE)
        if query_dates and chunk_dates:
            # If query has a date, prioritize chunks with matching dates
            if any(query_date.lower() in [d.lower() for d in chunk_dates] for query_date in query_dates):
                score += 2.0
        
        # 2. Check for numerical values and dollar amounts
        dollar_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        if re.search(dollar_pattern, chunk):
            score += 1.0
            
        # 3. Check for query term coverage
        query_terms = set(query.lower().split())
        chunk_terms = set(chunk.lower().split())
        term_overlap = len(query_terms.intersection(chunk_terms))
        score += term_overlap * 0.5
        
        # 4. Get semantic similarity score from LLM
        similarity_prompt = (
            f"Rate the relevance of this chunk to the query on a scale of 0-1:\n\n"
            f"Query: {query}\n\n"
            f"Chunk: {chunk[:500]}...\n\n"
            f"Score (0-1):"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a relevance scorer for financial documents."},
                    {"role": "user", "content": similarity_prompt}
                ],
                max_tokens=10,
                temperature=0.0,
            )
            similarity_score = float(response.choices[0].message.content.strip())
            score += similarity_score * 2.0  # Weight semantic similarity heavily
        except:
            # If LLM scoring fails, continue with other scores
            pass
            
        chunk_scores.append((i, score))
    
    # Sort chunks by score
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select chunks up to max_context_chars
    selected = []
    total_len = 0
    for idx, _ in chunk_scores:
        chunk = candidate_chunks[idx]
        if total_len + len(chunk) <= max_context_chars:
            selected.append(chunk)
            total_len += len(chunk)
        else:
            break
            
    # Fallback: if nothing valid was selected, select the first chunk
    if not selected and candidate_chunks:
        selected = [candidate_chunks[0]]
        print("[Info] Fallback: selected first candidate chunk because no chunks met relevance criteria.")
        
    print(f"[advanced_context_window_selection] Selected {len(selected)} chunks with total length {total_len}")
    return selected
