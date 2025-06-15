# src/edgar_rag/utils.py

# This module contains utility functions for the RAG pipeline.
# It includes functions for token counting, chunk splitting, metadata parsing, HTML stripping, table extraction, and pipeline summary.

import re
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import tiktoken

def count_tokens(text, model="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_chunk_on_token_limit(text, model="text-embedding-3-small", max_tokens=8192):
    """
    Recursively split text into chunks each under max_tokens for the model.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    # Split in half and recurse
    midpoint = len(tokens) // 2
    first_half = encoding.decode(tokens[:midpoint])
    second_half = encoding.decode(tokens[midpoint:])
    return split_chunk_on_token_limit(first_half, model, max_tokens) + split_chunk_on_token_limit(second_half, model, max_tokens)

def parse_metadata(text):
    """
    Extract basic metadata from the full submission text.
    Returns a dictionary with cik, company_name, filing_date, accession_number.
    """
    def match(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None
    return {
        "cik": match(r"CENTRAL INDEX KEY:\s*(\d+)"),
        "company_name": match(r"COMPANY CONFORMED NAME:\s*([A-Za-z0-9 .,'()\-&]+)"),
        "filing_date": match(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})"),
        "accession_number": match(r"ACCESSION NUMBER:\s*([\d-]+)"),
    }

def strip_html(text: str) -> str:
    """
    Remove HTML/SGML tags and return human-readable text.
    """
    return BeautifulSoup(text, "lxml").get_text(separator="\n")

def extract_tables(text: str) -> list:
    """
    Extract HTML tables from filing text and return as list of pandas DataFrames.
    If none found, returns empty list.
    """
    # Extract all <table>...</table> segments
    soup = BeautifulSoup(text, "lxml")
    tables = []
    for tbl in soup.find_all("table"):
        try:
            # Convert each table to a string and then DataFrame
            tbl_html = str(tbl)
            dfs = pd.read_html(StringIO(tbl_html))  # <- wrap here
            for df in dfs:
                tables.append(df)
        except Exception as e:
            print(f"[WARN] Could not parse a table: {e}")
    return tables

def extract_clean_text_and_tables(text: str) -> tuple:
    """
    Cleans the raw text and extracts tables.
    Returns (clean_text, tables) tuple.
    """
    clean_text = strip_html(text)
    tables = extract_tables(text)
    return clean_text, tables

def print_metrics_block(metrics: dict):
    print("\n[Metrics Block] Retrieval Timing & Params:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

def print_pipeline_summary(state):
    print("---- PIPELINE SUMMARY ----")
    print(f"Filings loaded: {len(getattr(state, 'filings', []))}")
    print(f"Chunks created: {len(getattr(state, 'chunks', []))}")
    print(f"Chunk metrics: {getattr(state, 'chunk_metrics', {})}")
    
    if getattr(state, "embeddings", None) is not None:
        emb = state.embeddings
        if hasattr(emb, "shape"):
            print(f"Embeddings: {emb.shape}")
        else:
            try:
                print(f"Embeddings: {len(emb)}")
            except TypeError:
                print(f"Embeddings: (type: {type(emb)})")
    else:
        print("Embeddings: N/A")


    # print(f"Embeddings: {len(getattr(state, 'embeddings', [])) if hasattr(state, 'embeddings') else 'N/A'}")
    # print("Embeddings:", state.embeddings.shape if state.embeddings is not None else "N/A")
    print(f"Embedding metrics: {getattr(state, 'embedding_metrics', {}) if hasattr(state, 'embedding_metrics') else 'N/A'}")
    if hasattr(state, 'tables'):
        print(f"Tables extracted: {len(getattr(state, 'tables', []))}")
    print("-------------------------")
    # Optionally preview first chunk, first table, etc.
    if hasattr(state, 'chunks') and state.chunks:
        print("First chunk preview:")
        print(repr(state.chunks[0]['chunk'])[:400], "...\n")
    if hasattr(state, 'tables') and state.tables:
        print("First table preview:")
        print(state.tables[0].head())
'''
def format_chat_history(history, context, query, prompt_template):
    # History: list of dicts, context: string, query: string
    messages = [
        {"role": "system", "content": "You are an expert assistant for financial filings."},
        {"role": "user", "content": f"[CONTEXT]\n{context}"}  # Always re-supply latest context
    ]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    # Add the latest user query
    if prompt_template:
        user_content = prompt_template.format(query=query, context=context)
    else:
        user_content = f"Question: {query}\n\nAnswer:"
    messages.append({"role": "user", "content": user_content})
    return messages
'''

def format_chat_history(history, context, query, prompt_template=None):
    # Ignore history for now if multi-turn is not needed
    if prompt_template:
        user_prompt = prompt_template.format(context=context, query=query)
        messages = [
            {"role": "system", "content": "You are a professional financial data analyst."},
            {"role": "user", "content": user_prompt},
        ]
        return messages
    
    system_prompt = (
        "You are a professional financial data analyst. "
        "Use ONLY the context provided below to answer the user's question. "
        "If you find a direct answer, quote the relevant number, date, or text verbatim and include the surrounding sentence for clarity. "
        "If the answer is not present in the context, respond ONLY with 'Not found in the provided context.' "
    )
    user_prompt = (
        f"[CONTEXT]\n{context}\n\n"
        f"[QUESTION]\n{query}\n\n"
        f"[INSTRUCTIONS]\n"
        "- Only use information from the context. Do NOT use prior knowledge.\n"
        "- For numbers or financial metrics, quote the number and the source sentence.\n"
        "- If there are multiple relevant figures, list them all, each with its context sentence.\n"
        "- Do NOT make up or infer information not present in the context.\n"
        "- If no answer can be found, say: Not found in the provided context.\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def clean_10k_text(raw_text):
    # Remove SGML/HTML tags except tables
    soup = BeautifulSoup(raw_text, "lxml")
    # Extract tables
    tables = soup.find_all("table")
    dataframes = []
    for tbl in tables:
        try:
            # Parse each table HTML to a DataFrame
            df = pd.read_html(StringIO(str(tbl)))[0]
            dataframes.append(df)
        except Exception as e:
            print(f"[WARNING] Table parsing failed: {e}")
    # Remove tables from soup for main text cleaning...
    # Remove tables from soup (so not double-counted in text)
    for tbl in tables:
        tbl.decompose()
    # Remove all other tags, normalize whitespace
    clean_txt = soup.get_text()
    clean_txt = re.sub(r'\n{2,}', '\n', clean_txt)
    clean_txt = re.sub(r'\s{2,}', ' ', clean_txt)
    # Remove PAGE tags, headers/footers
    clean_txt = re.sub(r"<PAGE>.*", "", clean_txt)
    # Remove artifacts
    clean_txt = re.sub(r"-{5,}", "", clean_txt)
    return clean_txt, dataframes

def find_sec_item_headings(text):
    """
    Finds all SEC ITEM headings in the text, robust to inline headings,
    extra whitespace, and PART/SECTION prefixes.
    Returns a list of (item_number, section_title, match_start, match_end).
    """
    text = text.replace('\xa0', ' ')
    # Regex: ITEM (number + optional letter) [punctuation or space] [title]
    item_heading_pattern = re.compile(
        r"(ITEM\s*([0-9]+[A-Z]?)\s*[\.\-:—]?\s*([A-Z][^\n]{0,80}))",
        re.IGNORECASE
    )
    matches = []
    for match in item_heading_pattern.finditer(text):
        full_match = match.group(1)
        item_number = match.group(2)
        section_title = match.group(3).strip()
        start = match.start(1)
        end = match.end(1)
        matches.append((item_number, section_title, start, end))
    return matches

def chunk_by_sec_headings(text):
    """
    Chunks the filing text by SEC ITEM headings.
    Returns a list of dicts with 'section', 'title', 'chunk', and indices.
    """
    headings = find_sec_item_headings(text)
    chunks = []
    for idx, (item_num, sec_title, start, end) in enumerate(headings):
        # The chunk starts at the end of this heading...
        chunk_start = end
        # ...and ends at the start of the next heading (or end of doc)
        if idx + 1 < len(headings):
            chunk_end = headings[idx + 1][2]
        else:
            chunk_end = len(text)
        chunk_text = text[chunk_start:chunk_end].strip()
        if chunk_text:  # Only store non-empty chunks
            chunks.append({
                "section": item_num,
                "title": sec_title,
                "chunk": chunk_text,
                "start": chunk_start,
                "end": chunk_end
            })
    return chunks

def sub_chunk_section(text, max_chars=3000, overlap=500):
    """
    Split a section into overlapping sub-chunks of max_chars size.
    """
    chunks = []
    start = 0
    loop_count = 0
    while start < len(text):
        loop_count += 1
        if loop_count > 1000:  # Arbitrary sanity check
            print("Breaking out after 1000 iterations! Something is wrong.")
            break
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        # CRUCIAL: advance start by (max_chars - overlap)
        if end == len(text):
            break
        next_start = start + max_chars - overlap
        # Safety: Make sure we always advance
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks

def debug_print_chunks(label, chunks, n=3):
    print(f"\n=== [DEBUG] {label} (showing {min(n, len(chunks))} of {len(chunks)}) ===")
    for i, chunk in enumerate(chunks[:n]):
        chunk_text = chunk if isinstance(chunk, str) else chunk.get("chunk", "")  # Handle dict or str
        print(f"[{i}] len={len(chunk_text)}\nPreview: {chunk_text[:200]!r}\n")
    print("="*30)

def debug_print_metrics(label, metrics):
    print(f"\n=== [DEBUG] {label} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("="*30)

def debug_print_reranked(label, reranked, candidate_chunks, chunk_indices, n=5):
    print(f"\n=== [DEBUG] {label} ===")
    for i, (idx, score) in enumerate(reranked[:n]):
        # idx is index into *state.chunks*
        # chunk_indices maps position in candidate_chunks -> index in state.chunks
        try:
            candidate_idx = chunk_indices.index(idx)
            chunk = candidate_chunks[candidate_idx]
        except ValueError:
            chunk = f"[Chunk index {idx} not found in candidate_chunks]"
        print(f"[{i}] state.chunks idx={idx}, score={score:.3f}")
        print(f"Content preview: {chunk[:200]!r}")
    print("="*30)


def debug_state(state):
    print("\n=== [DEBUG] STATE DUMP ===")
    for k, v in state.__dict__.items():
        print(f"{k}: {type(v)} ({str(v)[:80]})")
    print("="*30)
