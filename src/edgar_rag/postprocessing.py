# src/edgar_rag/postprocessing.py
# This module contains the postprocessing functions.
# It includes functions for deduplicating chunks and removing noise from chunks.

def deduplicate_chunks(chunks):
    seen = set()
    out = []
    for chunk in chunks:
        h = hash(chunk)
        if h not in seen:
            seen.add(h)
            out.append(chunk)
    return out

def remove_noise(chunks):
    # E.g., drop chunks with too few alphanumeric characters, or only numbers
    return [c for c in chunks if sum(char.isalpha() for char in c) > 10]
