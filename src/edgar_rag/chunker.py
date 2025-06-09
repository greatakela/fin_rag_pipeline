# src/edgar_rag/chunker.py
# This module contains the Chunker class, which is used to chunk text data.
# It includes functions for token-based chunking, recursive chunking, sentence window chunking, markdown heading chunking, and semantic chunking.
# It also includes functions for chunking evaluation.

#========================================================================================
# 1. NEED TO FURTHER DEFINE CHUNKER FUNCTIONS
# 2. NEED TO DEFINE METRICS
#========================================================================================

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import re
import textwrap
from typing import List, Dict, Callable, Any, Optional
from collections import Counter
from nltk.tokenize import sent_tokenize


# For SOTA chunkers, you might want: langchain, llama-index, unstructured, or similar.

class Chunker:
    """
    Modular chunker class supporting multiple SOTA chunking methods via plug-in API.
    """
    def __init__(self, method: str = "recursive", chunk_size: int = 3000, overlap: int = 500, **kwargs):
        self.method = method
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.kwargs = kwargs

        # Register methods here
        self.methods: Dict[str, Callable] = {
            "token": self.token_chunker,
            "recursive": self.recursive_chunker,
            "sentence": self.sentence_chunker,
            "markdown_heading": self.markdown_heading_chunker,
            "semantic": self.semantic_chunker_stub,  # Placeholder for SOTA semantic
            # "table": self.table_chunker_stub,  # Could use 'unstructured' or regex
        }

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk the given text using the selected method.
        Returns a list of dicts: [{"chunk": ..., "start": ..., "end": ..., "metadata": ...}, ...]
        """
        if self.method not in self.methods:
            raise ValueError(f"Chunking method {self.method} not found. Available: {list(self.methods)}")
        chunks = self.methods[self.method](text)
        # Attach metadata to each chunk if provided
        if metadata:
            for chunk in chunks:
                chunk['metadata'] = metadata
        return chunks

    # ----- Chunker Implementations -----

    def token_chunker(self, text: str) -> List[Dict]:
        """
        Simple token-based chunker (splits by word count).
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk_text = " ".join(words[i:i+self.chunk_size])
            chunks.append({"chunk": chunk_text, "start": i, "end": i+self.chunk_size})
        return chunks

    def recursive_chunker(self, text: str) -> List[Dict]:
        """
        Recursive character-based chunking with overlap (LangChain-style).
        """
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(text), step):
            chunk_text = text[i:i+self.chunk_size]
            chunks.append({"chunk": chunk_text, "start": i, "end": i+self.chunk_size})
        return chunks

    def sentence_chunker(self, text: str) -> List[Dict]:
        """
        Sentence window chunker.
        """
        sents = sent_tokenize(text)
        chunks = []
        for i in range(0, len(sents), self.chunk_size):
            chunk_text = " ".join(sents[i:i+self.chunk_size])
            chunks.append({"chunk": chunk_text, "start": i, "end": i+self.chunk_size})
        return chunks

    def markdown_heading_chunker(self, text: str) -> List[Dict]:
        """
        Chunk by markdown-style or SEC section headings (e.g., 'ITEM 1A', 'ITEM 7').
        """
    #    pattern = re.compile(r'(ITEM\s+\d+[A-Z]?\.?\s+[A-Za-z \-,]+)', re.IGNORECASE)
        pattern = re.compile(
            r"""
            ^(
        ITEM         # 'ITEM'
        \s*
        \d+          # item number
        [A-Z]?       # optional letter
        [\.\:—\-]*   # optional period(s), colon(s), dash
        \s*
        [^\n]{0,80}  # up to 80 chars (title)
            )
            """, re.IGNORECASE | re.MULTILINE | re.VERBOSE
        )
        matches = list(pattern.finditer(text))

        matches = list(pattern.finditer(text))
        if not matches:
            # fallback: recursive chunk
            return self.recursive_chunker(text)
        chunks = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()
            chunks.append({"chunk": chunk_text, "start": start, "end": end, "heading": match.group(1)})
        return chunks

    def semantic_chunker_stub(self, text: str) -> List[Dict]:
        """
        Placeholder for semantic chunker (SOTA: use embedding similarity to break at semantic boundaries).
        """
        # Implement using TextTiling, LlamaParse, or LangChain's semantic splitter.
        # Here, we fallback to recursive for demo.
        return self.recursive_chunker(text)

    # def table_chunker_stub(self, text: str) -> List[Dict]:
    #     """
    #     Placeholder for table extraction (use 'unstructured' or regex for real use).
    #     """
    #     # TODO: Implement table-specific chunking using unstructured or pandas.read_html
    #     return []

# --- Chunking evaluation utilities ---

def chunk_coherence(chunks: List[Dict]) -> float:
    """
    Dummy coherence metric: average chunk length / stddev of chunk length.
    (In practice: use embedding similarity between adjacent chunks for real SOTA metric)
    """
    import numpy as np
    lens = [len(c["chunk"]) for c in chunks]
    if not lens or np.std(lens) == 0:
        return 1.0
    return float(np.mean(lens) / np.std(lens))

def chunk_redundancy(chunks: List[Dict]) -> float:
    """
    Redundancy metric: avg Jaccard overlap of adjacent chunks.
    """
    overlaps = []
    for i in range(1, len(chunks)):
        a = set(chunks[i-1]["chunk"].split())
        b = set(chunks[i]["chunk"].split())
        if a or b:
            overlaps.append(len(a & b) / len(a | b))
    return sum(overlaps) / len(overlaps) if overlaps else 0.0
