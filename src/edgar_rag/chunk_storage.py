# src/edgar_rag/chunk_storage.py
# This module contains the ChunkStorage class, which is used to store and load chunked data.
# It includes functions for saving and loading chunked data.

import os
import json
import glob
from typing import List, Dict, Optional

class ChunkStorage:
    """
    Plug-and-play chunked data storage.
    Stores and loads chunk lists as JSON files.
    """

    def __init__(self, out_dir: str = "data/chunked"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _make_filename(self, company: str, filing_type: str, accession: Optional[str] = None, chunker_name: Optional[str] = None) -> str:
        acc_str = accession or "unknown"
        chunker_str = f"_{chunker_name}" if chunker_name else ""
        # Safe file name: e.g., NVDA_10-K_0001045810-23-000009_recursive.json
        filename = f"{company}_{filing_type}_{acc_str}{chunker_str}.json".replace("/", "-").replace("\\", "-")
        return os.path.join(self.out_dir, filename)

    def save(self, chunks: List[Dict], company: str, filing_type: str, accession: Optional[str] = None, chunker_name: Optional[str] = None):
        path = self._make_filename(company, filing_type, accession, chunker_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[ChunkStorage] Saved {len(chunks)} chunks to {path}")

    def load(self, company: str, filing_type: str, accession: Optional[str] = None, chunker_name: Optional[str] = None) -> List[Dict]:
        path = self._make_filename(company, filing_type, accession, chunker_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunk file {path} not found.")
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return chunks

    def load_all(self, chunker_name: Optional[str] = None) -> List[Dict]:
        # Load all chunked files (optionally filtered by chunker_name)
        pattern = f"*{f'_{chunker_name}' if chunker_name else ''}.json"
        files = glob.glob(os.path.join(self.out_dir, pattern))
        all_chunks = []
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))
        print(f"[ChunkStorage] Loaded {len(all_chunks)} chunks from {len(files)} files.")
        return all_chunks
