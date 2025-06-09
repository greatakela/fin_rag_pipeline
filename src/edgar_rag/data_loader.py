# src/edgar_rag/data_loader.py
# This module contains the EdgarIngestor class, which is used to ingest SEC filings.
# It includes functions for downloading, caching, parsing, and loading filings.
# It also includes functions for parsing metadata from filings.

from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path
from sec_edgar_downloader import Downloader
try:
    from .utils import parse_metadata
except ImportError:
    from utils import parse_metadata
import json
import pandas as pd

class EdgarIngestor:
    """
    Ingests SEC filings for a given ticker and filing type.
    Downloads, caches, parses text and metadata.
    """

    def __init__(self, cache_dir="sec-edgar-filings", email=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        email = email or os.environ.get("EDGAR_EMAIL")
        if not email:
            raise ValueError("EDGAR_EMAIL is not set in environment or passed explicitly.")
        self.dl = Downloader(email, self.cache_dir.as_posix())

    def download_filings(self, ticker, filing_type="10-K", num_filings=5, force_download=False):
        output_dir = self.cache_dir / ticker / filing_type
        self.dl.num_filings = num_filings
        if force_download and output_dir.exists():
            # Remove old subdirs to force fresh download
            for folder in output_dir.iterdir():
                if folder.is_dir():
                    for file in folder.glob("*"):
                        file.unlink()
                    folder.rmdir()
        self.dl.get(filing_type, ticker)
        # Find all full-submission.txt files, sorted by folder (accession number = date-encoded)
        files = sorted(output_dir.glob("*/full-submission.txt"), key=os.path.getmtime, reverse=True)
        # Find all full-submission.txt files, sorted by by filing date
        # files = sorted(output_dir.glob("*/full-submission.txt"), reverse=True)
        # Only keep the latest num_filings
        files = files[:num_filings]
        print(f"[DEBUG] Found {len(files)} filings in {output_dir.resolve()}")
        return files

    def load_filings(self, ticker, filing_type="10-K", num_filings=5, 
                     output_format="dict", force_download=False, drop_empty=True):
        """
        Loads and parses filings, returns data in chosen format: "dict", "json", or "df" (DataFrame)
        """
        txt_files = self.download_filings(ticker, filing_type, num_filings, force_download=force_download)
        results = []
        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if drop_empty and not text.strip():
                    print(f"[WARN] Empty file: {txt_file}")
                    continue
                meta = parse_metadata(text)
                results.append({
                    "path": str(txt_file),
                    "company": ticker,
                    "filing_type": filing_type,
                    "text": text,
                    **meta
                })
            except Exception as e:
                print(f"[ERROR] Could not read {txt_file}: {e}")
        if output_format == "dict":
            return results
        elif output_format == "json":
            return json.dumps(results, indent=2)
        elif output_format == "df":
            return pd.DataFrame(results)
        else:
            raise ValueError("output_format must be 'dict', 'json', or 'df'")

if __name__ == "__main__":
    ingestor = EdgarIngestor()
    filings = ingestor.load_filings("NVDA", "10-K", num_filings=2)
    print(f"Found {len(filings)} filings.")
    for filing in filings:
        print(f"\nLoaded {filing['filing_type']} for {filing['company']}: {filing['path']}")
        print(f"Metadata: {{'cik': {filing['cik']}, 'company_name': {filing['company_name']}, 'filing_date': {filing['filing_date']}, 'accession_number': {filing['accession_number']}}}")
        print(f"First 300 chars:\n{filing['text'][:300]}\n---")
