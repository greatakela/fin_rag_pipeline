import os
import pytest
from src.edgar_rag.data_loader import EdgarIngestor

@pytest.fixture(scope="module")
def edgar_ingestor():
    return EdgarIngestor(email=os.environ.get("EDGAR_EMAIL", "test@example.com"))

def test_load_filings(edgar_ingestor):
    filings = edgar_ingestor.load_filings("NVDA", "10-K", num_filings=1)
    assert isinstance(filings, list)
    assert len(filings) > 0
    assert "text" in filings[0]
    assert filings[0]["filing_type"] == "10-K"
    assert filings[0]["company"] == "NVDA"
    assert filings[0]["cik"] is not None
