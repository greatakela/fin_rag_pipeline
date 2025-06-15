# SEC EDGAR RAG Pipeline

A sophisticated Retrieval-Augmented Generation (RAG) pipeline designed for analyzing SEC EDGAR filings (10-K and 10-Q reports). This project implements state-of-the-art techniques for document processing, retrieval, and question answering.

## Features

- **Document Processing**
  - SEC filing ingestion and parsing
  - Intelligent text chunking with section awareness
  - Table extraction and processing
  - Noise removal and text cleaning

- **Advanced Retrieval**
  - Hybrid retrieval combining dense (FAISS) and sparse (BM25) methods
  - Multiple embedding options (OpenAI, Sentence Transformers)
  - Configurable reranking using Cohere or BGE models
  - Advanced context window selection

- **Query Processing**
  - Query rewriting and expansion
  - Keyword extraction
  - Multi-turn conversation support
  - Configurable LLM integration

- **Quality Assurance**
  - Answer validation and hallucination detection
  - Chunk coherence and redundancy metrics
  - Embedding quality evaluation
  - Comprehensive pipeline metrics

## Prerequisites

- Python 3.8+
- OpenAI API key
- Cohere API key (optional, for reranking)
- LangSmith API key (optional, for observability)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

The pipeline is highly configurable through the `EXPERIMENT_CONFIG` in `src/edgar_rag/config.py`. Key configuration options include:

- Filing type and ticker selection
- Chunking parameters
- Embedding model selection
- Retrieval and reranking settings
- LLM model configuration

## Usage

### Basic Pipeline Execution

```bash
python -m src.edgar_rag.langgraph_pipeline
```

```python
from src.edgar_rag.langgraph_pipeline import run_pipeline
from src.edgar_rag.config import EXPERIMENT_CONFIG

# Run with default configuration
result = run_pipeline()

# Or with custom configuration
custom_config = EXPERIMENT_CONFIG.copy()
custom_config.update({
    "ticker": "AAPL",
    "filing_type": "10-K",
    "num_filings": 2
})
result = run_pipeline(config=custom_config)
```

### Query Processing

```python
from src.edgar_rag.query_processing import QueryProcessor

processor = QueryProcessor(method="rewrite_llm")
result = processor.process("What was the company's revenue in 2022?")
```

## Project Structure

```
rag-pipeline/
├── src/
│   └── edgar_rag/
│       ├── config.py           # Pipeline configuration
│       ├── data_loader.py      # SEC filing ingestion
│       ├── chunker.py          # Text chunking
│       ├── embedding.py        # Text embedding
│       ├── hybrid_storage.py   # Vector and BM25 storage
│       ├── query_processing.py # Query processing
│       ├── retrieval.py        # Document retrieval
│       ├── reranker.py         # Result reranking
│       ├── llm.py             # LLM integration
│       └── utils.py           # Utility functions
├── tests/                     # Test suite
├── data/                      # Data storage
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for embedding and LLM capabilities
- Cohere for reranking functionality
- FAISS for efficient similarity search
- LangGraph for pipeline orchestration
