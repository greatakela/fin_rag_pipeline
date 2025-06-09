"gpt-4o" and "gpt-4o-mini" are used in multiple places:
In config.py as default LLM models
In doc_selector.py for context selection
In llm.py for answer generation
In query_processing.py for query rewriting
In validation.py for answer validation
"text-embedding-3-small" is used for embeddings:
In embedding.py as the default embedding model
In utils.py for token counting
The code is designed to be configurable, with model names being specified in the configuration rather than hardcoded. The main configuration is in config.py where the default models are set.