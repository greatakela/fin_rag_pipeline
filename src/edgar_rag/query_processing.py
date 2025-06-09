# src/edgar_rag/query_processing.py

# This module contains the QueryProcessor class, which is used to process the user query.
# It includes functions for rewriting the query using an LLM, expanding the query, extracting keywords, and rewriting the query.
#
# LLM selection is required here, in the future change to CONFIG-defined models

from typing import List, Dict
import re
import openai
import os

class QueryProcessor:
    def __init__(self, method="none", model=None, llm_model_name=None):
        self.method = method
        self.model = model  # Can be a model name or callable
        self.llm_model_name = llm_model_name or "gpt-4o-mini"

    def process(self, query: str) -> Dict:
        if self.method == "rewrite_llm":
            return self.rewrite_query_llm(query)
        if self.method == "none":
            return {"processed_query": query}
        elif self.method == "expand":
            return self.expand_query(query)
        elif self.method == "extract_keywords":
            return self.extract_keywords(query)
        elif self.method == "rewrite":
            return self.rewrite_query(query)
        elif self.method == "chain":
            # Example: extract then expand
            keywords = self.extract_keywords(query)["keywords"]
            expanded = self.expand_query(" ".join(keywords))["processed_query"]
            return {"processed_query": expanded, "keywords": keywords}
        else:
            raise ValueError(f"Unknown query processing method: {self.method}")

    def rewrite_query_llm(self, query: str) -> dict:
        """
        Rewrite the query using an LLM (OpenAI example).
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "Rewrite the following user query to be as clear, complete, and answerable as possible by an AI system for financial document search.\n\n"
            f"User query: {query}\n\nRewritten:"
        )
        response = client.chat.completions.create(
            model=self.llm_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that reformulates queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=64,
            temperature=0.2,
        )
        rewritten = response.choices[0].message.content.strip()
        return {"processed_query": rewritten}
    
    def expand_query(self, query: str) -> Dict:
        # Example: simple synonym expansion, or call an LLM if model is set
        synonyms = {
            "revenue": ["sales", "income", "turnover"],
            "net income": ["profit", "earnings"],
            # ...
        }
        words = query.split()
        expanded = set(words)
        for word in words:
            for k, syns in synonyms.items():
                if word.lower() in [k] + syns:
                    expanded.update([k] + syns)
        return {"processed_query": " ".join(expanded)}

    def extract_keywords(self, query: str) -> Dict:
        # Simple: extract noun phrases or all nouns
        # For demo, just split by words > 3 letters or call a real NLP pipeline
        keywords = [w for w in re.findall(r'\w+', query) if len(w) > 3]
        return {"keywords": keywords, "processed_query": " ".join(keywords)}

    def rewrite_query(self, query: str) -> Dict:
        # Use LLM to rewrite, or simple rule-based for demo
        if callable(self.model):
            rewritten = self.model(query)
        else:
            rewritten = f"Please answer: {query}"
        return {"processed_query": rewritten}
