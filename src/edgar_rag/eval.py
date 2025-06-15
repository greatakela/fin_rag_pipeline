# src/edgar_rag/eval.py

import numpy as np
from typing import List, Dict, Any
from ragas.metrics import faithfulness, answer_relevancy, context_precision

def evaluate_generation_with_ragas(answer: str, context: str, question: str) -> Dict[str, float]:
    """
    Compute RAGAS metrics for LLM output.
    """
    metrics = {}
    if faithfulness and answer_relevancy and context_precision:
        try:
            metrics['faithfulness'] = float(faithfulness(answer, context))
            metrics['answer_relevancy'] = float(answer_relevancy(answer, question))
            metrics['context_precision'] = float(context_precision(answer, context))
        except Exception as e:
            metrics['error'] = str(e)
    else:
        metrics['faithfulness'] = metrics['answer_relevancy'] = metrics['context_precision'] = -1
        metrics['error'] = "RAGAS not installed"
    return metrics


def chunk_coverage(filings: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> float:
    """
    Computes the percent of original text that appears in the chunk set.
    """
    all_text = " ".join([f["text"] for f in filings])
    all_chunk_text = " ".join([c["chunk"] for c in chunks])
    return len(all_chunk_text) / max(len(all_text), 1)

'''
def chunk_redundancy(chunks: List[Dict[str, Any]]) -> float:
    """
    Simple token overlap measure for redundancy.
    """
    seen = set()
    repeated = 0
    for c in chunks:
        for tok in c["chunk"].split():
            if tok in seen:
                repeated += 1
            else:
                seen.add(tok)
    total = sum(len(c["chunk"].split()) for c in chunks)
    return repeated / max(total, 1)
'''
def recall_at_k(query_emb, chunk_embs, gold_idx, k=5):
    """
    query_emb: np.array
    chunk_embs: np.array shape [n, d]
    gold_idx: int
    """
    sims = chunk_embs @ query_emb
    top_k = np.argsort(sims)[-k:]
    return int(gold_idx in top_k)


def retrieval_metrics(query: str, candidate_chunks: List[str], gold_answers: List[str], k=5) -> Dict[str, float]:
    """
    Checks if any gold answer is in top-k retrieved chunks.
    """
    hit = 0
    rr = 0.0
    for gold in gold_answers:
        try:
            idx = [c.lower() for c in candidate_chunks].index(gold.lower())
            if idx < k:
                hit = 1
                rr = 1.0 / (idx + 1)
                break
        except ValueError:
            continue
    return {"recall@k": hit, "mrr": rr}

def answer_exact_match(answer: str, gold: str) -> float:
    return float(answer.strip().lower() == gold.strip().lower())

def json_safe(obj):
    """Recursively convert numpy types to their native Python equivalents for JSON serialization."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj




