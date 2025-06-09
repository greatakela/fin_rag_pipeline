from src.edgar_rag.query_processing import QueryProcessor

def test_llm_query_rewrite():
    # This will make a real OpenAI API call (needs your API key)
    processor = QueryProcessor(method="rewrite_llm", llm_model_name="gpt-4o-mini")
    input_query = "Nvidia 2022 total sales?"
    result = processor.process(input_query)
    print("Original:", input_query)
    print("Rewritten:", result["processed_query"])
    assert "Nvidia" in result["processed_query"]
    assert "2022" in result["processed_query"]
    assert "sales" in result["processed_query"] or "revenue" in result["processed_query"]
