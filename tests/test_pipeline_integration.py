def test_pipeline_query_processing():
    # Simulate a pipeline state
    from src.edgar_rag.query_processing import QueryProcessor
    from src.edgar_rag.langgraph_pipeline import query_processing_node, PipelineState
    config = {
        "query_processing": "rewrite_llm",
        "retrieval_query": "net income in 2023?",
        "llm_query_model": "gpt-4o-mini"
    }
    state = PipelineState()
    new_state = query_processing_node(state, config)
    print("Final processed query:", new_state.query)
    assert isinstance(new_state.query, str) and len(new_state.query) > 10
