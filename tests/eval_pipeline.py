from edgar_rag.langgraph_pipeline import run_pipeline
from edgar_rag.eval_utils import batch_ragas_eval, log_langsmith, load_eval_set

eval_set = load_eval_set("test/eval_set.csv")
results = []
for i, row in eval_set.iterrows():
    output = run_pipeline(query=row["query"], config=...)
    answer = output["llm_answer"]
    context = "\n".join(output["selected_context"])
    results.append({
        "question": row["query"],
        "answer": answer,
        "context": context,
        "gold_answer": row["gold_answer"]
    })

metrics = batch_ragas_eval(results)
log_langsmith("full_pipeline_eval", config, metrics)
print("RAGAS Metrics:", metrics)
