import os

def openai_validate_answer(answer, context, query, model="gpt-4o-mini", api_key=None, max_tokens=256):
    import openai
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    validation_prompt = (
        "You are an expert validator for AI-generated answers. "
        "Given the context and question below, for the provided answer, "
        "determine the following:\n"
        "1. Does the answer contain any hallucinated facts (not supported by context)? (yes/no)\n"
        "2. Is the answer grounded in the context (uses info from it)? (yes/no)\n"
        "3. Does the answer directly answer the user's question? (yes/no)\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:\n{answer}\n\n"
        "Reply with a JSON: {\"hallucination\": yes/no, \"grounded\": yes/no, \"answers_question\": yes/no, \"explanation\": ...}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a rigorous answer validator."},
            {"role": "user", "content": validation_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def hf_validate_answer(answer, context, query, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_tokens=256):
    from transformers import pipeline
    validation_prompt = (
        "You are an expert validator for AI-generated answers. "
        "Given the context and question below, for the provided answer, "
        "determine the following:\n"
        "1. Does the answer contain any hallucinated facts (not supported by context)? (yes/no)\n"
        "2. Is the answer grounded in the context (uses info from it)? (yes/no)\n"
        "3. Does the answer directly answer the user's question? (yes/no)\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:\n{answer}\n\n"
        "Reply with a JSON: {\"hallucination\": yes/no, \"grounded\": yes/no, \"answers_question\": yes/no, \"explanation\": ...}"
    )
    generator = pipeline("text-generation", model=model_name)
    result = generator(validation_prompt, max_new_tokens=max_tokens, do_sample=False)
    return result[0]['generated_text'].strip()

def parse_validation_output(raw_output):
    import json
    try:
        return json.loads(raw_output)
    except Exception:
        content = raw_output.lower()
        return {
            "hallucination": "yes" if "hallucinat" in content and "yes" in content else "no",
            "grounded": "yes" if "grounded" in content and "yes" in content else "no",
            "answers_question": "yes" if "answer" in content and "yes" in content else "no",
            "explanation": content,
        }

def validate_answer(answer, context, query, config=None, method=None):
    """
    Flexible answer validation via OpenAI, HuggingFace, or any callable (method).
    If 'config' is provided, uses config['validation_llm_method'] and model/config params.
    If 'method' is provided, uses method(answer, context, query, ...)
    """
    if method:
        # Use custom callable for validation
        raw_output = method(answer, context, query)
        return parse_validation_output(raw_output)
    if config is None:
        config = {}
    llm_method = config.get("validation_llm_method", "openai")
    model = config.get("validation_llm_model", "gpt-4o-mini")
    max_tokens = config.get("validation_llm_max_tokens", 256)
    api_key = config.get("openai_api_key", None)
    if llm_method == "openai":
        raw_output = openai_validate_answer(
            answer, context, query,
            model=model, api_key=api_key, max_tokens=max_tokens
        )
    elif llm_method == "hf":
        raw_output = hf_validate_answer(
            answer, context, query,
            model_name=model, max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unknown validation_llm_method: {llm_method}")
    return parse_validation_output(raw_output)
