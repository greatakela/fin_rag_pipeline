# src/edgar_rag/llm.py

# This module contains the LLM answer generator functions.
# It includes functions for generating answers using OpenAI and HuggingFace models.
# It also includes a function for finetuning an LLM on a dataset.
#
# llm selection is required here, in the future change to CONFIG-defined models

import os

def openai_generate_answer(query, context, model="gpt-4o", prompt_template=None, api_key=None, max_tokens=400):
    import openai
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    if prompt_template:
        prompt = prompt_template.format(query=query, context=context)
    else:
        prompt = f"Answer the following question using only the context provided. Be concise and specific.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial filings QA assistant. ONLY answer from the context."},
            {"role": "user", "content": prompt_template.format(context=context, query=query)}
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def hf_generate_answer(query, context, model_name="meta-llama/Meta-Llama-3-8B-Instruct", prompt_template=None, max_tokens=400):
    from transformers import pipeline
    if prompt_template:
        prompt = prompt_template.format(query=query, context=context)
    else:
        prompt = f"[INST] Context: {context}\n\nQuestion: {query}\n\nAnswer: [/INST]"
    generator = pipeline("text-generation", model=model_name)
    out = generator(prompt, max_new_tokens=max_tokens, do_sample=False)
    return out[0]['generated_text'].strip()

def llm_generate_answer(query, context, config):
    """Main LLM answer generator switcher."""
    method = config.get("llm_method", "openai")
    model = config.get("llm_model", "gpt-4o")
    prompt_template = config.get("prompt_template", None)
    max_tokens = config.get("llm_max_tokens", 400)
    api_key = config.get("openai_api_key", None)
    if method == "openai":
        return openai_generate_answer(query, context, model=model, prompt_template=prompt_template, api_key=api_key, max_tokens=max_tokens)
    elif method == "hf":
        return hf_generate_answer(query, context, model_name=model, prompt_template=prompt_template, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown LLM method: {method}")
    
    
def finetune_llm_on_dataset(dataset, model="gpt-4o", method="dpo", **kwargs):
    """
    Placeholder: Implement DPO/fine-tuning loop with your data and platform.
    """
    pass  # Implement with preferred library/platform (e.g., HuggingFace, OpenAI)

