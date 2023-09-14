from pathlib import Path

from peft import AutoPeftModelForCausalLM, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def get_peft_output(base_model_path: Path, lora_path: Path, prompt: str, max_new_tokens: int, top_k: int, top_p: float, temperature: float):
    model = AutoPeftModelForCausalLM.from_pretrained(lora_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return _run_generation(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, temperature=temperature)


def get_transformers_output(base_model_path: Path, lora_path: Path, prompt: str, max_new_tokens: int, top_k: int, top_p: float, temperature: float):
    model = AutoModelForCausalLM.from_pretrained(base_model_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return _run_generation(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, temperature=temperature)


def _run_generation(model, tokenizer, prompt: str, max_new_tokens: int, top_k: int, top_p: float, temperature: float):
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to("cuda")
    attention_mask = tokens.attention_mask.to("cuda")

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    output = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            generation_config=generation_config)
    return tokenizer.decode(output.squeeze())
