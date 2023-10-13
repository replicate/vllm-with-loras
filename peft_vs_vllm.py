from pathlib import Path

from peft import AutoPeftModelForCausalLM, get_peft_model
from termcolor import cprint
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from hf_manager import get_peft_output, get_transformers_output
from vllm_manager import get_vllm_output

LORA_PATH = Path("/root/adapters/sql")
MODEL_PATH = Path("/root/Llama-2-7b")

PROMPT = """
You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
What is the total number of decile for the redwood school locality?

### Context:
CREATE TABLE table_name_34 (decile VARCHAR, name VARCHAR)

### Response:
"""

MAX_NEW_TOKENS = 128
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 0.75
ENGINES_REGISTRY = {"vllm": get_vllm_output,
                    "peft": get_peft_output, "hf": get_transformers_output}
engine_kwargs = {"base_model_path": MODEL_PATH, "lora_path": LORA_PATH,
                 "prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS, "top_k": TOP_K, "top_p": TOP_P, "temperature": TEMPERATURE}

cprint("Running base model:", "blue")
base_output = ENGINES_REGISTRY['hf'](**engine_kwargs)

cprint("Running PEFT:", "green")
peft_output = ENGINES_REGISTRY['peft'](**engine_kwargs)

cprint("Running vLLM:", "red")
vllm_output = ENGINES_REGISTRY['vllm'](**engine_kwargs)

cprint("Base Model Output:", "blue")
print(base_output)

cprint("\nPEFT:", "green")
print(peft_output)

cprint("\nVLLM:", "red")
print(vllm_output)
