import asyncio
import json
from pathlib import Path

import torch
from termcolor import cprint

from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

prompt = """
You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
What is the total number of decile for the redwood school locality?

### Context:
CREATE TABLE table_name_34 (decile VARCHAR, name VARCHAR)

### Response:
"""
max_new_tokens = 128
top_k = 50
top_p = 0.9
temperature = 0.75
base_model_path = Path("/root/Llama-2-7b")
lora_path = Path("/root/adapters/sql/")
tokenizer_path = str(base_model_path.resolve())
args = AsyncEngineArgs(
    model=base_model_path,
    tokenizer=tokenizer_path,
    dtype="auto",
    max_num_seqs=16384,
)
engine = AsyncLLMEngine.from_engine_args(args)

with open(lora_path / "adapter_config.json", "r") as f:
    adapter_config = json.load(f)

adapter_model = torch.load(
    lora_path / "adapter_model.bin", map_location="cpu")
print("Started loading lroa")
engine.engine.load_lora(adapter_config, adapter_model)
print("Finished loading lora")

sampling_params = SamplingParams(
    n=1,
    top_p=top_p,
    top_k=top_k,
    temperature=temperature,
    max_tokens=max_new_tokens,
)


async def run_engine(engine, prompt, sampling_params):
    results_generator = engine.generate(prompt, sampling_params, 0)
    async for request_output in results_generator:
        pass
    return request_output.outputs[0].text

cprint("Lora is loaded output:", "green")
completion = asyncio.run(run_engine(engine, prompt, sampling_params))
cprint(completion, "green")

cprint("Lora is deleted output:", "red")
engine.engine.delete_lora()
completion = asyncio.run(run_engine(engine, prompt, sampling_params))
cprint(completion, "red")

cprint("Lora is re-loaded output:", "green")
engine.engine.load_lora(adapter_config, adapter_model)
completion = asyncio.run(run_engine(engine, prompt, sampling_params))
cprint(completion, "green")
