import asyncio
import json
from pathlib import Path

import torch

from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams


def get_vllm_output(base_model_path: Path, lora_path: Path, prompt: str, max_new_tokens: int, top_k: int, top_p: float, temperature: float):
    completion = asyncio.run(_run_vllm(base_model_path, lora_path, prompt,
                max_new_tokens, top_k, top_p, temperature))
    return prompt + completion


async def _run_vllm(base_model_path: Path, lora_path: Path, prompt: str, max_new_tokens: int, top_k: int, top_p: float, temperature: float):
    tokenizer_path = str(base_model_path.resolve())
    args = AsyncEngineArgs(
        model=base_model_path,
        tokenizer=tokenizer_path,
        dtype="float16",
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
    results_generator = engine.generate(prompt, sampling_params, 0)

    async for request_output in results_generator:
        pass
        # print(request_output)

    return request_output.outputs[0].text
