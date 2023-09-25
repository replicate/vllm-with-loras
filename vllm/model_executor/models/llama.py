# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantized_linear import ParallelLinear
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import \
    VocabParallelEmbedding
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor, hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab, load_tensor_parallel_weights)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ParallelLinear.column(hidden_size,
                                                  2 * intermediate_size,
                                                  bias=False,
                                                  gather_output=False,
                                                  perform_initialization=False,
                                                  quant_config=quant_config)
        self.down_proj = ParallelLinear.row(intermediate_size,
                                            hidden_size,
                                            bias=False,
                                            input_is_parallel=True,
                                            perform_initialization=False,
                                            quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LinearLoRALayer(nn.Module):
    def __init__(self, hidden_dim, rank, dropout, scaling):
        super().__init__()
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False)
        self.scaling = scaling
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    @classmethod
    def from_state_dict(cls, state_dict):
        expected_keys = set(['loras', 'dropout', 'scaling'])
        if state_dict.keys() != expected_keys:
            raise ValueError(
                f"Expected keys {expected_keys}, got {state_dict.keys()}")

        rank, hidden_dim = state_dict['loras']['lora_A.weight'].shape
        dropout = state_dict['dropout']
        scaling = state_dict['scaling']
        lora_layer = cls(hidden_dim=hidden_dim, rank=rank,
                         dropout=dropout, scaling=scaling)
        lora_layer.load_state_dict(state_dict['loras'], strict=True)
        return lora_layer

    def forward(self, x):
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = ParallelLinear.column(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) *
            self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.o_proj = ParallelLinear.row(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.attn = PagedAttentionWithRoPE(self.num_heads,
                                           self.head_dim,
                                           self.scaling,
                                           base=self.rope_theta,
                                           rotary_dim=self.head_dim,
                                           num_kv_heads=self.num_kv_heads)

        self.q_lora = None
        self.v_lora = None
        self.merged = False
        self.cached_qkv_proj = None
        # TODO (Moin): expose this variable
        self.merge_activations = False

    def load_lora(self, q_lora_state_dict, v_lora_state_dict):
        # TODO (Moin): generalize this abstraction later on

        if self.merged:
            raise RuntimeError(
                "A LoRA is currently merged into the model. You should delete the active LoRA before merging this one.")

        if self.merge_activations:
            raise NotImplementedError(
                "Merging LoRAs in activation space is currently unsupported.")

            # (Moin): Leaving this comemnted for posterity + to fix later
            # self.q_lora = LinearLoRALayer.from_state_dict(
            # q_lora_state_dict).to("cuda").half()
            # self.v_lora = LinearLoRALayer.from_state_dict(
            # v_lora_state_dict).to("cuda").half()
        else:
            device = self.qkv_proj.weight.device

            lora_B = q_lora_state_dict['loras']['lora_B.weight']
            lora_A = q_lora_state_dict['loras']['lora_A.weight']
            scaling = q_lora_state_dict['scaling']
            q_lora = (lora_B @ lora_A) * scaling
            self.q_lora = q_lora.to(device)

            lora_B = v_lora_state_dict['loras']['lora_B.weight']
            lora_A = v_lora_state_dict['loras']['lora_A.weight']
            scaling = v_lora_state_dict['scaling']
            v_lora = (lora_B @ lora_A) * scaling
            self.v_lora = v_lora.to(device)

            self.qkv_proj.weight.requires_grad = False
            self.qkv_proj.weight[:q_lora.shape[0], :] += self.q_lora
            self.qkv_proj.weight[-v_lora.shape[0]:, :] += self.v_lora
            self.merged = True

    def delete_lora(self):
        if not self.merged:
            raise RuntimeError("No LoRA is currently merged into the model.")

        if self.merge_activations:
            raise NotImplementedError(
                "Merging LoRAs in activation space is currently unsupported.")
            self.q_lora = None
            self.v_lora = None
        else:
            # Leaving this other approach commented out
            # This could yield floating point issues, so I wanted to avoid it for now
            # self.qkv_proj.weight[:self.q_lora.shape[0], :] -= self.q_lora
            # self.qkv_proj.weight[-self.v_lora.shape[0]:, :] -= self.v_lora
            self.merged = False
            self.qkv_proj.weight.data = self.cached_qkv_proj

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # TODO (Moin): add the LoRA here
        if self.merge_activations and self.q_lora is not None and self.v_lora is not None:
            # matrix multiplies have the tendency to overflow, so we have to be careful about datatypes here
            prev_q_dtype = q.dtype
            prev_v_dtype = v.dtype
            assert prev_q_dtype == self.q_lora.lora_A.weight.dtype
            assert prev_q_dtype == self.q_lora.lora_B.weight.dtype

            assert prev_v_dtype == self.v_lora.lora_A.weight.dtype
            assert prev_v_dtype == self.v_lora.lora_B.weight.dtype

            q = q.to(self.q_lora.lora_A.weight.dtype)
            v = q.to(self.v_lora.lora_A.weight.dtype)

            q += self.q_lora(q)
            v += self.v_lora(v)

            q = q.to(prev_q_dtype)
            v = v.to(prev_v_dtype)

        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size, config.hidden_size, perform_initialization=False)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        # NOTE: The LM head is not quantized.
        self.lm_head = ParallelLinear.column(config.hidden_size,
                                             vocab_size,
                                             bias=False,
                                             gather_output=False,
                                             perform_initialization=False,
                                             quant_config=None)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_layers = []
    _row_parallel_layers = ["o_proj", "down_proj"]

    def load_lora(self, lora_config, lora_state_dict):
        # load configs, map to CPU in case # of GPUs is variable
        if isinstance(lora_state_dict, str):
            lora_state_dict = torch.load(lora_state_dict, map_location="cpu")

        if isinstance(lora_config, str):
            with open(lora_config, "r") as lora_config_file:
                lora_config = json.load(lora_config_file)

        # assemble the final state dict
        dropout = lora_config['lora_dropout']
        scaling = lora_config['lora_alpha'] / lora_config['r']
        for layer_idx, layer in enumerate(self.model.layers):
            q_lora_A_weight = lora_state_dict[
                f'base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight']
            q_lora_B_weight = lora_state_dict[
                f'base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight']
            q_lora_state_dict = {"loras": {"lora_A.weight": q_lora_A_weight,
                                           'lora_B.weight': q_lora_B_weight}, "scaling": scaling, "dropout": dropout}

            v_lora_A_weight = lora_state_dict[
                f'base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight']
            v_lora_B_weight = lora_state_dict[
                f'base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight']
            v_lora_state_dict = {"loras": {"lora_A.weight": v_lora_A_weight,
                                           'lora_B.weight': v_lora_B_weight}, "scaling": scaling, "dropout": dropout}

            layer.self_attn.load_lora(q_lora_state_dict, v_lora_state_dict)

    def delete_lora(self):
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn.delete_lora()

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        if self.quant_config is None:
            weight_suffixes = ["weight"]
        else:
            weight_suffixes = self.quant_config.get_tp_tensor_names()

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        tp_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        q_proj_shard_size = (self.config.hidden_size // tp_size)
        kv_proj_shard_size = (self.config.hidden_size //
                              self.config.num_attention_heads *
                              self.config.num_key_value_heads // tp_size)
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size,
             q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue

            is_packed = False
            is_transposed = False
            if self.quant_config is not None:
                is_packed = self.quant_config.is_packed(name)
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight.T

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]
                if is_transposed:
                    param = param.T

                if is_packed:
                    shard_size //= self.quant_config.pack_factor
                    offset //= self.quant_config.pack_factor

                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[offset:offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                if is_transposed:
                    param = param.T

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            if is_transposed:
                param = param.T

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight,
                                                  tensor_model_parallel_rank)
                continue

            load_tensor_parallel_weights(param, loaded_weight, name,
                                         column_parallel_weights,
                                         row_parallel_weights,
                                         tensor_model_parallel_rank)

        # cache the original qkv proj matricies for restoring the original model when deleting a lora
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn.cached_qkv_proj = layer.self_attn.qkv_proj.weight.data.clone()
            assert torch.allclose(
                layer.self_attn.cached_qkv_proj, layer.self_attn.qkv_proj.weight.data)
