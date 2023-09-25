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
    ):
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

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) *
            self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
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
            raise RuntimeError("A LoRA is currently merged into the model. You should delete the active LoRA before merging this one.")

        if self.merge_activations:
            raise NotImplementedError("Merging LoRAs in activation space is currently unsupported.")

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
            raise NotImplementedError("Merging LoRAs in activation space is currently unsupported.")
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

class LoraLlamaForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.lm_head = ColumnParallelLinear(config.hidden_size,
                                            vocab_size,
                                            bias=False,
                                            gather_output=False,
                                            perform_initialization=False)
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

    _column_parallel_weights = [
        "qkv_proj.weight", "gate_proj.weight", "up_proj.weight"
    ]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

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
                     load_format: str = "auto"):
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
                model_name_or_path, cache_dir, load_format):
            if "rotary_emb.inv_freq" in name:
                continue

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]

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

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight,
                                                  tensor_model_parallel_rank)
                continue

            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)

        # cache the original qkv proj matricies for restoring the original model when deleting a lora 
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn.cached_qkv_proj = layer.self_attn.qkv_proj.weight.data.clone()
            assert torch.allclose(layer.self_attn.cached_qkv_proj, layer.self_attn.qkv_proj.weight.data)
