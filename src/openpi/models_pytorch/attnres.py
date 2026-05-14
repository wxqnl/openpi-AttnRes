from __future__ import annotations

import math
import inspect
import types
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch

import openpi.models_pytorch.pi0_pytorch as pi0_pytorch


@dataclass
class PI05PaliGemmaAttnResOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    alpha_list: list[torch.Tensor] | None = None
    skip_trace: list[dict] | None = None
    entropy_penalty: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    fp = x.float()
    inv = torch.rsqrt(fp.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (fp * inv).to(x.dtype)


class BlockAttnResRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_sources: int,
        temperature: float = 1.0,
        use_positional_bias: bool = True,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sources = num_sources
        self.base_temperature = temperature
        self.w_query = nn.Parameter(torch.empty(num_sources, hidden_size))
        if use_positional_bias:
            self.key_pos_bias = nn.Parameter(torch.empty(num_sources, hidden_size))
        else:
            self.register_parameter("key_pos_bias", None)
        nn.init.normal_(self.w_query, std=initializer_range)
        if self.key_pos_bias is not None:
            nn.init.normal_(self.key_pos_bias, std=initializer_range)

    def route(self, position: int, completed_outputs: list[torch.Tensor]):
        values = torch.stack(completed_outputs, dim=0)  # [N, B, T, H]
        keys = _rms_norm(values)
        if self.key_pos_bias is not None:
            bias = self.key_pos_bias[:position].to(keys.dtype)
            keys = keys + bias[:, None, None, :]
        query = self.w_query[position].to(keys.dtype)
        scale = math.sqrt(values.shape[-1]) * self.base_temperature
        scores = torch.einsum("h,nbth->nbt", query, keys) / scale
        alpha = torch.softmax(scores.float(), dim=0).to(values.dtype)
        routed = torch.einsum("nbt,nbth->bth", alpha, values)
        return routed, alpha.permute(1, 2, 0)


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_rank: int = 128):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_rank, bias=False)
        self.up = nn.Linear(adapter_rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


class PI05PaliGemmaAttnResRetrofit(nn.Module):
    """AttnRes retrofit wrapper for pi0.5's PaliGemma language model.

    The wrapper patches only ``paligemma.model.language_model.forward``. Vision
    encoding, image-token merge, and LM head remain owned by PaliGemma itself.
    With gamma initialized at zero, each block input is exactly the original
    previous hidden state, so the full path is identity-at-init.
    """

    def __init__(
        self,
        paligemma,
        num_blocks: int = 9,
        skippable_blocks: Iterable[int] | None = None,
        adapter_rank: int = 128,
        initializer_range: float = 0.02,
        no_adapter: bool = False,
    ):
        super().__init__()
        # Keep references to the already-owned PaliGemma modules without
        # registering them again under this retrofit module. This lets an
        # owning model save one clean state dict: base VLA weights once, plus
        # router/adapters/gamma.
        object.__setattr__(self, "base_model", paligemma)
        object.__setattr__(self, "language_model", paligemma.model.language_model)
        cfg = self.language_model.config
        self.hidden_size = int(cfg.hidden_size)
        self.vocab_size = int(getattr(getattr(paligemma, "config", None), "text_config", cfg).vocab_size)
        self.num_layers = int(cfg.num_hidden_layers)
        if self.num_layers % num_blocks != 0:
            raise ValueError(f"num_hidden_layers={self.num_layers} must be divisible by num_blocks={num_blocks}")

        self.num_blocks = int(num_blocks)
        self.layers_per_block = self.num_layers // self.num_blocks
        if skippable_blocks is None:
            skippable_blocks = range(self.num_blocks)
        self.skippable_blocks = tuple(sorted(set(int(x) for x in skippable_blocks)))
        self.skippable_block_set = set(self.skippable_blocks)

        self.router = BlockAttnResRouter(
            hidden_size=self.hidden_size,
            num_sources=self.num_blocks + 1,
            use_positional_bias=True,
            initializer_range=initializer_range,
        )
        self.no_adapter = bool(no_adapter)
        if self.no_adapter:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(self.num_blocks)])
        else:
            self.adapters = nn.ModuleList(
                [ResidualAdapter(self.hidden_size, adapter_rank=adapter_rank) for _ in range(self.num_blocks)]
            )
        self.gamma = nn.Parameter(torch.zeros(self.num_blocks))
        retrofit_dtype = self._infer_retrofit_dtype()
        self.router.to(dtype=retrofit_dtype)
        self.adapters.to(dtype=retrofit_dtype)

        self._return_alpha_flag = False
        self._active_skip_blocks: set[int] = set()
        self._fwd_alpha_list: list[torch.Tensor] | None = None
        self._fwd_skip_trace: list[dict] | None = None
        self._fwd_entropy: torch.Tensor | None = None

        self._install_retrofit_forward()

    @property
    def text_layers(self):
        return self.language_model.layers

    def _infer_retrofit_dtype(self) -> torch.dtype:
        for param in self.language_model.parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.float32

    def _install_retrofit_forward(self) -> None:
        retrofit = self
        lm = self.language_model
        if not hasattr(lm, "_pi05_attnres_original_forward"):
            lm._pi05_attnres_original_forward = lm.forward

        def patched_forward(
            self_lm,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            cache_position=None,
            adarms_cond=None,
            **kwargs,
        ):
            return retrofit._language_model_forward(
                self_lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                adarms_cond=adarms_cond,
                **kwargs,
            )

        lm.forward = types.MethodType(patched_forward, lm)

    def _compute_block_input(
        self,
        block_idx: int,
        prev_block: torch.Tensor,
        completed: list[torch.Tensor],
        collect_alpha: bool,
        compute_entropy: bool,
    ):
        if block_idx not in self.skippable_block_set:
            return prev_block, None, None
        routed, alpha = self.router.route(block_idx + 1, completed)
        if self.no_adapter:
            gamma = self.gamma[block_idx].to(prev_block.dtype)
            corrected = (1 - gamma) * prev_block + gamma * routed
        else:
            delta = routed - prev_block
            corrected = prev_block + self.gamma[block_idx].to(prev_block.dtype) * self.adapters[block_idx](delta)
        entropy = None
        if compute_entropy:
            entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return corrected, alpha if collect_alpha else None, entropy

    def _make_causal_mask(
        self,
        text_model,
        inputs_embeds: torch.Tensor,
        attention_mask,
        position_ids,
        past_key_values,
        cache_position,
    ):
        if hasattr(text_model, "_update_causal_mask"):
            return text_model._update_causal_mask(
                attention_mask,
                past_key_values,
                cache_position,
                inputs_embeds,
                False,
            )
        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask
        return attention_mask

    def _language_model_forward(
        self,
        text_model,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        adarms_cond=None,
        **kwargs,
    ):
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        output_attentions = bool(output_attentions) if output_attentions is not None else False
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = text_model.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            try:
                past_key_values = DynamicCache(config=text_model.config)
            except TypeError:
                past_key_values = DynamicCache()
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        if (
            len(text_model.layers) > 0
            and hasattr(text_model.layers[0], "self_attn")
            and text_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            hidden_states = hidden_states.to(torch.bfloat16)

        position_embeddings = None
        if hasattr(text_model, "rotary_emb"):
            position_embeddings = text_model.rotary_emb(hidden_states, position_ids)
        causal_mask = self._make_causal_mask(
            text_model,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            cache_position,
        )
        if isinstance(causal_mask, torch.Tensor) and causal_mask.is_floating_point():
            causal_mask = causal_mask.to(dtype=hidden_states.dtype)

        collect_trace = self._return_alpha_flag or self.training
        compute_entropy = self.training
        completed: list[torch.Tensor] = [hidden_states]
        prev_block = hidden_states
        alpha_list: list[torch.Tensor] = []
        skip_trace: list[dict] = []
        entropy_accum: torch.Tensor | None = None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        layer_counter = 0

        for block_idx in range(self.num_blocks):
            block_input, alpha, entropy = self._compute_block_input(
                block_idx,
                prev_block,
                completed,
                collect_alpha=collect_trace,
                compute_entropy=compute_entropy,
            )
            if entropy is not None:
                entropy_accum = entropy if entropy_accum is None else entropy_accum + entropy
            if alpha is not None:
                alpha_list.append(alpha)

            should_skip = block_idx in self._active_skip_blocks and block_idx in self.skippable_block_set
            h = block_input
            if not should_skip:
                for _ in range(self.layers_per_block):
                    if output_hidden_states:
                        all_hidden_states += (h,)
                    layer = text_model.layers[layer_counter]
                    layer_signature = inspect.signature(layer.forward)
                    past_key_arg = (
                        "past_key_values" if "past_key_values" in layer_signature.parameters else "past_key_value"
                    )
                    layer_kwargs = dict(
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        adarms_cond=adarms_cond,
                    )
                    layer_kwargs[past_key_arg] = past_key_values
                    layer_kwargs.update(kwargs)
                    try:
                        out = layer(h, **layer_kwargs)
                    except TypeError:
                        minimal = {
                            k: v
                            for k, v in layer_kwargs.items()
                            if k not in {"past_key_value", "past_key_values", "cache_position"}
                        }
                        out = layer(h, **minimal)
                    h = out[0] if isinstance(out, tuple) else out
                    if output_attentions and isinstance(out, tuple) and len(out) > 1:
                        all_self_attns += (out[1],)
                    layer_counter += 1
            else:
                layer_counter += self.layers_per_block

            prev_block = h
            completed.append(h)
            if collect_trace:
                skip_trace.append(
                    {
                        "block_idx": block_idx,
                        "used_attnres": block_idx in self.skippable_block_set,
                        "skipped": should_skip,
                        "skip_requested": block_idx in self._active_skip_blocks,
                        "gamma": float(self.gamma.detach()[block_idx].cpu()),
                    }
                )

        if hasattr(text_model, "norm"):
            try:
                hidden_states, _ = text_model.norm(completed[-1], adarms_cond)
            except TypeError:
                hidden_states = text_model.norm(completed[-1])
        else:
            hidden_states = completed[-1]
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        self._fwd_alpha_list = alpha_list if collect_trace else None
        self._fwd_skip_trace = skip_trace if collect_trace else None
        self._fwd_entropy = entropy_accum
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        token_type_ids=None,
        cache_position=None,
        inputs_embeds=None,
        labels=None,
        return_alpha: bool = False,
        skip_block_indices: Iterable[int] | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> PI05PaliGemmaAttnResOutput:
        self._return_alpha_flag = return_alpha
        self._active_skip_blocks = set(int(x) for x in (skip_block_indices or []))
        self._fwd_alpha_list = None
        self._fwd_skip_trace = None
        self._fwd_entropy = None

        out = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            **kwargs,
        )
        logits = out.logits
        loss = None
        if labels is not None:
            shifted = torch.cat([labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1)
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), shifted.reshape(-1), ignore_index=-100)

        return PI05PaliGemmaAttnResOutput(
            loss=loss,
            logits=logits,
            alpha_list=self._fwd_alpha_list if return_alpha else None,
            skip_trace=self._fwd_skip_trace,
            entropy_penalty=self._fwd_entropy,
            hidden_states=getattr(out, "hidden_states", None),
            attentions=getattr(out, "attentions", None),
        )

    def freeze_base(self) -> None:
        for p in self.base_model.parameters():
            p.requires_grad = False

    def retrofit_parameters(self):
        return list(self.router.parameters()) + list(self.adapters.parameters()) + [self.gamma]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class PI05AttnResPytorch(nn.Module):
    """Standalone pi0.5 VLA model with registered PaliGemma AttnRes modules."""

    def __init__(
        self,
        config,
        *,
        num_blocks: int = 9,
        adapter_rank: int = 256,
        no_adapter: bool = False,
    ):
        super().__init__()
        self.config = config
        self.pi0 = pi0_pytorch.PI0Pytorch(config)
        self.attnres = PI05PaliGemmaAttnResRetrofit(
            self.pi0.paligemma_with_expert.paligemma,
            num_blocks=num_blocks,
            adapter_rank=adapter_rank,
            no_adapter=no_adapter,
        )

    @property
    def paligemma_with_expert(self):
        return self.pi0.paligemma_with_expert

    def gradient_checkpointing_enable(self):
        return self.pi0.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        return self.pi0.gradient_checkpointing_disable()

    def sample_noise(self, *args, **kwargs):
        return self.pi0.sample_noise(*args, **kwargs)

    def embed_prefix(self, *args, **kwargs):
        return self.pi0.embed_prefix(*args, **kwargs)

    def embed_suffix(self, *args, **kwargs):
        return self.pi0.embed_suffix(*args, **kwargs)

    def denoise_step(self, *args, **kwargs):
        return self.pi0.denoise_step(*args, **kwargs)

    def sample_actions(self, *args, **kwargs):
        return self.pi0.sample_actions(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.pi0(*args, **kwargs)

    def retrofit_parameters(self):
        return self.attnres.retrofit_parameters()

    def set_attnres_trainable(self, trainable: bool) -> None:
        self.attnres.train(trainable)
        for param in self.attnres.retrofit_parameters():
            param.requires_grad = trainable

    def set_attnres_gamma(self, value: float) -> None:
        with torch.no_grad():
            self.attnres.gamma.data.fill_(value)

    def load_base_weights(self, weight_path: str, *, device: str | torch.device | None = None) -> None:
        load_kwargs = {"strict": False}
        if device is not None:
            load_kwargs["device"] = str(device)
        missing, unexpected = safetensors.torch.load_model(self.pi0, weight_path, **load_kwargs)
        allowed_tied_keys = {"paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"}
        disallowed_missing = sorted(set(missing) - allowed_tied_keys)
        disallowed_unexpected = sorted(set(unexpected) - allowed_tied_keys)
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                "Error(s) in loading base PI0Pytorch state_dict for PI05AttnResPytorch: "
                f"missing={disallowed_missing}, unexpected={disallowed_unexpected}"
            )

    def load_attnres_state(self, state_path: str, *, device: torch.device | str | None = None) -> None:
        ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
        dtype = next(self.attnres.parameters()).dtype
        target = torch.device(device) if device is not None else next(self.attnres.parameters()).device
        self.attnres.router.load_state_dict({k: v.to(device=target, dtype=dtype) for k, v in ckpt["router"].items()})
        if not bool(ckpt.get("config", {}).get("no_adapter", False)):
            self.attnres.adapters.load_state_dict(
                {k: v.to(device=target, dtype=dtype) for k, v in ckpt["adapters"].items()}
            )
        self.attnres.gamma.data.copy_(ckpt["gamma"].to(device=target, dtype=dtype))
