"""Register a ``megatron.bridge`` bridge for GLM-4.7-Flash (``Glm4MoeLiteForCausalLM``).

GLM-4.7-Flash is a **DeepSeek-V3-architecture** model: Multi-Latent Attention
(MLA: ``q_a_proj``/``q_b_proj``/``kv_a_proj_with_mqa``/``kv_b_proj`` +
``q_a_layernorm``/``kv_a_layernorm``) + sigmoid-router MoE (64 routed / 1 shared,
``first_k_dense_replace=1``) + one MTP layer. It is NOT standard multi-head
attention.

Earlier this subclassed ``GLM45Bridge`` (GLM-4.5, ``Glm4MoeForCausalLM``) — but
GLM-4.5 uses **standard QKV attention**, so its ``mapping_registry`` looks for
``self_attn.q_proj/k_proj/v_proj`` (fused into ``linear_qkv``). Those tensors
don't exist in an MLA checkpoint, so the HF→Megatron load emitted hundreds of
"Can't find HF parameters" warnings and then died in ``scatter_to_tp_ranks``
(``expected (2048,320), got (2048,1280)``) mis-sharding the MLA weights at TP=4.

Fix: inherit ``DeepSeekV3Bridge`` instead. Its ``provider_bridge`` builds the MLA
GPTModel and ``mapping_registry`` (``deepseek/common.get_common_mapping_list``)
carries the exact MLA + MoE + MTP weight mappings. Verified against the actual
GLM-4.7-Flash checkpoint: MLA names match, MoE (router/experts/shared) match, and
the MTP layer-47 names (``enorm``/``hnorm``/``eh_proj``/``shared_head.norm``)
match DeepSeek-V3 exactly. All config fields DeepSeekV3Bridge reads are present
(``first_k_dense_replace``, ``n_shared_experts``, ``moe_intermediate_size``,
``num_nextn_predict_layers``, ``rope_theta``; ``rope_scaling`` is None → plain rope).

MTP is disabled here (``mtp_num_layers=None``): it is not used for the RL policy
forward, and skipping it avoids any MTP↔LoRA weight-sync edge cases (LoRA adapters
already target only ``decoder.layers.*`` so MTP carries no adapter). The unused
MTP mappings in the common registry simply have no destination and are skipped;
the layer-47 HF weights go unmapped (harmless warnings). This mirrors GLM5Bridge's
choice for the same reason.

Importing this module is idempotent — safe to import multiple times.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _register_glm4moe_lite_bridge() -> None:
    from transformers import Glm4MoeLiteForCausalLM

    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
    from megatron.bridge.models.mla_provider import MLAModelProvider
    from megatron.core.models.gpt.gpt_model import GPTModel

    # Already registered (idempotent import) → nothing to do.
    if getattr(DeepSeekV3Bridge, "_miles_lite_registered", False):
        return

    @MegatronModelBridge.register_bridge(
        source=Glm4MoeLiteForCausalLM,
        target=GPTModel,
        provider=MLAModelProvider,
        model_type="glm4_moe_lite",
    )
    class MilesGLM4MoeLiteBridge(DeepSeekV3Bridge):
        """GLM-4.7-Flash (Glm4MoeLite): DeepSeek-V3-style MLA+MoE, reusing DeepSeekV3Bridge."""

        def provider_bridge(self, hf_pretrained):
            provider = super().provider_bridge(hf_pretrained)
            # RL policy forward doesn't use the MTP head; disabling it sidesteps any
            # MTP↔LoRA weight-sync edge cases (adapters target decoder.layers only).
            provider.mtp_num_layers = None
            return provider

    DeepSeekV3Bridge._miles_lite_registered = True
    logger.warning(
        "miles GLM4MoeLite (GLM-4.7-Flash) bridge registered for megatron.bridge "
        "AutoBridge — inheriting DeepSeekV3Bridge (MLA+MoE), MTP disabled"
    )


try:
    _register_glm4moe_lite_bridge()
except Exception as _e:  # pragma: no cover - defensive; don't block other models
    logger.warning("miles glm4moe_lite bridge failed to register: %s", _e)
