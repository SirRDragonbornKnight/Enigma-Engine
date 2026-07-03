"""
Weight Mapping - Convert external model weights to Forge/Enigma format.

Maps weight names from HuggingFace, GGUF, and ONNX formats to the Forge
model architecture used by Enigma Engine.

Forge model weight structure:
    tok_embeddings.weight           - Token embeddings
    layers.{i}.attention.wq.weight  - Query projection
    layers.{i}.attention.wk.weight  - Key projection
    layers.{i}.attention.wv.weight  - Value projection
    layers.{i}.attention.wo.weight  - Output projection
    layers.{i}.feed_forward.w1.weight - FFN gate projection (SwiGLU)
    layers.{i}.feed_forward.w2.weight - FFN down projection
    layers.{i}.feed_forward.w3.weight - FFN up projection (SwiGLU)
    layers.{i}.attention_norm.weight  - Pre-attention RMSNorm
    layers.{i}.ffn_norm.weight        - Pre-FFN RMSNorm
    norm.weight                       - Final RMSNorm
    output.weight                     - Output projection (tied with tok_embeddings)
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace → Forge weight name mappings
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry: (HuggingFace pattern, Forge replacement)
# Patterns use regex. Group \1 captures the layer index.

HF_LLAMA_MAP = [
    # Embeddings
    (r"model\.embed_tokens\.weight", "tok_embeddings.weight"),
    (r"lm_head\.weight", "output.weight"),
    # Layer norms
    (r"model\.layers\.(\d+)\.input_layernorm\.weight", r"layers.\1.attention_norm.weight"),
    (r"model\.layers\.(\d+)\.post_attention_layernorm\.weight", r"layers.\1.ffn_norm.weight"),
    # Attention
    (r"model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)", r"layers.\1.attention.wq.\2"),
    (r"model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)", r"layers.\1.attention.wk.\2"),
    (r"model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)", r"layers.\1.attention.wv.\2"),
    (r"model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)", r"layers.\1.attention.wo.\2"),
    # FFN (SwiGLU)
    (r"model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)", r"layers.\1.feed_forward.w1.\2"),
    (r"model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)", r"layers.\1.feed_forward.w2.\2"),
    (r"model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)", r"layers.\1.feed_forward.w3.\2"),
    # Final norm
    (r"model\.norm\.weight", "norm.weight"),
]

HF_GPT2_MAP = [
    # Embeddings
    (r"transformer\.wte\.weight", "tok_embeddings.weight"),
    (r"transformer\.wpe\.weight", "_positional_embeddings.weight"),  # GPT-2 uses absolute position
    # Layer norms
    (r"transformer\.h\.(\d+)\.ln_1\.(weight|bias)", r"layers.\1.attention_norm.\2"),
    (r"transformer\.h\.(\d+)\.ln_2\.(weight|bias)", r"layers.\1.ffn_norm.\2"),
    # Attention (GPT-2 uses fused QKV)
    (r"transformer\.h\.(\d+)\.attn\.c_attn\.(weight|bias)", r"layers.\1.attention._qkv_fused.\2"),
    (r"transformer\.h\.(\d+)\.attn\.c_proj\.(weight|bias)", r"layers.\1.attention.wo.\2"),
    # FFN
    (r"transformer\.h\.(\d+)\.mlp\.c_fc\.(weight|bias)", r"layers.\1.feed_forward.w1.\2"),
    (r"transformer\.h\.(\d+)\.mlp\.c_proj\.(weight|bias)", r"layers.\1.feed_forward.w2.\2"),
    # Final norm
    (r"transformer\.ln_f\.(weight|bias)", r"norm.\1"),
]

HF_MISTRAL_MAP = HF_LLAMA_MAP  # Same architecture as Llama

HF_PHI_MAP = [
    # Embeddings
    (r"model\.embed_tokens\.weight", "tok_embeddings.weight"),
    (r"lm_head\.(weight|bias)", r"output.\1"),
    # Layer norms
    (r"model\.layers\.(\d+)\.input_layernorm\.(weight|bias)", r"layers.\1.attention_norm.\2"),
    (r"model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", r"layers.\1.ffn_norm.\2"),
    # Attention (Phi uses dense projections)
    (r"model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)", r"layers.\1.attention.wq.\2"),
    (r"model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)", r"layers.\1.attention.wk.\2"),
    (r"model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)", r"layers.\1.attention.wv.\2"),
    (r"model\.layers\.(\d+)\.self_attn\.dense\.(weight|bias)", r"layers.\1.attention.wo.\2"),
    # FFN
    (r"model\.layers\.(\d+)\.mlp\.fc1\.(weight|bias)", r"layers.\1.feed_forward.w1.\2"),
    (r"model\.layers\.(\d+)\.mlp\.fc2\.(weight|bias)", r"layers.\1.feed_forward.w2.\2"),
    # Final norm
    (r"model\.final_layernorm\.(weight|bias)", r"norm.\1"),
]

HF_QWEN_MAP = [
    # Embeddings
    (r"model\.embed_tokens\.weight", "tok_embeddings.weight"),
    (r"lm_head\.weight", "output.weight"),
    # Layer norms
    (r"model\.layers\.(\d+)\.input_layernorm\.weight", r"layers.\1.attention_norm.weight"),
    (r"model\.layers\.(\d+)\.post_attention_layernorm\.weight", r"layers.\1.ffn_norm.weight"),
    # Attention
    (r"model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)", r"layers.\1.attention.wq.\2"),
    (r"model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)", r"layers.\1.attention.wk.\2"),
    (r"model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)", r"layers.\1.attention.wv.\2"),
    (r"model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)", r"layers.\1.attention.wo.\2"),
    # FFN (SwiGLU)
    (r"model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)", r"layers.\1.feed_forward.w1.\2"),
    (r"model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)", r"layers.\1.feed_forward.w2.\2"),
    (r"model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)", r"layers.\1.feed_forward.w3.\2"),
    # Final norm
    (r"model\.norm\.weight", "norm.weight"),
    # QK normalization (Qwen3)
    (r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight", r"layers.\1.attention.q_norm.weight"),
    (r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight", r"layers.\1.attention.k_norm.weight"),
]

# Map model_type → mapping rules
HF_MODEL_MAPS = {
    "llama": HF_LLAMA_MAP,
    "mistral": HF_MISTRAL_MAP,
    "gpt2": HF_GPT2_MAP,
    "phi": HF_PHI_MAP,
    "phi3": HF_PHI_MAP,
    "qwen2": HF_QWEN_MAP,
    "qwen3": HF_QWEN_MAP,
    "gemma": HF_LLAMA_MAP,  # Gemma uses same layout as Llama
    "gemma2": HF_LLAMA_MAP,
}

# ═══════════════════════════════════════════════════════════════════════════════
# GGUF → Forge weight name mappings
# ═══════════════════════════════════════════════════════════════════════════════

GGUF_WEIGHT_MAP = [
    # Embeddings
    (r"token_embd\.weight", "tok_embeddings.weight"),
    (r"output\.weight", "output.weight"),
    # Layer norms
    (r"blk\.(\d+)\.attn_norm\.weight", r"layers.\1.attention_norm.weight"),
    (r"blk\.(\d+)\.ffn_norm\.weight", r"layers.\1.ffn_norm.weight"),
    # Attention
    (r"blk\.(\d+)\.attn_q\.weight", r"layers.\1.attention.wq.weight"),
    (r"blk\.(\d+)\.attn_k\.weight", r"layers.\1.attention.wk.weight"),
    (r"blk\.(\d+)\.attn_v\.weight", r"layers.\1.attention.wv.weight"),
    (r"blk\.(\d+)\.attn_output\.weight", r"layers.\1.attention.wo.weight"),
    # FFN
    (r"blk\.(\d+)\.ffn_gate\.weight", r"layers.\1.feed_forward.w1.weight"),
    (r"blk\.(\d+)\.ffn_down\.weight", r"layers.\1.feed_forward.w2.weight"),
    (r"blk\.(\d+)\.ffn_up\.weight", r"layers.\1.feed_forward.w3.weight"),
    # Final norm
    (r"output_norm\.weight", "norm.weight"),
]


class WeightMapper:
    """
    Maps weight names between external formats and Forge/Enigma format.

    Supports:
    - HuggingFace transformers (Llama, Mistral, GPT-2, Phi, Qwen, Gemma)
    - GGUF (llama.cpp format)
    - ONNX (Open Neural Network Exchange)

    Usage:
        mapper = WeightMapper()
        forge_weights = mapper.map_huggingface_to_forge(hf_state_dict)
        forge_weights = mapper.map_gguf_to_forge(gguf_tensors, config)
        forge_weights = mapper.map_onnx_to_forge(onnx_weights, config)
    """

    def __init__(self):
        self._stats = {
            "mapped": 0,
            "skipped": 0,
            "failed": 0,
        }

    def _apply_mapping(self, source_dict: dict, mapping_rules: list) -> dict:
        """
        Apply regex mapping rules to convert weight names.

        Args:
            source_dict: Source weight dictionary {name: tensor}
            mapping_rules: List of (pattern, replacement) regex rules

        Returns:
            Dictionary with Forge-compatible weight names
        """
        result = {}
        self._stats = {"mapped": 0, "skipped": 0, "failed": 0}

        for source_name, tensor in source_dict.items():
            mapped = False

            for pattern, replacement in mapping_rules:
                match = re.fullmatch(pattern, source_name)
                if match:
                    forge_name = re.sub(pattern, replacement, source_name)
                    result[forge_name] = tensor
                    self._stats["mapped"] += 1
                    mapped = True
                    logger.debug(f"Mapped: {source_name} → {forge_name}")
                    break

            if not mapped:
                # Keep unmapped weights with a warning
                self._stats["skipped"] += 1
                logger.warning(f"Skipped unmapped weight: {source_name}")

        total = self._stats["mapped"] + self._stats["skipped"]
        logger.info(f"Weight mapping: {self._stats['mapped']} mapped, {self._stats['skipped']} skipped")

        # Raise if more than 10% of weights are unmapped — likely wrong
        # model type or corrupted checkpoint.
        if total > 0 and self._stats["skipped"] / total > 0.10:
            raise ValueError(
                f"{self._stats['skipped']}/{total} weights unmapped "
                f"({100 * self._stats['skipped'] / total:.0f}%). "
                f"Check model type or weight names."
            )

        return result

    def _detect_hf_model_type(self, state_dict: dict) -> str:
        """
        Auto-detect HuggingFace model type from weight names.

        Returns:
            Model type string (e.g., 'llama', 'gpt2', 'phi')
        """
        sample_keys = list(state_dict.keys())[:20]
        key_str = " ".join(sample_keys)

        if "transformer.h." in key_str and "transformer.wte" in key_str:
            return "gpt2"
        if "model.layers." in key_str and "self_attn.q_proj" in key_str:
            # Could be llama, mistral, qwen, gemma...
            if any("mlp.gate_proj" in k for k in state_dict.keys()):
                return "llama"  # SwiGLU FFN → Llama-style
            if any("mlp.fc1" in k for k in state_dict.keys()):
                return "phi"
            return "llama"  # Default for this pattern

        logger.warning("Could not auto-detect model type, defaulting to llama")
        return "llama"

    def map_huggingface_to_forge(self, hf_state_dict: dict, model_type: Optional[str] = None) -> dict:
        """
        Convert HuggingFace state dict to Forge format.

        Args:
            hf_state_dict: HuggingFace model state dict
            model_type: Model type ('llama', 'gpt2', 'phi', etc.)
                       Auto-detected if None.

        Returns:
            Forge-compatible state dict
        """
        if model_type is None:
            model_type = self._detect_hf_model_type(hf_state_dict)

        model_type = model_type.lower()
        mapping_rules = HF_MODEL_MAPS.get(model_type, HF_LLAMA_MAP)

        logger.info(f"Mapping HuggingFace weights (type={model_type}, rules={len(mapping_rules)})")

        return self._apply_mapping(hf_state_dict, mapping_rules)

    def map_gguf_to_forge(self, gguf_tensors: dict, config: Any = None) -> dict:
        """
        Convert GGUF tensors to Forge format.

        Args:
            gguf_tensors: Dictionary of {tensor_name: tensor_data}
            config: Optional ForgeConfig for validation

        Returns:
            Forge-compatible state dict
        """
        logger.info(f"Mapping {len(gguf_tensors)} GGUF tensors to Forge format")
        return self._apply_mapping(gguf_tensors, GGUF_WEIGHT_MAP)

    def map_onnx_to_forge(self, onnx_weights: dict, config: Any = None) -> dict:
        """
        Convert ONNX weights to Forge format.

        ONNX models don't have a standard naming convention, so we try
        multiple strategies:
        1. Match against known HuggingFace patterns (many ONNX models
           are exported from HuggingFace)
        2. Match against GGUF patterns
        3. Heuristic matching based on tensor shapes

        Args:
            onnx_weights: Dictionary of {weight_name: tensor_data}
            config: Optional ForgeConfig for shape-based matching

        Returns:
            Forge-compatible state dict
        """
        logger.info(f"Mapping {len(onnx_weights)} ONNX weights to Forge format")

        # Try HuggingFace mapping first (most ONNX models come from HF)
        model_type = self._detect_hf_model_type(onnx_weights)
        mapping_rules = HF_MODEL_MAPS.get(model_type, [])

        # Combine with GGUF mapping as fallback
        all_rules = mapping_rules + GGUF_WEIGHT_MAP

        result = self._apply_mapping(onnx_weights, all_rules)

        if not result and config is not None:
            # Last resort: shape-based heuristic matching
            logger.info("Trying shape-based heuristic matching...")
            result = self._shape_based_mapping(onnx_weights, config)

        return result

    def _shape_based_mapping(self, weights: dict, config: Any) -> dict:
        """
        Map weights based on tensor shapes when name-based mapping fails.

        This is a last resort for ONNX models with non-standard naming.
        """
        result = {}
        dim = getattr(config, "dim", None)
        vocab_size = getattr(config, "vocab_size", None)

        if dim is None or vocab_size is None:
            logger.warning("Cannot do shape-based mapping without dim and vocab_size in config")
            return result

        for name, tensor in weights.items():
            shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()

            # Token embeddings: [vocab_size, dim]
            if shape == (vocab_size, dim) and "tok_embeddings.weight" not in result:
                result["tok_embeddings.weight"] = tensor
                logger.info(f"Shape-matched: {name} → tok_embeddings.weight")

            # Final norm: [dim]
            elif shape == (dim,) and "norm.weight" not in result:
                # Could be any norm - skip for safety
                pass

        return result

    def get_stats(self) -> dict:
        """Get statistics from the last mapping operation."""
        return self._stats.copy()
