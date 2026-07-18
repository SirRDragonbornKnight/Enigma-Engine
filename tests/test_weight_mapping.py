"""Tests for weight mapping between HuggingFace, GGUF, ONNX and Forge formats."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Helper ───────────────────────────────────────────────────────────────────


def _make_dummy_tensor():
    """Return a lightweight stand-in for a weight tensor."""
    return MagicMock(shape=(64, 64))


# ── HuggingFace → Forge mapping ─────────────────────────────────────────────


class TestHuggingFaceLlamaMapping:
    """Test Llama-style HF → Forge weight mapping."""

    def test_embed_tokens(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {"model.embed_tokens.weight": _make_dummy_tensor()}
        result = mapper.map_huggingface_to_forge(sd, model_type="llama")
        assert "tok_embeddings.weight" in result

    def test_lm_head(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {"lm_head.weight": _make_dummy_tensor()}
        result = mapper.map_huggingface_to_forge(sd, model_type="llama")
        assert "output.weight" in result

    def test_attention_projections(self):
        """Identity asserts, not just key presence: tensors pass through by
        reference, so `is` pins WHICH source landed on WHICH target -- with
        four identical mocks and `in` checks, a q<->k table swap (a silently
        scrambled converted model) was undetectable (audit 2026-07-17)."""
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "model.layers.0.self_attn.q_proj.weight": _make_dummy_tensor(),
            "model.layers.0.self_attn.k_proj.weight": _make_dummy_tensor(),
            "model.layers.0.self_attn.v_proj.weight": _make_dummy_tensor(),
            "model.layers.0.self_attn.o_proj.weight": _make_dummy_tensor(),
        }
        result = mapper.map_huggingface_to_forge(sd, model_type="llama")
        assert result["layers.0.attention.wq.weight"] is sd["model.layers.0.self_attn.q_proj.weight"]
        assert result["layers.0.attention.wk.weight"] is sd["model.layers.0.self_attn.k_proj.weight"]
        assert result["layers.0.attention.wv.weight"] is sd["model.layers.0.self_attn.v_proj.weight"]
        assert result["layers.0.attention.wo.weight"] is sd["model.layers.0.self_attn.o_proj.weight"]

    def test_ffn_projections(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "model.layers.2.mlp.gate_proj.weight": _make_dummy_tensor(),
            "model.layers.2.mlp.down_proj.weight": _make_dummy_tensor(),
            "model.layers.2.mlp.up_proj.weight": _make_dummy_tensor(),
        }
        result = mapper.map_huggingface_to_forge(sd, model_type="llama")
        # identity, not presence: a w1<->w3 (gate/up) swap must fail here
        assert result["layers.2.feed_forward.w1.weight"] is sd["model.layers.2.mlp.gate_proj.weight"]
        assert result["layers.2.feed_forward.w2.weight"] is sd["model.layers.2.mlp.down_proj.weight"]
        assert result["layers.2.feed_forward.w3.weight"] is sd["model.layers.2.mlp.up_proj.weight"]

    def test_layer_norms(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "model.layers.0.input_layernorm.weight": _make_dummy_tensor(),
            "model.layers.0.post_attention_layernorm.weight": _make_dummy_tensor(),
            "model.norm.weight": _make_dummy_tensor(),
        }
        result = mapper.map_huggingface_to_forge(sd, model_type="llama")
        assert "layers.0.attention_norm.weight" in result
        assert "layers.0.ffn_norm.weight" in result
        assert "norm.weight" in result

    def test_multi_layer_index(self):
        """Layer indices should be preserved correctly for deeper layers."""
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "model.layers.15.self_attn.q_proj.weight": _make_dummy_tensor(),
        }
        result = mapper.map_huggingface_to_forge(sd, model_type="llama")
        assert "layers.15.attention.wq.weight" in result


class TestHuggingFaceGPT2Mapping:
    """Test GPT-2 HF → Forge weight mapping."""

    def test_gpt2_embeddings(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "transformer.wte.weight": _make_dummy_tensor(),
            "transformer.wpe.weight": _make_dummy_tensor(),
        }
        result = mapper.map_huggingface_to_forge(sd, model_type="gpt2")
        assert "tok_embeddings.weight" in result
        assert "_positional_embeddings.weight" in result

    def test_gpt2_attention_fused_qkv(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "transformer.h.0.attn.c_attn.weight": _make_dummy_tensor(),
            "transformer.h.0.attn.c_proj.weight": _make_dummy_tensor(),
        }
        result = mapper.map_huggingface_to_forge(sd, model_type="gpt2")
        assert "layers.0.attention._qkv_fused.weight" in result
        assert "layers.0.attention.wo.weight" in result


class TestHuggingFaceQwenMapping:
    """Test Qwen HF → Forge weight mapping (includes QK norm)."""

    def test_qwen_qk_norm(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "model.layers.0.self_attn.q_norm.weight": _make_dummy_tensor(),
            "model.layers.0.self_attn.k_norm.weight": _make_dummy_tensor(),
        }
        result = mapper.map_huggingface_to_forge(sd, model_type="qwen3")
        assert "layers.0.attention.q_norm.weight" in result
        assert "layers.0.attention.k_norm.weight" in result


# ── GGUF → Forge mapping ────────────────────────────────────────────────────


class TestGGUFMapping:
    """Test GGUF → Forge weight mapping."""

    def test_gguf_basic_mapping(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "token_embd.weight": _make_dummy_tensor(),
            "output.weight": _make_dummy_tensor(),
            "blk.0.attn_q.weight": _make_dummy_tensor(),
            "blk.0.attn_k.weight": _make_dummy_tensor(),
            "blk.0.attn_v.weight": _make_dummy_tensor(),
            "blk.0.attn_output.weight": _make_dummy_tensor(),
            "blk.0.ffn_gate.weight": _make_dummy_tensor(),
            "blk.0.ffn_down.weight": _make_dummy_tensor(),
            "blk.0.ffn_up.weight": _make_dummy_tensor(),
            "blk.0.attn_norm.weight": _make_dummy_tensor(),
            "blk.0.ffn_norm.weight": _make_dummy_tensor(),
            "output_norm.weight": _make_dummy_tensor(),
        }
        result = mapper.map_gguf_to_forge(sd)
        # identity, not presence (see TestHuggingFaceLlamaMapping): the
        # attention/FFN rows are where a table swap would scramble a model
        assert result["tok_embeddings.weight"] is sd["token_embd.weight"]
        assert result["output.weight"] is sd["output.weight"]
        assert result["layers.0.attention.wq.weight"] is sd["blk.0.attn_q.weight"]
        assert result["layers.0.attention.wk.weight"] is sd["blk.0.attn_k.weight"]
        assert result["layers.0.attention.wv.weight"] is sd["blk.0.attn_v.weight"]
        assert result["layers.0.attention.wo.weight"] is sd["blk.0.attn_output.weight"]
        assert result["layers.0.feed_forward.w1.weight"] is sd["blk.0.ffn_gate.weight"]
        assert result["layers.0.feed_forward.w2.weight"] is sd["blk.0.ffn_down.weight"]
        assert result["layers.0.feed_forward.w3.weight"] is sd["blk.0.ffn_up.weight"]
        assert result["norm.weight"] is sd["output_norm.weight"]

    def test_gguf_multi_block(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "blk.7.attn_q.weight": _make_dummy_tensor(),
        }
        result = mapper.map_gguf_to_forge(sd)
        assert "layers.7.attention.wq.weight" in result


# ── Auto-detection ───────────────────────────────────────────────────────────


class TestAutoDetection:
    """Test _detect_hf_model_type auto-detection."""

    def test_detect_gpt2(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {"transformer.h.0.attn.c_attn.weight": None, "transformer.wte.weight": None}
        assert mapper._detect_hf_model_type(sd) == "gpt2"

    def test_detect_llama(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {"model.layers.0.self_attn.q_proj.weight": None, "model.layers.0.mlp.gate_proj.weight": None}
        assert mapper._detect_hf_model_type(sd) == "llama"

    def test_detect_phi(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {"model.layers.0.self_attn.q_proj.weight": None, "model.layers.0.mlp.fc1.weight": None}
        assert mapper._detect_hf_model_type(sd) == "phi"

    def test_auto_detect_used_when_no_type(self):
        """map_huggingface_to_forge should auto-detect when model_type=None."""
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "model.embed_tokens.weight": _make_dummy_tensor(),
            "model.layers.0.self_attn.q_proj.weight": _make_dummy_tensor(),
            "model.layers.0.mlp.gate_proj.weight": _make_dummy_tensor(),
            "model.norm.weight": _make_dummy_tensor(),
            "lm_head.weight": _make_dummy_tensor(),
        }
        result = mapper.map_huggingface_to_forge(sd, model_type=None)
        assert "tok_embeddings.weight" in result


# ── Error handling ───────────────────────────────────────────────────────────


class TestWeightMappingErrors:
    """Test error handling for bad weight dicts."""

    def test_high_unmapped_ratio_raises(self):
        """More than 10% unmapped weights should raise ValueError."""
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        # All garbage keys — none will match llama patterns
        sd = {f"garbage_key_{i}": _make_dummy_tensor() for i in range(20)}
        with pytest.raises(ValueError, match="unmapped"):
            mapper.map_huggingface_to_forge(sd, model_type="llama")

    def test_empty_dict_returns_empty(self):
        """Empty state dict should return empty result."""
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        result = mapper.map_huggingface_to_forge({}, model_type="llama")
        assert result == {}


# ── Stats tracking ───────────────────────────────────────────────────────────


class TestWeightMapperStats:
    """Test get_stats() reporting."""

    def test_stats_after_mapping(self):
        from enigma_engine.core.weight_mapping import WeightMapper

        mapper = WeightMapper()
        sd = {
            "model.embed_tokens.weight": _make_dummy_tensor(),
            "lm_head.weight": _make_dummy_tensor(),
        }
        mapper.map_huggingface_to_forge(sd, model_type="llama")
        stats = mapper.get_stats()
        assert stats["mapped"] == 2
        assert stats["skipped"] == 0
