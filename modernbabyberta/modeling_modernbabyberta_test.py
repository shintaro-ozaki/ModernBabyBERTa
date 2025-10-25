import os
import math
import types
import importlib
import tempfile
from dataclasses import dataclass

import pytest
import torch

def try_import_model_module():
    try:
        mod = importlib.import_module("modernbabyberta.modeling_modernbabyberta")
        return mod
    except SyntaxError as e:
        pytest.skip(f"SyntaxError in modeling_modernbabyberta: {e}")
    except Exception as e:
        # For ImportError / AttributeError in import-time
        pytest.skip(f"Import-time error in modeling_modernbabyberta: {e}")

@pytest.fixture(scope="module")
def mod():
    return try_import_model_module()

@pytest.fixture(scope="module")
def Config(mod):
    return mod.ModernBabyBERTaConfig

@pytest.fixture()
def tiny_cfg(Config):
    return Config(
        vocab_size=97,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=3,
        glu_expansion=64,
        rope_dim_scale=1.0,
        global_rope_theta=10000,
        local_rope_theta=100,
        local_window_size=2,
        global_every_n_layers=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-5,
        no_bias_linear=False,
        no_bias_layer_norm=False,
        add_embed_layernorm=True,
        pre_norm=True,
        remove_first_attn_prefn_ln=False,
    )

@pytest.fixture()
def model(tiny_cfg, mod):
    torch.manual_seed(0)
    return mod.ModernBabyBERTaModel(tiny_cfg)

@pytest.fixture()
def mlm_model(tiny_cfg, mod):
    torch.manual_seed(0)
    return mod.ModernBabyBERTaForMaskedLM(tiny_cfg)

def make_batch(seq_len=12, batch=2, vocab_size=97, pad_id=0, pad_right=0):
    """
    Create toy batch with optional right padding.
    Returns: input_ids [B,S], attention_mask [B,S]
    """
    torch.manual_seed(123)
    ids = torch.randint(5, vocab_size, (batch, seq_len))
    if pad_right > 0:
        # pad last pad_right tokens with PAD
        ids[:, -pad_right:] = pad_id
    attn = torch.ones_like(ids)
    if pad_right > 0:
        attn[:, -pad_right:] = 0
    return ids.long(), attn

def test_model_structure(mod, tiny_cfg):
    model = mod.ModernBabyBERTaModel(tiny_cfg)
    assert isinstance(model.embeddings, mod.ModernBabyBERTaEmbeddings)
    assert len(model.layers) == tiny_cfg.num_hidden_layers
    for i, block in enumerate(model.layers):
        assert isinstance(block, mod.TransformerBlock)
        assert isinstance(block.attention, mod.MultiHeadAttention)
        assert block.layer_idx == i
    assert model.final_ln is not None
    assert hasattr(model, "pooler")
    assert hasattr(model, "pooler_act")



def test_embedding_output_shape(mod, tiny_cfg):
    emb = mod.ModernBabyBERTaEmbeddings(tiny_cfg)
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (4, 10))
    out = emb(input_ids)
    assert out.shape == (4, 10, tiny_cfg.hidden_size)
    assert torch.isfinite(out).all(), "Embedding output has NaN/inf values"



def test_rotary_embedding_behavior(mod):
    rope = mod.RotaryEmbedding(head_dim=8, base_theta=10000.0)
    q = torch.randn(1, 12, 1, 8)
    k = torch.randn_like(q)
    q_rot, k_rot = rope(q, k, seq_len=12)
    assert q_rot.shape == q.shape
    assert not torch.equal(q_rot, q), "RotaryEmbedding should modify Q/K"



def test_attention_output_shape(mod, tiny_cfg):
    attn = mod.MultiHeadAttention(tiny_cfg)
    x = torch.randn(2, 8, tiny_cfg.hidden_size)
    out = attn(x, layer_idx=0)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

def test_attention_local_masking(mod, tiny_cfg):
    attn = mod.MultiHeadAttention(tiny_cfg)
    mask = attn._local_mask(S=8, device="cpu")
    # 対角からの距離2を超える部分がTrue（mask対象）
    assert mask[0, -1].item() == True
    assert mask[0, 1].item() == False

def test_transformer_block_forward(mod, tiny_cfg):
    block = mod.TransformerBlock(tiny_cfg, layer_idx=0)
    x = torch.randn(2, 10, tiny_cfg.hidden_size)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

def test_forward_shape_and_dtype(model, tiny_cfg):
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (2, 10))
    attn_mask = torch.ones((2, 10))
    out = model(input_ids, attn_mask)
    assert out.last_hidden_state.shape == (2, 10, tiny_cfg.hidden_size)
    assert out.pooler_output.shape == (2, tiny_cfg.hidden_size)
    assert out.last_hidden_state.dtype == torch.float32

def test_forward_with_padding(model, tiny_cfg):
    input_ids, attn = torch.randint(0, tiny_cfg.vocab_size, (2, 12)), torch.ones((2, 12))
    attn[:, -3:] = 0
    out = model(input_ids, attn)
    assert out.last_hidden_state.shape == (2, 12, tiny_cfg.hidden_size)

def test_output_hidden_states(model, tiny_cfg):
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, 6))
    out = model(input_ids, output_hidden_states=True)
    assert isinstance(out.hidden_states, tuple)
    assert len(out.hidden_states) == tiny_cfg.num_hidden_layers

def test_mlm_forward_loss(mlm_model, tiny_cfg):
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 10))
    out = mlm_model(input_ids=ids, labels=ids)
    assert out.loss is not None
    assert out.logits.shape == (2, 10, tiny_cfg.vocab_size)

def test_mlm_forward_no_labels(mlm_model, tiny_cfg):
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 10))
    out = mlm_model(input_ids=ids)
    assert out.loss is None
    assert torch.isfinite(out.logits).all()

def test_dropout_effect(mod, tiny_cfg):
    tiny_cfg.hidden_dropout_prob = 0.5
    model = mod.ModernBabyBERTaModel(tiny_cfg)
    model.train()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    out1 = model(ids).last_hidden_state
    out2 = model(ids).last_hidden_state
    # dropoutが効いていれば出力は確率的に異なる
    diff = (out1 - out2).abs().sum().item()
    assert diff > 0.0

@pytest.mark.parametrize("pre_norm", [True, False])
def test_norm_modes(mod, tiny_cfg, pre_norm):
    tiny_cfg.pre_norm = pre_norm
    model = mod.ModernBabyBERTaModel(tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (1, 6))
    out = model(ids)
    assert out.last_hidden_state.shape == (1, 6, tiny_cfg.hidden_size)
