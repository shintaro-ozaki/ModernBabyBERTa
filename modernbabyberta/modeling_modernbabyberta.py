import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput
from modernbabyberta.configuration_modernbabyberta import ModernBabyBERTaConfig

def _make_linear(input_dim, output_dim, bias: bool):
    return nn.Linear(input_dim, output_dim, bias=bias)

def _make_layernorm(dim, eps, affine: bool):
    """Layer normalization formula:
    (x - mean) / sqrt(var + eps) * gamma + beta
    """
    return nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base_theta: float, rot_dim_ratio: float = 1.0):
        super().__init__()
        self.head_dim = head_dim
        self.rot_dim = int(head_dim * rot_dim_ratio)
        self.base = base_theta

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        device = q.device
        dtype = q.dtype
        rot_dim = self.rot_dim
        if rot_dim % 2 == 1:
            rot_dim -= 1

        # [S, rot_dim/2]
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = 1.0 / (self.base ** (torch.arange(0, rot_dim, 2, device=device, dtype=dtype) / rot_dim))
        ang = torch.einsum("s,d->sd", t, freqs)

        cos = torch.cos(ang)[None, :, None, :]
        sin = torch.sin(ang)[None, :, None, :]

        def apply_rot(x):
            x1 = x[..., :rot_dim]
            even, odd = x1[..., ::2], x1[..., 1::2]
            x1_rot_even = even * cos - odd * sin
            x1_rot_odd  = odd * cos + even * sin
            x1_rot = torch.stack([x1_rot_even, x1_rot_odd], dim=-1).reshape_as(x1)
            return torch.cat([x1_rot, x[..., rot_dim:]], dim=-1)

        return apply_rot(q), apply_rot(k)

class GeGLU(nn.Module):
    """GeGLU (Gated GELU):
    out = (W_1x + b_1) * (W_2x+b_2)
    See: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, dim_in, dim_out, bias=False):
        super().__init__()
        # This linear layer can becaome linear1 layer.
        self.proj = _make_linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * torch.nn.functional.gelu(b)


class MLP(nn.Module):
    def __init__(self, config: ModernBabyBERTaConfig):
        super().__init__()
        bias = not config.no_bias_linear
        self.activation = GeGLU(config.hidden_size, config.glu_expansion // 2, bias=bias)
        self.linear2 = nn.Linear(config.glu_expansion // 2, config.hidden_size, bias=bias)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        # GeGLU uses a gated path; simple variant here:
        # Use self.activation over x, then project back
        # GeGLU doesn't use linear1 since GeGLU itself contains linear1 function.
        h = self.activation(x)
        h = self.drop(h)
        h = self.linear2(h)
        return h

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModernBabyBERTaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        assert self.head_dim * self.num_heads == config.hidden_size

        bias = not config.no_bias_linear
        self.qkv = _make_linear(config.hidden_size, 3 * config.hidden_size, bias=bias)
        self.out = _make_linear(config.hidden_size, config.hidden_size, bias=bias)

        self.rope_global = RotaryEmbedding(self.head_dim, config.global_rope_theta, config.rope_dim_scale)
        self.rope_local  = RotaryEmbedding(self.head_dim, config.local_rope_theta,  config.rope_dim_scale)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.local_window = config.local_window_size
        self.global_every = config.global_every_n_layers

    def _split_heads(self, x):
        # [B, S, D] to [B, S, H, D_head]
        B, S, D = x.shape
        return x.reshape(B, S, self.num_heads, self.head_dim)

    def _merge_heads(self, x):
        # [B, S, H, D_head] to [B, S, H*D_head]
        B, S, H, D = x.shape
        return x.reshape(B, S, H * D)

    def _local_mask(self, S: int, device) -> torch.Tensor:
        idxs = torch.arange(S, device=device)
        dist = (idxs[None, :] - idxs[:, None]).abs()
        mask = dist > self.local_window
        return mask  # [S, S] (True = mask)

    def forward(self, x, layer_idx: int, attention_mask: Optional[torch.Tensor] = None):
        B, S, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(D, dim=-1)
        q = self._split_heads(q)  # [B, S, H, D_head]
        k = self._split_heads(k)
        v = self._split_heads(v)

        is_global = (layer_idx % self.global_every) == 0
        # RoPEは角度が大きいほど、long sequenceに対応できる。
        if is_global:
            q, k = self.rope_global(q, k, seq_len=S)
        else:
            q, k = self.rope_local(q, k, seq_len=S)

        attn_scores = torch.einsum("bshd,bthd->bhst", q, k) / math.sqrt(self.head_dim)

        if not is_global:
            local_mask = self._local_mask(S, x.device)[None, None, :, :]  # [1,1,S,S]
            attn_scores = attn_scores.masked_fill(local_mask, float("-inf"))

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhst,bthd->bshd", attn, v)
        out = self._merge_heads(out)  # [B, S, H*D]
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModernBabyBERTaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        ln_affine = not config.no_bias_layer_norm
        self.layernorm1 = _make_layernorm(dim=config.hidden_size, eps=config.layer_norm_eps, affine=ln_affine)
        self.attention = MultiHeadAttention(config)
        self.layernorm2 = _make_layernorm(dim=config.hidden_size, eps=config.layer_norm_eps, affine=ln_affine)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, attention_mask=None, is_first_layer=False):
        if  self.config.pre_norm:
            # Basically, ModernBERT adopts Pre-Ln which this time we follow, and we need to avoid ln function two times.
            # NOTE: Our model adopts Pre-LN.
            h = x if (is_first_layer and self.config.remove_first_attn_prefn_ln) else self.layernorm1(x)
            x = x + self.dropout(self.attention(h, self.layer_idx, attention_mask=attention_mask))
            x = x + self.dropout(self.mlp(self.layernorm2(x)))
        else:
            # NOTE: Here is for Post-LN which is primarily for a research purpose.
            x = self.layernorm1(x + self.dropout(self.attention(x, self.layer_idx, attention_mask=attention_mask)))
            x = self.layernorm2(x + self.dropout(self.mlp(x)))
        return x


class ModernBabyBERTaEmbeddings(nn.Module):
    def __init__(self, config: ModernBabyBERTaConfig):
        super().__init__()
        bias = not config.no_bias_linear
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = None     # Insteadly, we adopt RoPE.
        ln_affine = not config.no_bias_layer_norm
        if config.add_embed_layernorm:
            self.emb_ln = _make_layernorm(dim=config.hidden_size, eps=config.layer_norm_eps, affine=ln_affine)
        else:
            self.emb_ln = nn.Identity()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        x = self.word_embeddings(input_ids)
        x = self.emb_ln(x)
        x = self.dropout(x)
        return x

class ModernBabyBERTaModel(PreTrainedModel):
    config_class = ModernBabyBERTaConfig
    base_model_prefix = "modernbabyberta"

    def __init__(self, config:ModernBabyBERTaConfig):
        super().__init__(config)
        self.embeddings = ModernBabyBERTaEmbeddings(config)
        self.layers =nn.ModuleList([TransformerBlock(config=config, layer_idx=i) for i in range(config.num_hidden_layers)])
        ln_affine = not config.no_bias_layer_norm
        self.final_ln = _make_layernorm(dim=config.hidden_size, eps=config.layer_norm_eps, affine=ln_affine)

        self.pooler = nn.Linear(config.hidden_size, config.hidden_size, bias=not config.no_bias_linear)
        self.pooler_act = nn.Tanh()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        # attention_mask: [B,S] -> [B,1,1,S]
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=self.dtype)) * -1e4

        hidden_states = self.embeddings(input_ids)
        all_hidden = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                is_first_layer=True if i==0 else False,
            )
            if output_hidden_states:
                all_hidden.append(hidden_states)

        hidden_states = self.final_ln(hidden_states)
        pooled = self.pooler_act(self.pooler(hidden_states[:, 0]))

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=tuple(all_hidden) if output_hidden_states else None,
        )

class ModernBabyBERTaForMaskedLM(PreTrainedModel):
    config_class = ModernBabyBERTaConfig

    def __init__(self, config):
        super().__init__(config)
        self.modernbabyberta = ModernBabyBERTaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.post_init()

    def forward(self, input_ids, labels=None, attention_mask=None):
        outputs = self.modernbabyberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )


if __name__ == "__main__":
    config = ModernBabyBERTaConfig()
    model = ModernBabyBERTaForMaskedLM(config)

    # model.configの出力とarchitectureの動作確認
    print("Model config:", model.config)
    print("Model architecture:", model)

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    attention_mask = torch.ones((2, 16))

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    print("Logits shape:", outputs.logits.shape)
    print("Loss:", outputs.loss)
