from transformers.configuration_utils import PretrainedConfig

class ModernBabyBERTaConfig(PretrainedConfig):
    model_type = "modernbabyberta"

    def __init__(
        self,
        vocab_size=8192,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        glu_expansion=512,
        max_position_embeddings=2048,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        layer_norm_eps=1e-5,
        pre_norm=True,
        rope_dim_scale=1.0,
        global_rope_theta=160000,
        local_rope_theta=10000,
        local_window_size=128,
        global_every_n_layers=3,
        no_bias_linear=True,
        no_bias_layer_norm=True,
        add_embed_layernorm=True,
        remove_first_attn_prefn_ln=True,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.glu_expansion = glu_expansion
        self.max_position_embeddings = max_position_embeddings
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.pre_norm = pre_norm
        self.rope_dim_scale = rope_dim_scale
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta
        self.local_window_size = local_window_size
        self.global_every_n_layers = global_every_n_layers
        self.no_bias_linear = no_bias_linear
        self.no_bias_layer_norm = no_bias_layer_norm
        self.add_embed_layernorm = add_embed_layernorm
        self.remove_first_attn_prefn_ln = remove_first_attn_prefn_ln
