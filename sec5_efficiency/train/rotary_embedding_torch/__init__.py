from rotary_embedding_torch.rotary_embedding_torch import (
    apply_rotary_emb,
    RotaryEmbedding,
    apply_learned_rotations,
    broadcat
)
from rotary_embedding_torch.rotary_new import apply_rotary_emb as qwen2_apply_rotary_emb
from rotary_embedding_torch.rotary_new import RotaryEmbedding as Qwen2RotaryEmbedding
from rotary_embedding_torch.rotary_b1 import ChannelPhaseRoPE