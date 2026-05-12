"""StreamQwen3_5Moe configuration — extends Qwen3_5MoeTextConfig with stream-specific attributes."""

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class StreamQwen3_5MoeTextConfig(Qwen3_5MoeTextConfig):
    """Qwen3_5MoeTextConfig extended with num_channels, channel_embedding_method, and role gating.

    Loads from the text_config inside any Qwen3.5-MoE config.json.
    Stream-specific attributes are set dynamically by the training script.
    """

    model_type = "qwen3_5_moe_text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
