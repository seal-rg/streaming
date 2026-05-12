"""StreamQwen3_5 configuration — extends Qwen3_5TextConfig with stream-specific attributes."""

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class StreamQwen3_5TextConfig(Qwen3_5TextConfig):
    """Qwen3_5TextConfig extended with num_channels, channel_embedding_method, and role gating.

    Loads from the text_config inside any Qwen3.5 config.json.
    Stream-specific attributes are set dynamically by the training script.
    """

    model_type = "qwen3_5_text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
