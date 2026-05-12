import torch
import torch.nn.functional as F


class ExternalSampler:
    @staticmethod
    def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature is None or temperature == 1.0:
            return logits
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        return logits / temperature

    @staticmethod
    def apply_top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Top-k
        """
        if top_k <= 0:
            return logits

        top_k = min(max(top_k, 1), logits.size(-1))  # Safety check

        # Get top-k values and indices
        top_k_scores, top_k_indices = torch.topk(logits, top_k, dim=-1, largest=True, sorted=False)

        # Create a mask for values to keep
        indices_to_remove = logits < torch.gather(top_k_scores, -1, top_k_indices[..., -1, None])
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

        return logits

    @staticmethod
    def apply_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Top-p (nucleus)
        """
        if top_p <= 0 or top_p >= 1.0:
            return logits

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep at least 1 token
        sorted_indices_to_remove[..., 0] = False

        # Set logits to -inf for tokens to remove
        sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))

        # Scatter back to original indexing
        logits = logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)

        return logits

    @staticmethod
    def apply_min_p_filtering(logits: torch.Tensor, min_p: float) -> torch.Tensor:
        """
        Min-p
        """
        if min_p <= 0 or min_p > 1.0:
            return logits

        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]
        threshold = min_p * max_prob

        indices_to_remove = probs < threshold
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

        return logits

    @staticmethod
    def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        """
        transformers.generation.utils.RepetitionPenaltyLogitsProcessor
        """
        if penalty == 1.0:
            return logits

        batch_size, vocab_size = logits.shape

        for batch_idx in range(batch_size):
            for token_id in set(input_ids[batch_idx].tolist()):
                # If score < 0, multiply by penalty; if score >= 0, divide by penalty
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= penalty
                else:
                    logits[batch_idx, token_id] /= penalty

        return logits

    @staticmethod
    def apply_length_penalty(logits: torch.Tensor, eos_token_id: int, cur_len: int, min_length: int) -> torch.Tensor:
        if cur_len < min_length:
            logits[:, eos_token_id] = float("-inf")
        return logits


def sample_tokens_like_transformers(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    min_length: int = 0,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    **kwargs,
) -> torch.Tensor:

    if logits.dtype != torch.float32:
        logits = logits.float()

    scores = logits.clone()

    if repetition_penalty != 1.0:
        scores = ExternalSampler.apply_repetition_penalty(scores, input_ids, repetition_penalty)

    if min_length > 0 and eos_token_id is not None:
        cur_len = input_ids.shape[-1]
        scores = ExternalSampler.apply_length_penalty(scores, eos_token_id, cur_len, min_length)

    if do_sample and temperature != 1.0:
        scores = ExternalSampler.apply_temperature(scores, temperature)

    if do_sample and top_k > 0:
        scores = ExternalSampler.apply_top_k_filtering(scores, top_k)

    if do_sample and 0 < top_p < 1.0:
        scores = ExternalSampler.apply_top_p_filtering(scores, top_p)

    if do_sample and min_p > 0:
        scores = ExternalSampler.apply_min_p_filtering(scores, min_p)

    if do_sample:
        probs = F.softmax(scores, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / probs.size(-1)

        try:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        except RuntimeError as e:
            print(f"Sampling failed: {e}, falling back to argmax")
            next_tokens = scores.argmax(dim=-1)
    else:
        next_tokens = scores.argmax(dim=-1)

    return next_tokens


if __name__ == "__main__":
    pass
