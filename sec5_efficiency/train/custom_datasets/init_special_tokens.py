"""
Initialize special tokens for parallel multi-channel generation

This module provides utilities to add and initialize communication tokens
and chunk markers with proper embedding initialization.

Usage:
    from init_special_tokens import add_and_init_communication_tokens

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    add_and_init_communication_tokens(model, tokenizer,
                                      max_channels=10,
                                      max_chunks=20)
"""

import torch
from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


def add_and_init_communication_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_channels: int = 10,
    max_chunks: int = 20,
    custom_tokens: Optional[List[str]] = None
):
    """
    Adds communication and chunk marker tokens to tokenizer and initializes embeddings.

    Args:
        model: The model to update
        tokenizer: The tokenizer to update
        max_channels: Maximum number of channels to support (default: 10)
        max_chunks: Maximum number of chunks to support (default: 20)
        custom_tokens: Optional list of additional custom tokens

    Returns:
        tuple: (updated_model, updated_tokenizer, added_tokens_dict)
    """

    # Define all special tokens
    communication_tokens = []

    # 1. Channel communication tokens
    for i in range(max_channels):
        communication_tokens.extend([
            f"<to:channel_{i}>",
            # f"</to:channel_{i}>",
        ])

    # 2. General communication tokens
    communication_tokens.extend([
        "<to:all>",
        # "</to:all>",
        # "<to>",
        "</to>",
    ])

    # 3. Chunk marker tokens
    for i in range(max_chunks):
        communication_tokens.extend([
            f"<chunk_{i}>",
            f"</chunk_{i}>",
        ])

    # 4. Generic chunk markers
    communication_tokens.extend([
        "<chunk>",
        "</chunk>",
    ])

    # 5. Add custom tokens if provided
    if custom_tokens:
        communication_tokens.extend(custom_tokens)

    # Check which tokens need to be added
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [tok for tok in communication_tokens if tok not in existing_vocab]

    if not new_tokens:
        print("✓ All tokens already exist in vocabulary")
        return model, tokenizer, {}

    print(f"Adding {len(new_tokens)} new tokens to tokenizer and model...")

    # Add tokens to tokenizer
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"  Added {num_added} tokens to tokenizer")

    # Resize model embeddings
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    new_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"  Resized embeddings: {original_vocab_size} -> {new_vocab_size}")

    # Initialize new token embeddings
    print("  Initializing new token embeddings...")
    added_tokens_info = initialize_token_embeddings(model, tokenizer, new_tokens)

    print(f"✓ Successfully added and initialized {len(new_tokens)} tokens")

    return model, tokenizer, added_tokens_info


def initialize_token_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    new_tokens: List[str]
) -> dict:
    """
    Initialize embeddings for new tokens using semantic base words.

    Strategy:
    1. Extract base word from token (e.g., "channel" from "<to:channel_0>")
    2. Tokenize base word and get its embeddings
    3. Average embeddings to initialize new token
    4. Apply to both input embeddings and output embeddings (lm_head)

    Args:
        model: The model with resized embeddings
        tokenizer: The tokenizer with new tokens
        new_tokens: List of newly added token strings

    Returns:
        dict: Information about initialized tokens
    """
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    # Check if embeddings are tied
    tied = embed.weight.data_ptr() == lm_head.weight.data_ptr()

    added_tokens_info = {}
    initialized_count = 0
    fallback_count = 0

    for tok in new_tokens:
        # Extract base word for semantic initialization
        base_word = extract_base_word(tok)

        # Tokenize base word
        base_ids = tokenizer(base_word, add_special_tokens=False).input_ids

        # Filter out unknown tokens
        valid_ids = [i for i in base_ids if i != tokenizer.unk_token_id]

        if valid_ids:
            # Use average of base word embeddings
            with torch.no_grad():
                avg_embed = embed(torch.tensor(valid_ids, device=model.device)).mean(dim=0)

                special_id = tokenizer.convert_tokens_to_ids(tok)
                embed.weight.data[special_id] = avg_embed

                # Also initialize lm_head if not tied
                if not tied and lm_head.weight.shape[0] == embed.weight.shape[0]:
                    avg_lm_logits = lm_head.weight.data[valid_ids].mean(dim=0)
                    lm_head.weight.data[special_id] = avg_lm_logits.clone()

            initialized_count += 1
            added_tokens_info[tok] = {
                'base_word': base_word,
                'base_tokens': [tokenizer.convert_ids_to_tokens(i) for i in valid_ids],
                'method': 'semantic_averaging'
            }
        else:
            # Fallback: use random initialization (already done by resize)
            fallback_count += 1
            added_tokens_info[tok] = {
                'base_word': base_word,
                'method': 'random_fallback'
            }
            print(f"    Warning: No valid base tokens for '{tok}', using random initialization")

    print(f"    Initialized: {initialized_count} semantic, {fallback_count} random")

    return added_tokens_info


def extract_base_word(token: str) -> str:
    """
    Extract semantic base word from a special token.

    Examples:
        "<to:channel_0>" -> "channel"
        "</to:channel_1>" -> "channel"
        "<chunk_5>" -> "chunk"
        "</chunk>" -> "chunk"
        "<to:all>" -> "all"
        "<to>" -> "to"

    Args:
        token: Special token string

    Returns:
        str: Base word for semantic initialization
    """
    # Remove angle brackets
    token = token.strip('<>')

    # Handle different patterns
    if 'channel' in token:
        return 'channel'
    elif 'chunk' in token:
        return 'chunk'
    elif token.startswith('to:'):
        # Extract the target word after "to:"
        parts = token.split(':')
        if len(parts) > 1:
            target = parts[1].split('_')[0]  # Remove numbers
            return target if target else 'to'
        return 'to'
    elif token.startswith('/'):
        # Closing tag
        return token[1:].split('_')[0]
    else:
        # Generic case: use first word before underscore or number
        base = token.split('_')[0].split(':')[-1]
        return base if base else token


def save_tokenizer_with_tokens(tokenizer: PreTrainedTokenizer, output_path: str):
    """
    Save tokenizer with all added tokens.

    Args:
        tokenizer: Tokenizer to save
        output_path: Path to save directory
    """
    import os
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Tokenizer saved to: {output_path}")


def save_model_with_tokens(model: PreTrainedModel, output_path: str):
    """
    Save model with resized embeddings.

    Args:
        model: Model to save
        output_path: Path to save directory
    """
    import os
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    print(f"✓ Model saved to: {output_path}")


def verify_token_initialization(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sample_tokens: Optional[List[str]] = None
):
    """
    Verify that tokens are properly initialized by checking embedding norms.

    Args:
        model: Model to verify
        tokenizer: Tokenizer to verify
        sample_tokens: Optional list of tokens to check (default: first few new tokens)
    """
    print("\n" + "="*60)
    print("Verifying Token Initialization")
    print("="*60)

    if sample_tokens is None:
        # Check first few communication tokens
        sample_tokens = [
            "<to:channel_0>", "</to>", "<chunk_0>", "</chunk_0>", "<to:all>"
        ]

    embed = model.get_input_embeddings()

    for tok in sample_tokens:
        if tok in tokenizer.get_vocab():
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            embedding = embed.weight.data[tok_id]
            norm = embedding.norm().item()

            # Check if embedding is not zero (properly initialized)
            if norm > 0:
                print(f"  ✓ {tok:20s}: ID={tok_id:6d}, norm={norm:.4f}")
            else:
                print(f"  ✗ {tok:20s}: ID={tok_id:6d}, norm={norm:.4f} (NOT INITIALIZED!)")
        else:
            print(f"  ✗ {tok:20s}: NOT IN VOCABULARY")

    print("="*60)


# Example usage in main
def main():
    """Example usage"""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Add and initialize communication tokens")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer (can be same as model)")
    parser.add_argument("--output", required=True, help="Output path for model and tokenizer")
    parser.add_argument("--max-channels", type=int, default=10, help="Max channels")
    parser.add_argument("--max-chunks", type=int, default=20, help="Max chunks")
    parser.add_argument("--verify", action="store_true", help="Verify initialization")

    args = parser.parse_args()

    print("="*60)
    print("Adding Communication Tokens to Model")
    print("="*60)

    # Load model and tokenizer
    print(f"\nLoading model from: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)

    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Add and initialize tokens
    print(f"\nAdding tokens (max_channels={args.max_channels}, max_chunks={args.max_chunks})")
    model, tokenizer, added_info = add_and_init_communication_tokens(
        model, tokenizer,
        max_channels=args.max_channels,
        max_chunks=args.max_chunks
    )

    # Verify if requested
    if args.verify:
        verify_token_initialization(model, tokenizer)

    # Save
    print(f"\nSaving to: {args.output}")
    save_model_with_tokens(model, args.output)
    save_tokenizer_with_tokens(tokenizer, args.output)

    print("\n" + "="*60)
    print("✓ Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. The model now has {len(tokenizer)} tokens")
    print(f"  2. Use this model for training with parallel chunk data")
    print(f"  3. The embeddings are initialized semantically from base words")


if __name__ == "__main__":
    main()
