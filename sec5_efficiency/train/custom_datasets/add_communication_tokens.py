"""
Add communication tokens to tokenizer for multi-channel parallel generation

This script extends a tokenizer with special tokens used in parallel chunk-based generation:
- <to:channel_X> / </to> for directed communication
- <to:all> / </to:all> for broadcast communication

Usage:
    python add_communication_tokens.py --tokenizer /path/to/tokenizer --output /path/to/output --max-channels 10
"""

import argparse
from transformers import AutoTokenizer
import json
import os


def add_communication_tokens(tokenizer_path: str, output_path: str, max_channels: int = 10, save: bool = True):
    """
    Add communication tokens to a tokenizer

    Args:
        tokenizer_path: Path to the original tokenizer
        output_path: Path to save the extended tokenizer
        max_channels: Maximum number of channels to support (adds tokens for 0 to max_channels-1)
        save: Whether to save the tokenizer to disk

    Returns:
        tokenizer: The extended tokenizer
        num_added: Number of tokens added
    """
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")

    # Define communication tokens
    comm_tokens = []

    # Add tokens for each channel
    for i in range(max_channels):
        comm_tokens.extend([
            f"<to:channel_{i}>",
            f"</to:channel_{i}>",
        ])

    # Add general communication tokens
    comm_tokens.extend([
        "<to:all>",
        "</to:all>",
        "<to>",
        "</to>",
    ])

    # Add chunk marker tokens (for chunk identification)
    for i in range(20):  # Support up to 20 chunks
        comm_tokens.extend([
            f"<chunk_{i}>",
            f"</chunk_{i}>",
        ])
    # Generic chunk markers
    comm_tokens.extend([
        "<chunk>",
        "</chunk>",
    ])

    # Check which tokens are not in vocabulary
    new_tokens = []
    existing_tokens = []
    vocab = tokenizer.get_vocab()

    for token in comm_tokens:
        if token not in vocab:
            new_tokens.append(token)
        else:
            existing_tokens.append(token)

    if existing_tokens:
        print(f"\n{len(existing_tokens)} tokens already exist in vocabulary:")
        for token in existing_tokens[:5]:  # Show first 5
            print(f"  - {token}")
        if len(existing_tokens) > 5:
            print(f"  ... and {len(existing_tokens) - 5} more")

    if new_tokens:
        print(f"\nAdding {len(new_tokens)} new communication tokens...")
        num_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
        print(f"Successfully added {num_added} tokens")

        # Show some examples
        print("\nExample tokens added:")
        for token in new_tokens[:5]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {token}: {token_id}")
        if len(new_tokens) > 5:
            print(f"  ... and {len(new_tokens) - 5} more")
    else:
        print("\nAll communication tokens already exist in vocabulary")
        num_added = 0

    new_vocab_size = len(tokenizer)
    print(f"\nFinal vocabulary size: {new_vocab_size} (added {new_vocab_size - original_vocab_size} tokens)")

    if save and num_added > 0:
        print(f"\nSaving extended tokenizer to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        tokenizer.save_pretrained(output_path)

        # Save metadata
        metadata = {
            'original_tokenizer': tokenizer_path,
            'original_vocab_size': original_vocab_size,
            'new_vocab_size': new_vocab_size,
            'tokens_added': num_added,
            'max_channels': max_channels,
            'communication_tokens': comm_tokens
        }

        metadata_path = os.path.join(output_path, 'communication_tokens_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to: {metadata_path}")
        print("\nIMPORTANT: You need to:")
        print("  1. Resize model token embeddings to match new vocabulary size")
        print("  2. Initialize new token embeddings (e.g., copy from similar tokens or random init)")
        print("  3. Fine-tune the model with the new tokens")

    return tokenizer, num_added


def verify_tokens(tokenizer, max_channels: int = 10):
    """Verify that communication tokens are properly added"""
    print("\n" + "="*60)
    print("Verifying communication tokens...")
    print("="*60)

    test_tokens = [
        "<to:channel_0>",
        "</to>",
        "<to:all>",
        "</to:all>",
        f"<to:channel_{max_channels-1}>",
    ]

    all_valid = True
    for token in test_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        decoded = tokenizer.decode([token_id])

        # Check if token was properly added (not UNK)
        unk_id = tokenizer.unk_token_id
        if token_id == unk_id:
            print(f"❌ {token}: FAILED (maps to UNK)")
            all_valid = False
        else:
            print(f"✓ {token}: ID={token_id}, decodes to '{decoded}'")

    if all_valid:
        print("\n✅ All communication tokens verified successfully!")
    else:
        print("\n⚠️  Some tokens failed verification")

    return all_valid


def test_encoding(tokenizer):
    """Test encoding a sample conversation with communication tokens"""
    print("\n" + "="*60)
    print("Testing encoding with communication tokens...")
    print("="*60)

    # Sample text with communication tokens
    test_text = """<|im_start|>assistant
Hello! <to:channel_1>Please handle the math part.</to> I'll work on the logic.<|im_end|>"""

    print(f"Input text:\n{test_text}\n")

    # Encode
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"Token IDs: {tokens}")

    # Decode
    decoded = tokenizer.decode(tokens)
    print(f"\nDecoded text:\n{decoded}\n")

    # Check if roundtrip is successful
    if decoded.strip() == test_text.strip():
        print("✅ Encoding/decoding roundtrip successful!")
    else:
        print("⚠️  Decoded text differs from input")
        print("This might be expected if the tokenizer normalizes whitespace")


def main():
    parser = argparse.ArgumentParser(description="Add communication tokens to tokenizer")
    parser.add_argument("--tokenizer", required=True, help="Path to original tokenizer")
    parser.add_argument("--output", required=True, help="Path to save extended tokenizer")
    parser.add_argument("--max-channels", type=int, default=10,
                       help="Maximum number of channels to support (default: 10)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save the tokenizer (dry run)")
    parser.add_argument("--verify", action="store_true", default=True,
                       help="Verify tokens after adding (default: True)")
    parser.add_argument("--test", action="store_true",
                       help="Run encoding/decoding test")

    args = parser.parse_args()

    # Add tokens
    tokenizer, num_added = add_communication_tokens(
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        max_channels=args.max_channels,
        save=not args.no_save
    )

    # Verify
    if args.verify:
        verify_tokens(tokenizer, args.max_channels)

    # Test
    if args.test:
        test_encoding(tokenizer)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
