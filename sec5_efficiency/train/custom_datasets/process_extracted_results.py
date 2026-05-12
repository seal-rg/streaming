"""
Process extracted_results dataset - handles parallel chunk-based multi-channel data
Usage:
    python process_extracted_results.py --input ./extracted_results --output ./cache --tokenizer /path/to/tokenizer
"""

import torch
import json
import os
import hashlib
import numpy as np
import argparse
import time
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer
from pathlib import Path


class ExtractedResultsProcessor:
    """Processor for extracted_results format with parallel chunks and multiple channels"""

    def __init__(self,
                 tokenizer_path: str,
                 max_seq_length: int,
                 truncation_strategy: str = "random",
                 max_position_embeddings: int = 32678,
                 position_ids_2d: bool = True,
                 system_message: str = "You are a helpful assistant.",
                 add_communication_tokens: bool = True,
                 chunk_alignment_strategy: str = "target",
                 chunk_target_length: int = None,
                 add_chunk_markers: bool = True,
                 analyze_chunk_lengths: bool = True):

        self.tokenizer_path = tokenizer_path
        self.max_seq_length = max_seq_length
        self.truncation_strategy = truncation_strategy
        self.max_position_embeddings = max_position_embeddings
        self.position_ids_2d = position_ids_2d
        self.system_message = system_message
        self.add_communication_tokens = add_communication_tokens
        self.chunk_alignment_strategy = chunk_alignment_strategy
        self.chunk_target_length = chunk_target_length
        self.add_chunk_markers = add_chunk_markers
        self.analyze_chunk_lengths = analyze_chunk_lengths

        print("Mode: Extracted Results - Parallel Chunks with Multiple Channels")
        print(f"Chunk alignment strategy: {chunk_alignment_strategy}")
        if chunk_target_length:
            print(f"Chunk target length: {chunk_target_length} tokens")
        print(f"Add chunk markers: {add_chunk_markers}")
        print(f"Analyze chunk lengths: {analyze_chunk_lengths}")

        # Initialize tokenizer
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # Setup special tokens
        self._setup_special_tokens()

        # Add communication tokens if needed
        if self.add_communication_tokens:
            self._add_communication_tokens()

    def _setup_special_tokens(self):
        """Pre-compute special tokens"""
        self.im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
        self.newline_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        print(f"Special tokens loaded: assistant_tokens={self.assistant_tokens}, newline_token={self.newline_token}")
        print(f"Special tokens - im_start: {self.im_start}, im_end: {self.im_end}")
        # Precompute chunk marker token ids (only if add_chunk_markers)
        self.chunk_start_ids = {}
        self.chunk_end_ids = {}
        if self.add_chunk_markers:
            for i in range(20):
                self.chunk_start_ids[i] = self.tokenizer.convert_tokens_to_ids(f"<chunk_{i}>")
                self.chunk_end_ids[i] = self.tokenizer.convert_tokens_to_ids(f"</chunk_{i}>")


    def _add_communication_tokens(self):
        """Add communication tokens for channel messaging and chunk markers"""
        # Define communication tokens
        comm_tokens = []

        # Add tokens for channels 0-9 (can extend if needed)
        for i in range(10):
            comm_tokens.extend([
                f"<to:channel_{i}>",
                # f"</to:channel_{i}>",
            ])

        # Add general communication tokens
        comm_tokens.extend([
            "<to:all>",
            # "</to:all>",
            # "<to>",
            "</to>",
        ])

        # Add chunk marker tokens (for chunk identification)
        if self.add_chunk_markers:
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
        for token in comm_tokens:
            if token not in self.tokenizer.get_vocab():
                new_tokens.append(token)

        if new_tokens:
            print(f"Adding {len(new_tokens)} communication tokens to tokenizer")
            num_added = self.tokenizer.add_tokens(new_tokens)
            print(f"Successfully added {num_added} tokens")

            # Note: In production, you would need to save and reload the tokenizer
            # and resize the model's token embeddings
            print("WARNING: Remember to save tokenizer and resize model embeddings!")
        else:
            print("All communication tokens already in vocabulary")

    def save_single_sample(self, processed_sample: Dict, output_dir: str, sample_idx: int):
        """Save a single sample to a separate .npz file"""
        sample_file = os.path.join(output_dir, f"sample_{sample_idx:06d}.npz")

        # Prepare single sample data
        sample_data = {
            'input_ids': np.array(processed_sample['input_ids'], dtype=np.int32),
            'labels': np.array(processed_sample['labels'], dtype=np.int32),
            'position_ids': np.array(processed_sample['position_ids'], dtype=np.int16),
            'attention_mask': processed_sample['attention_mask'].astype(np.float16),
            'num_heads': np.int8(processed_sample['num_heads']),
            'seq_length': np.int16(processed_sample['seq_length'])
        }

        # Save single file
        np.savez_compressed(sample_file, **sample_data)
        return sample_file

    def process_extracted_results_dir(self, input_dir: str, output_dir: str) -> str:
        """
        Process all JSON files in extracted_results directory
        """
        print(f"Processing extracted_results directory: {input_dir}")
        print(f"Output directory: {output_dir}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Find all JSON files (exclude README and combined files)
        json_files = []
        for file in Path(input_dir).glob("*.json"):
            if file.name not in ["README.md", "all_extracted_results.json"]:
                # Only process numbered files like 0.json, 1.json, etc.
                # if file.stem.isdigit() or file.name.startswith("extracted_result_"):
                json_files.append(file)

        json_files.sort()
        print(f"Found {len(json_files)} JSON files to process")

        # Analyze chunk lengths if requested
        if self.analyze_chunk_lengths:
            print("\n" + "="*60)
            print("Step 1: Analyzing chunk lengths...")
            print("="*60)
            self._analyze_chunk_length_distribution(json_files)
            print("\n" + "="*60)
            print("Step 2: Processing data...")
            print("="*60)

        # Generate index file path
        index_file_path = os.path.join(output_dir, "index.json")

        # Process data
        start_time = time.time()
        total_processed = 0
        total_failed = 0
        sample_files = []

        print("Starting processing...")

        for json_file in json_files:
            print(f"\nProcessing: {json_file.name}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    item = json.load(f)

                if 'parsed_response' in item:
                    item = item['parsed_response']
                
                if not item:
                    #print(f"  Warning: Empty item in {json_file.name}, skipping")
                    continue

                # Standardize data item
                standardized = self._standardize_extracted_result(item)
                if not standardized:
                    print(f"  Skipped: Could not standardize {json_file.name}")
                    total_failed += 1
                    continue

                # Process single item
                processed = self._process_single_item(standardized)
                if not processed:
                    print(f"  Failed to process {json_file.name}")
                    total_failed += 1
                    continue

                # Save as separate .npz file
                sample_file = self.save_single_sample(processed, output_dir, total_processed)
                sample_files.append({
                    'file': os.path.basename(sample_file),
                    'sample_idx': total_processed,
                    'seq_length': processed['seq_length'],
                    'num_heads': processed['num_heads'],
                    'source_file': json_file.name
                })

                total_processed += 1
                print(f"  ✓ Processed successfully (seq_len={processed['seq_length']}, heads={processed['num_heads']})")

            except json.JSONDecodeError as e:
                print(f"  Error: Failed to parse {json_file.name}: {e}")
                total_failed += 1
                continue
            except Exception as e:
                print(f"  Error processing {json_file.name}: {e}")
                import traceback
                traceback.print_exc()
                total_failed += 1
                continue

        if total_processed == 0:
            raise ValueError("No valid processed data")

        print(f"\n{'='*60}")
        print(f"Processing completed: {total_processed} items processed, {total_failed} failed")
        print(f"{'='*60}")

        # Save index file
        self._save_index_file(index_file_path, input_dir, sample_files)

        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")

        # Display statistics
        self._print_statistics(total_processed, output_dir)

        return output_dir

    def _analyze_chunk_length_distribution(self, json_files: List[Path]):
        """Analyze token length distribution of chunks across all data"""
        import numpy as np

        all_chunk_lengths = []
        total_chunks = 0
        total_files = 0

        print("Scanning files to analyze chunk lengths...")
        print(len(json_files))

        for json_file in json_files: #[:min(50, len(json_files))]:  # Sample up to 50 files
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    item = json.load(f)
                if 'parsed_response' in item:
                    item = item['parsed_response']
                
                if not item:
                    #print(f"  Warning: Empty item in {json_file.name}, skipping")
                    continue
                
                chunks = item.get('chunks', [])
                n_channels = item.get('n_channels', 0)
                # if chunks == []:
                #     print(f"  Error reading chunks from {json_file.name}")
                #     chunks = item.get("parsed_response", {}).get('chunks', [])
                #     n_channels = item.get("parsed_response", {}).get('n_channels', 0)
                #     print(n_channels)

                if not chunks or n_channels == 0:
                    # Try to infer n_channels
                    if chunks:
                        first_chunk = chunks[0]
                        channel_keys = [k for k in first_chunk.keys() if k.startswith('channel_')]
                        n_channels = len(channel_keys)

                if n_channels == 0:
                    print(f"  Warning: No channels found in {json_file.name}, skipping")
                    continue

                for chunk in chunks:
                    for ch_idx in range(n_channels):
                        channel_key = f'channel_{ch_idx}'
                        content = chunk.get(channel_key, '')
                        if content:
                            tokens = self.tokenizer.encode(content, add_special_tokens=False)
                            all_chunk_lengths.append(len(tokens))
                            total_chunks += 1

                total_files += 1

            except Exception as e:
                import traceback
                print(traceback.print_exc())
                continue

        if not all_chunk_lengths:
            print("Warning: No chunk data found for analysis")
            return

        # Calculate statistics
        lengths = np.array(all_chunk_lengths)

        print(f"\n📊 Chunk Length Statistics (analyzed {total_files} files, {total_chunks} chunks):")
        print(f"  Mean:       {lengths.mean():.1f} tokens")
        print(f"  Median:     {np.median(lengths):.1f} tokens")
        print(f"  Std Dev:    {lengths.std():.1f} tokens")
        print(f"  Min:        {lengths.min()} tokens")
        print(f"  Max:        {lengths.max()} tokens")
        print(f"  25th %ile:  {np.percentile(lengths, 25):.1f} tokens")
        print(f"  75th %ile:  {np.percentile(lengths, 75):.1f} tokens")
        print(f"  90th %ile:  {np.percentile(lengths, 90):.1f} tokens")
        print(f"  95th %ile:  {np.percentile(lengths, 95):.1f} tokens")

        # Suggest target length
        suggested_target = int(np.percentile(lengths, 75))
        # Round to nearest power of 2 or multiple of 16
        if suggested_target <= 32:
            suggested_target = 32
        elif suggested_target <= 64:
            suggested_target = 64
        elif suggested_target <= 128:
            suggested_target = 128
        else:
            suggested_target = ((suggested_target + 15) // 16) * 16  # Round to multiple of 16

        print(f"\n💡 Recommended chunk_target_length: {suggested_target} tokens")
        print(f"   (Based on 75th percentile, rounded appropriately)")

        # Show distribution histogram
        print(f"\n📈 Length Distribution:")
        bins = [0, 16, 32, 48, 64, 80, 96, 128, 160, 256, float('inf')]
        bin_labels = ['0-16', '17-32', '33-48', '49-64', '65-80', '81-96', '97-128', '129-160', '161-256', '256+']

        for i in range(len(bins) - 1):
            count = ((lengths > bins[i]) & (lengths <= bins[i+1])).sum()
            percentage = count / len(lengths) * 100
            bar = '█' * int(percentage / 2)  # Scale down for display
            print(f"  {bin_labels[i]:>10}: {count:>6} ({percentage:>5.1f}%) {bar}")

        print()

    def _standardize_extracted_result(self, item: Dict) -> Optional[Dict]:
        """
        Standardize extracted_results format to internal format

        Input format:
        {
            "problem": "...",
            "n_channels": 3,
            "chunks": [
                {
                    "chunk_id": 0,
                    "channel_0": "...",
                    "channel_1": "...",
                    "channel_2": "..."
                },
                ...
            ]
        }

        Output format:
        {
            "question": "...",
            "answers": ["channel_0_all_chunks", "channel_1_all_chunks", ...]
        }
        """
        # Extract problem/question
        question = item.get('problem') or item.get('question') or item.get('prompt')
        if not question:
            print("  Warning: No problem/question found")
            return None

        # Extract chunks
        chunks = item.get('chunks', [])
        if not chunks:
            print("  Warning: No chunks found")
            return None

        # Get number of channels
        n_channels = item.get('n_channels')
        if n_channels is None:
            # Try to infer from first chunk
            first_chunk = chunks[0]
            channel_keys = [k for k in first_chunk.keys() if k.startswith('channel_')]
            n_channels = len(channel_keys)

        if n_channels == 0:
            print("  Warning: No channels found")
            return None

        print(f"  Found {len(chunks)} chunks with {n_channels} channels each")

        # Get or infer chunk target length
        target_length = self.chunk_target_length
        if target_length is None:
            target_length = item.get('target_chunk_tokens', 64)  # Default to 64 if not specified

        print(f"  Using chunk target length: {target_length} tokens")
        print(f"  Chunk alignment strategy: {self.chunk_alignment_strategy}")

        # Aggregate content by channel with chunk alignment
        # Each channel sees all its own chunks in sequence
        channel_contents = [[] for _ in range(n_channels)]

        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', -1)

            # Collect text for all channels in this chunk
            chunk_texts = []
            for ch_idx in range(n_channels):
                channel_key = f'channel_{ch_idx}'
                content = chunk.get(channel_key, '')
                chunk_texts.append(content)

            # Align chunk texts to same length
            aligned_texts = self._align_chunk_texts(chunk_texts, target_length, chunk_id)

            # Add aligned texts to channel contents
            for ch_idx, aligned_text in enumerate(aligned_texts):
                if aligned_text:
                    channel_contents[ch_idx].append(aligned_text)

        # Join chunks for each channel
        answers = []
        for ch_idx, contents in enumerate(channel_contents):
            if contents:
                # Join chunks with space (chunks are already aligned)
                channel_text = ' '.join(contents)
                answers.append(channel_text)
            else:
                print(f"  Warning: Channel {ch_idx} has no content")
                return None

        if not answers or len(answers) != n_channels:
            print(f"  Warning: Expected {n_channels} channels, got {len(answers)}")
            return None

        return {
            'question': question,
            'answers': answers,
            'n_channels': n_channels,
            'n_chunks': len(chunks)
        }

    def _align_chunk_texts(self, chunk_texts: List[str], target_length: int, chunk_id: int) -> List[str]:
        """
        Align all channel texts within a chunk to the same token length

        Args:
            chunk_texts: List of text strings for each channel in this chunk
            target_length: Target token length for alignment
            chunk_id: Chunk ID for logging

        Returns:
            List of aligned text strings
        """
        # Add chunk markers if enabled
        if self.add_chunk_markers:
            chunk_start_marker = f"<chunk_{chunk_id}>"
            chunk_end_marker = f"</chunk_{chunk_id}>"
            chunk_texts = [f"{chunk_start_marker}{text}{chunk_end_marker}" for text in chunk_texts]

        # Tokenize all texts
        tokenized_texts = []
        for text in chunk_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokenized_texts.append(tokens)

        align_length = target_length
        # Determine alignment length based on strategy
        # if self.chunk_alignment_strategy == "target":
        #     # Use target length strictly
        #     align_length = target_length

        # elif self.chunk_alignment_strategy == "max_per_chunk":
        #     # Use maximum length in this chunk
        #     max_len = max(len(tokens) for tokens in tokenized_texts)
        #     align_length = max_len
        #     if max_len > target_length * 1.5:  # Warn if much larger than target
        #         print(f"    Warning: Chunk {chunk_id} has max length {max_len} (target: {target_length})")

        # elif self.chunk_alignment_strategy == "adaptive":
        #     # Use target, but expand if any text is much longer
        #     max_len = max(len(tokens) for tokens in tokenized_texts)
        #     if max_len > target_length:
        #         align_length = max_len
        #         print(f"    Info: Chunk {chunk_id} expanded to {align_length} (target: {target_length})")
        #     else:
        #         align_length = target_length

        # else:  # "fixed" or default
        #     align_length = target_length

        # Align all tokenized texts
        aligned_tokens = []
        truncated_count = 0
        padded_count = 0

        for idx, tokens in enumerate(tokenized_texts):
            if len(tokens) > align_length:
                # Truncate
                aligned = tokens[:align_length]
                truncated_count += 1
            elif len(tokens) < align_length:
                # Pad with pad_token_id
                padding_needed = align_length - len(tokens)
                aligned = tokens + [self.pad_token_id] * padding_needed
                padded_count += 1
            else:
                # Already correct length
                aligned = tokens

            aligned_tokens.append(aligned)

        # Log alignment statistics for this chunk
        original_lengths = [len(t) for t in tokenized_texts]
        if chunk_id % 10 == 0 or truncated_count > 0:  # Log every 10th chunk or if truncation occurred
            print(f"    Chunk {chunk_id}: lengths {original_lengths} -> {align_length} " +
                  f"(truncated: {truncated_count}, padded: {padded_count})")

        # Decode back to text
        aligned_texts = []
        for tokens in aligned_tokens:
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            aligned_texts.append(text)

        return aligned_texts

    def _process_single_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item"""
        try:
            conversation = self._format_conversation(item['question'], item['answers'])
            input_ids = self.tokenizer.encode(conversation, add_special_tokens=False)
            boundaries = self._find_boundaries(input_ids)

            if not boundaries['heads']:
                print("  Warning: No head boundaries found")
                return None

            # Note: truncation is disabled by default, uncomment if needed
            # input_ids, boundaries = self._apply_truncation(input_ids, boundaries)

            labels = self._create_labels(input_ids, boundaries)
            position_ids = self._create_position_ids(input_ids, boundaries)
            attention_mask = self._create_attention_mask(input_ids, boundaries)
            try:
                print(f"  Processed item: seq_len={len(input_ids)}, heads={len(boundaries['heads'])}")
                print(f"    Boundaries: {boundaries}")
                print(f"    Shape Of Input IDs: {len(input_ids)}")
                print(f"    Shape Of Labels: {len(labels)}")
                print(f"    Shape Of Position IDs: {len(position_ids)}")
                print(f"    Shape Of Attention Mask: {attention_mask.shape}")
            except:
                pass
            
            return {
                'input_ids': input_ids,
                'labels': labels,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'num_heads': len(boundaries['heads']),
                'seq_length': len(input_ids)
            }
        except Exception as e:
            import traceback
            print(f"  Error in _process_single_item: {e}")
            traceback.print_exc()
            return None

    def _save_index_file(self, index_file_path: str, source_dir: str,
                        sample_files: List[Dict]):
        """Save index file recording information about all sample files"""
        index_data = {
            'version': '1.0-extracted-results',
            'total_samples': len(sample_files),
            'source_dir': source_dir,
            'cache_created': time.time(),
            'config': {
                'tokenizer_path': self.tokenizer_path,
                'max_seq_length': self.max_seq_length,
                'truncation_strategy': self.truncation_strategy,
                'position_ids_2d': self.position_ids_2d,
                'system_message': self.system_message,
                'add_communication_tokens': self.add_communication_tokens
            },
            'sample_files': sample_files
        }

        with open(index_file_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        print(f"Index file saved: {index_file_path}")

    def _print_statistics(self, total_samples: int, output_dir: str):
        """Print statistics"""
        print("\n=== Dataset Statistics ===")
        print(f"Total samples: {total_samples}")
        print(f"Files created: {total_samples} .npz files + 1 index.json")

        # Calculate total file size
        total_size = 0
        for filename in os.listdir(output_dir):
            if filename.endswith('.npz') or filename.endswith('.json'):
                file_path = os.path.join(output_dir, filename)
                total_size += os.path.getsize(file_path)

        print(f"Total cache size: {total_size / (1024 * 1024):.2f} MB")
        if total_samples > 0:
            print(f"Average size per sample: {total_size / total_samples / 1024:.2f} KB")

    # ============ Format and Boundary Detection ============

    def _format_conversation(self, question: str, answers: List[str]) -> str:
        """
        Format conversation with multi-head assistant responses
        Each answer represents one channel's aggregated chunks
        """
        conv = f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
        conv += f"<|im_start|>user\n{question}<|im_end|>\n"
        for answer in answers:
            conv += f"<|im_start|>assistant\n{answer}<|im_end|>"
        return conv

    def _find_boundaries(self, input_ids: List[int]) -> Dict[str, Any]:
        """Find assistant response boundaries"""
        boundaries = {'question_end': len(input_ids), 'heads': [], 'start': [], 'end': []}

        j = 0
        while j < len(input_ids):
            if input_ids[j] == self.im_start:
                boundaries['start'].append(j)
            if input_ids[j] == self.im_end:
                boundaries['end'].append(j)
            j += 1

        if len(boundaries['start']) < 3:
            print(f"  Warning: Not enough <|im_start|> markers found (expected >=3, got {len(boundaries['start'])})")
            return boundaries

        # The third <|im_start|> marks the beginning of the first assistant response
        boundaries['question_end'] = boundaries['start'][2]

        # Extract head boundaries (each assistant response is a head)
        for start, end in zip(boundaries['start'][2:], boundaries['end'][2:]):
            # Content starts after: <|im_start|> + "assistant" + "\n"
            content_start = start + 1 + len(self.assistant_tokens) + 1
            boundaries['heads'].append({'start': content_start, 'end': end})

        return boundaries

    def _apply_truncation(self, input_ids: List[int], boundaries: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """Apply truncation strategy (disabled by default for extracted_results)"""
        if len(input_ids) <= self.max_seq_length:
            return input_ids, boundaries

        print(f"  Warning: Sequence length {len(input_ids)} exceeds max {self.max_seq_length}")

        if self.truncation_strategy == "balanced":
            return self._truncate_balanced(input_ids, boundaries)
        else:
            return self._truncate_random(input_ids, boundaries)

    def _truncate_random(self, input_ids: List[int], boundaries: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """Random truncation of one head"""
        import random

        if not boundaries['heads']:
            return input_ids[:self.max_seq_length], self._find_boundaries(input_ids[:self.max_seq_length])

        head_idx = random.randint(0, len(boundaries['heads']) - 1)
        question_part = input_ids[:boundaries['start'][2]]
        available = self.max_seq_length - len(question_part)

        other_heads_len = sum(boundaries['end'][i+2]+1 - boundaries['start'][i+2]+1
                            for i, head in enumerate(boundaries['heads'])
                            if i != head_idx)

        selected_head_max_len = max(0, available - other_heads_len)

        truncated_ids = question_part[:]
        new_heads = []
        new_start = [boundaries['start'][i] for i in range(2)]
        new_end = [boundaries['end'][i] for i in range(2)]
        current_pos = len(question_part)

        for i, head in enumerate(boundaries['heads']):
            original_head_len = boundaries['end'][i+2] - boundaries['start'][i+2] + 1

            if i == head_idx:
                actual_len = min(original_head_len, selected_head_max_len)
            else:
                actual_len = original_head_len

            if actual_len > 0:
                if i == head_idx:
                    truncated_ids.extend(input_ids[boundaries['start'][i+2]:boundaries['start'][i+2] + actual_len -1]+[input_ids[boundaries['end'][i+2]]])
                else:
                    truncated_ids.extend(input_ids[boundaries['start'][i+2]:boundaries['start'][i+2] + actual_len])
                new_heads.append({'start': current_pos + 1 + len(self.assistant_tokens) + 1, 'end': current_pos + actual_len - 1})
                new_start.append(current_pos)
                new_end.append(current_pos + actual_len - 1)
                current_pos += actual_len

        return truncated_ids, {'question_end': boundaries['question_end'], 'heads': new_heads, 'start': new_start, 'end': new_end}

    def _truncate_balanced(self, input_ids: List[int], boundaries: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """Balanced truncation (simplified version)"""
        # For simplicity, just truncate to max length
        return input_ids[:self.max_seq_length], self._find_boundaries(input_ids[:self.max_seq_length])

    def _create_position_ids(self, input_ids: List[int], boundaries: Dict[str, Any]):
        """Create position IDs (2D for multi-head)"""
        seq_len = len(input_ids)

        if not self.position_ids_2d:
            # 1D position IDs
            position_ids = [0] * seq_len
            question_end = boundaries['question_end']

            for i in range(min(question_end, seq_len)):
                position_ids[i] = min(i, self.max_position_embeddings - 1)

            for head in boundaries['heads']:
                for i in range(head['start'], min(head['end'], seq_len)):
                    rel_pos = question_end + (i - head['start'])
                    position_ids[i] = min(rel_pos, self.max_position_embeddings - 1)

            return position_ids
        else:
            # 2D position IDs: [head_id, context_position]
            position_ids = [[0, 0]] * seq_len
            question_end = boundaries['start'][2]

            # Question part: head_id=0, incremental position
            for i in range(min(question_end, seq_len)):
                context_pos = min(i, self.max_position_embeddings - 1)
                position_ids[i] = [0, context_pos]

            # Each head: head_id=(idx+1), position relative to question_end
            for head_idx, head in enumerate(boundaries['heads']):
                for i in range(boundaries['start'][head_idx+2], min(boundaries['end'][head_idx+2]+1, seq_len)):
                    head_id = head_idx + 1
                    context_pos = min(question_end + (i - boundaries['start'][head_idx+2]), self.max_position_embeddings - 1)
                    position_ids[i] = [head_id, context_pos]

            return position_ids

    def _create_labels(self, input_ids: List[int], boundaries: Dict[str, Any]) -> List[int]:
        """Create labels (only assistant responses are trained)"""
        labels = [-100] * len(input_ids)

        for head_idx, head in enumerate(boundaries['heads']):
            for i in range(head['start'], min(boundaries['end'][head_idx+2]+1, len(input_ids))):
                labels[i] = input_ids[i]

        return labels

    # def _create_attention_mask(self, seq_len: int, boundaries: Dict[str, Any]) -> np.ndarray:
    #     """
    #     Create attention mask for parallel multi-head generation

    #     Attention rules:
    #     1. Question tokens can see previous question tokens
    #     2. Each head can see:
    #        - All question tokens
    #        - Its own tokens up to current position
    #        - Same relative position in ALL other heads (parallel sync)
    #     """
    #     mask = np.zeros((seq_len, seq_len), dtype=bool)
    #     question_end = boundaries['start'][2]

    #     # Self-attention for all positions
    #     for i in range(seq_len):
    #         mask[i, i] = True

    #     # Question part: causal attention
    #     for i in range(question_end):
    #         mask[i, :i+1] = True

    #     heads = boundaries['heads']

    #     # Multi-head attention with parallel visibility
    #     for i, head in enumerate(heads):
    #         head_start = min(boundaries['start'][2+i], seq_len)
    #         head_end = min(boundaries['end'][2+i]+1, seq_len)

    #         for pos in range(head_start, head_end):
    #             if pos >= seq_len:
    #                 break

    #             # Can see all question tokens
    #             mask[pos, :question_end] = True

    #             # Can see itself
    #             mask[pos, pos] = True

    #             # Relative position in this head
    #             rel_pos = pos - head_start
    #             visible_token_count = rel_pos + 1

    #             # Can see same relative position in ALL other heads
    #             for head_idx, other_head in enumerate(heads):
    #                 other_start = min(boundaries['start'][2+head_idx], seq_len)
    #                 other_end = min(boundaries['end'][2+head_idx], seq_len)
    #                 other_visible_token_count = other_end - other_start + 1

    #                 # See up to the same relative position (or end of other head)
    #                 for k in range(min(visible_token_count, other_visible_token_count)):
    #                     other_pos = other_start + k
    #                     if other_pos < seq_len:
    #                         mask[pos, other_pos] = True

    #     return mask.astype(np.float16)

    def _create_attention_mask(self, input_ids: List[int], boundaries: Dict[str, Any]) -> np.ndarray:
        seq_len = len(input_ids)
        mask = np.zeros((seq_len, seq_len), dtype=bool)

        question_end = boundaries["start"][2]  # first assistant start
        # question causal
        for i in range(question_end):
            mask[i, :i+1] = True

        heads = boundaries["heads"]
        H = len(heads)
        head_spans = [
            (min(boundaries["start"][2+h], seq_len), min(boundaries["end"][2+h] + 1, seq_len))
            for h in range(H)
        ]

        # --- build chunk_id per token in each head ---
        # chunk_id = -1 means outside any chunk marker (we'll treat as always-visible history)
        chunk_id = [np.full(seq_len, -1, dtype=np.int16) for _ in range(H)]

        if not self.add_chunk_markers:
            # fallback: no isolation possible; treat as single chunk 0
            for h, (hs, he) in enumerate(head_spans):
                chunk_id[h][hs:he] = 0
        else:
            start2c = {self.chunk_start_ids[c]: c for c in self.chunk_start_ids}
            end2c   = {self.chunk_end_ids[c]: c for c in self.chunk_end_ids}

            for h, (hs, he) in enumerate(head_spans):
                cur = -1
                for p in range(hs, he):
                    t = input_ids[p]
                    if t in start2c:
                        cur = start2c[t]
                    chunk_id[h][p] = cur
                    if t in end2c:
                        cur = -1

        # --- precompute spans per head per chunk ---
        # chunk_spans[h][c] = (start, end_excl)
        chunk_spans = [dict() for _ in range(H)]
        for h, (hs, he) in enumerate(head_spans):
            ids = chunk_id[h][hs:he]
            if ids.size == 0:
                continue
            run_c = int(ids[0]); run_s = hs
            for off in range(1, ids.size):
                c = int(ids[off]); p = hs + off
                if c != run_c:
                    if run_c >= 0:
                        chunk_spans[h][run_c] = (run_s, p)
                    run_c, run_s = c, p
            if run_c >= 0:
                chunk_spans[h][run_c] = (run_s, he)

        # --- fill mask for assistant tokens ---
        # rule for pos in head h, chunk c:
        #   see question + own causal + all heads' chunks < c
        for h, (hs, he) in enumerate(head_spans):
            for pos in range(hs, he):
                # always see full question
                mask[pos, :question_end] = True

                # own causal (includes own previous chunks automatically)
                mask[pos, hs:pos+1] = True

                c = int(chunk_id[h][pos])
                if c <= 0:
                    continue  # c==-1 or c==0 => no history chunks < c

                # allow all heads' history chunks < c
                for oh in range(H):
                    for pc in range(c):
                        span = chunk_spans[oh].get(pc, None)
                        if span is not None:
                            s, e = span
                            mask[pos, s:e] = True

        # self tokens always visible
        np.fill_diagonal(mask, True)
        return mask.astype(np.float16)



def main():
    parser = argparse.ArgumentParser(description="Process extracted_results dataset")
    parser.add_argument("--input", required=True, help="Input directory containing extracted_results JSON files")
    parser.add_argument("--output", required=True, help="Output cache directory")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or name")
    parser.add_argument("--config", help="Configuration JSON file")

    # Processing parameters
    parser.add_argument("--max-seq-length", type=int, default=32678)
    parser.add_argument("--truncation-strategy", choices=["random", "balanced"], default="random")
    parser.add_argument("--max-position-embeddings", type=int, default=32678)
    parser.add_argument("--position-ids-2d", action="store_true", default=True)
    parser.add_argument("--system-message", default="You are a helpful assistant.")
    parser.add_argument("--add-communication-tokens", action="store_true", default=True,
                       help="Add communication tokens like <to:channel_X>")

    # Chunk alignment parameters
    parser.add_argument("--chunk-alignment-strategy",
                       choices=["target", "max_per_chunk", "adaptive", "fixed"],
                       default="target",
                       help="Strategy for aligning chunk lengths: "
                            "'target' (use target_chunk_tokens from data or --chunk-target-length), "
                            "'max_per_chunk' (align to max length within each chunk), "
                            "'adaptive' (use target but expand if needed), "
                            "'fixed' (use --chunk-target-length strictly)")
    parser.add_argument("--chunk-target-length", type=int, default=None,
                       help="Target token length for chunks (default: use target_chunk_tokens from data, or 64)")
    parser.add_argument("--add-chunk-markers", action="store_true", default=True,
                       help="Add <chunk_N> markers to identify chunk boundaries (recommended)")
    parser.add_argument("--no-chunk-markers", dest="add_chunk_markers", action="store_false",
                       help="Disable chunk markers")
    parser.add_argument("--analyze-chunk-lengths", action="store_true", default=True,
                       help="Analyze chunk length distribution before processing")
    parser.add_argument("--no-analyze", dest="analyze_chunk_lengths", action="store_false",
                       help="Skip chunk length analysis")

    args = parser.parse_args()

    # Load config file (if provided)
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Merge configuration
    processor_config = {
        'tokenizer_path': args.tokenizer,
        'max_seq_length': config.get('max_seq_length', args.max_seq_length),
        'truncation_strategy': config.get('truncation_strategy', args.truncation_strategy),
        'max_position_embeddings': config.get('max_position_embeddings', args.max_position_embeddings),
        'position_ids_2d': config.get('position_ids_2d', args.position_ids_2d),
        'system_message': config.get('system_message', args.system_message),
        'add_communication_tokens': config.get('add_communication_tokens', args.add_communication_tokens),
        'chunk_alignment_strategy': config.get('chunk_alignment_strategy', args.chunk_alignment_strategy),
        'chunk_target_length': config.get('chunk_target_length', args.chunk_target_length),
        'add_chunk_markers': config.get('add_chunk_markers', args.add_chunk_markers),
        'analyze_chunk_lengths': config.get('analyze_chunk_lengths', args.analyze_chunk_lengths)
    }

    # Create processor
    processor = ExtractedResultsProcessor(**processor_config)

    # Process data
    try:
        output_dir = processor.process_extracted_results_dir(
            input_dir=args.input,
            output_dir=args.output
        )
        print(f"\n{'='*60}")
        print(f"Processing completed successfully!")
        print(f"{'='*60}")
        print(f"Files saved to: {output_dir}")
        print(f"Structure:")
        print(f"   ├── index.json          # Index file")
        print(f"   ├── sample_000000.npz   # First sample")
        print(f"   ├── sample_000001.npz   # Second sample")
        print(f"   └── ...")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Processing failed: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
