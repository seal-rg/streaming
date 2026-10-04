# Section 6: Security

Code for **Section 6 (Security)** of *"Multi-Stream LLMs: Unblocking Language
Models with Parallel Streams of Thoughts, Inputs and Outputs"*: training
Qwen2.5-7B-base and Qwen3-4B-base as parallel-stream models on a
multi-stream-reconstructed Alpaca dataset, then evaluating prompt-injection
robustness.

Streams are interleaved into a single token sequence with per-stream RoPE
counters, additive stream embeddings, a stream-causal attention mask, and
a single shared LM head.

## Layout

```
train/
  qwen2/                          Qwen2.5 backbone + multi-stream interleaved packing
  custom_datasets/                multi-head Dataset / Collator
  train/                          training entry + trainer + deepspeed configs
    train.py                      Sec 6 entry (Qwen2.5 base, multi-head)
    custom_trainer.py             trainer used by train.py
  scripts/
    multi_gpu/train_qwen2_security.sh    canonical training launcher
```

## Setup

```bash
pip install -r train/train/requirements.txt
export SEC6_ROOT=$(pwd)
```

The security evaluator requires the upstream **ASIDE / safety_evals** harness
(Zverev et al.), which is **not** vendored here. Clone separately and set
`ASIDE_ROOT`:

```bash
git clone https://github.com/eth-sri/aside /path/to/aside
export ASIDE_ROOT=/path/to/aside
```

## Training (Sec 6: Security, Qwen2.5-7B / Qwen3-4B)

`train/scripts/multi_gpu/train_qwen2_security.sh` is the canonical launcher.
Set `SEC6_ROOT`, `MODEL`, and `TRAIN_FILE`, then `bash` it. Underlying command:

```bash
accelerate launch \
  --num_processes=4 \
  --config_file $SEC6_ROOT/train/train/deepspeed_zero2.yaml \
  $SEC6_ROOT/train/train/train.py \
  --model_name /path/to/Qwen2.5-7B/snapshots/medusa \
  --train_file_path /path/to/alpaca_processed_cache \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=3 \
  --learning_rate=1e-5 \
  --warmup_ratio=0.1 \
  --bf16=True \
  --output_dir /path/to/checkpoints
```

`train.py` imports `Qwen2ForMultiStream` from `qwen2/` and
`CustomizedTrainer` from `custom_trainer.py`.

## Evaluation (Sec 6 benchmarks)

Benchmarks reported in paper Table 3:

| Type                       | Benchmark        | Status                                                    |
| -------------------------- | ---------------- | --------------------------------------------------------- |
| Direct prompt injection    | TensorTrust      | via `aside/experiments/evaluations/` — set `ASIDE_ROOT`   |
| Direct prompt injection    | Gandalf          | via `aside/...`                                           |
| Direct prompt injection    | Purple           | via `aside/...`                                           |
| Direct prompt injection    | RuLES            | via `aside/...`                                           |
| Indirect prompt injection  | StruQ-ID         | via `aside/.../struq/data/`                               |
| Indirect prompt injection  | StruQ-OOD        | via `aside/.../struq/data/`                               |
| Instruction following      | IFEval           | via `lm-eval-harness` with `ASIDE_ROOT` set               |

## Training data we released

The multi-stream-reconstructed Alpaca SFT data used to train both the
Qwen2.5-7B and Qwen3-4B security checkpoints (48,538 examples,
`instruction`/`input`/`output`) is public on Hugging Face:

**https://huggingface.co/datasets/guinansu/paragen-security-sft-alpaca**

Field mapping: `instruction` -> system message (falls back to a default if
empty), `input` -> user message (may be empty), `output` -> assistant
target. See that dataset's own README for provenance/license notes.

## Released checkpoints

The Qwen3-4B vanilla/stream pair from Table 3 is public on Hugging Face, each
pinned at a fixed revision:

| Variant | Repo | Revision |
| --- | --- | --- |
| Vanilla (baseline) | [`guinansu/vanilla-qwen3-4b-security`](https://huggingface.co/guinansu/vanilla-qwen3-4b-security) | `2a94879b4a03fc791b229dc443580ad89638176b` |
| Stream (ours) | [`guinansu/stream-qwen3-4b-security`](https://huggingface.co/guinansu/stream-qwen3-4b-security) | `6b1fd82efd84fc5aa841df1c5e392058155bbfe4` |

The Qwen2.5-7B pair is not currently released as checkpoints; only the
training data above is public for that variant.

## Notes

- All cluster-specific paths are placeholder shell variables: `${SEC6_ROOT}`
  (this folder), `${MODELS_ROOT}` (HF snapshots), `${DATA_ROOT}`
  (pre-processed datasets), `${RESULTS_ROOT}` (training outputs),
  `${ASIDE_ROOT}` (external ASIDE clone). Set them in your shell or
  edit the scripts.
- The `WANDB_API_KEY` env wiring in `train.py` has been redacted; set it
  in your shell instead. Default `wandb_project` is `multistream-security`.
