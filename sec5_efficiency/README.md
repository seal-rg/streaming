# Section 5: Efficiency

Code for **Section 5 (Efficiency)** of *"Multi-Stream LLMs: Unblocking
Language Models with Parallel Streams of Thoughts, Inputs and Outputs"*:
training Qwen3-1.7B / Qwen3-4B as parallel-stream models (2 streams for
solving-while-reading, 3 for auditing-while-solving) and evaluating on
math, logical reasoning, and QA benchmarks.

Streams are interleaved into a single token sequence with per-stream RoPE
counters, additive stream embeddings, a stream-causal attention mask, and
a single shared LM head.

## Layout

```
train/
  qwen3/                          Qwen3 backbone + multi-stream interleaved packing
  custom_datasets/                multi-head Dataset / Collator
  train/                          training entry + trainer + deepspeed configs
    train_qwen3.py                Sec 5 entry (Qwen3 base, multi-head)
    custom_trainer.py             trainer used by train_qwen3.py
  scripts/
    multi_gpu/train_qwen3_efficiency.sh    canonical training launcher
infer/
  infer_qwen3.py                  benchmark evaluation runner (multi-stream)
  infer.sh                        example invocation
```

## Setup

```bash
pip install -r train/train/requirements.txt
export SEC5_ROOT=$(pwd)
```

## Training (Sec 5: Efficiency, Qwen3-1.7B / 4B)

`train/scripts/multi_gpu/train_qwen3_efficiency.sh` is the canonical
launcher. Set `SEC5_ROOT`, `MODEL`, and `TRAIN_FILE`, then `bash` it.
Underlying command:

```bash
accelerate launch \
  --num_processes=4 \
  --config_file $SEC5_ROOT/train/train/deepspeed_zero2.yaml \
  $SEC5_ROOT/train/train/train_qwen3.py \
  --model_name /path/to/Qwen3-4B/snapshots/medusa \
  --train_file_path /path/to/processed_cache \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=3 \
  --learning_rate=1e-5 \
  --warmup_ratio=0.1 \
  --bf16=True \
  --output_dir /path/to/checkpoints
```

`train_qwen3.py` imports `Qwen3ForMultiStream` from `qwen3/` and
`CustomizedTrainer` from `custom_trainer.py`.

## Evaluation (Sec 5 benchmarks)

Tasks reported in paper Table 1 / 2: **GSM8K**, **MATH500**, **LogicNLI**,
**SQuAD**, **ProofWriter**, **PubMedQA**.

Run multi-stream inference on a checkpoint:

```bash
MODEL=/path/to/checkpoint TASK=gsm8k bash infer/infer.sh
```

Or directly:

```bash
python3 infer/infer_qwen3.py \
  --model  /path/to/checkpoint \
  --task   gsm8k \
  --stream \
  --out_dir output/eval_run
```

Supported `--task` values: `gsm8k`, `math500`, `proofwriter`, `logicnli`,
`logiqa`, `mathqa`, `arc_c`, `strategyqa`, `squad`, `pubmedqa`.

Pass `--stream` to use multi-stream inference (`Qwen3ForMultiStream`).
Omit it to run standard HF generation as a single-stream baseline.

## Notes

- All cluster-specific paths are placeholder shell variables: `${SEC5_ROOT}`
  (this folder), `${MODELS_ROOT}` (HF snapshots), `${DATA_ROOT}`
  (pre-processed datasets), `${RESULTS_ROOT}` (training outputs). Set them
  in your shell or edit the scripts.
- The `WANDB_API_KEY` env wiring in `train_qwen3.py` has been redacted;
  set it in your shell instead. Default `wandb_project` is `multistream-efficiency`.
