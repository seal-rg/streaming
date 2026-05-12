# Section 5: Efficiency

Code for **Section 5 (Efficiency)** of *"Multi-Stream LLMs: Unblocking
Language Models with Parallel Streams of Thoughts, Inputs and Outputs"*:
training Qwen3-1.7B / Qwen3-4B as parallel-stream models (2 streams for
solving-while-reading, 3 for auditing-while-solving) and evaluating on
math, logical reasoning, and QA benchmarks.

Streams are interleaved into a single token sequence with per-stream RoPE
counters, additive stream embeddings, a stream-causal attention mask, and
a single shared LM head.

This subfolder is self-contained for Sec 5. Sec 6 (security) lives in
`../sec6_security/` and shares much of the same multi-head training
infrastructure but is kept separate for clarity.

## Layout

```
train/
  qwen3/                          Qwen3 backbone + multi-stream interleaved packing
  rotary_embedding_torch/         vendored RoPE library (incl. ChannelPhaseRoPE)
  custom_datasets/                multi-head Dataset / Collator + data-prep scripts
  train/                          training entry + trainer + deepspeed configs
    train_qwen3.py                Sec 5 entry (Qwen3 base, multi-head)
    custom_trainer_optimized.py   trainer used by train_qwen3.py
  eval/                           offline evaluators (GSM8K, ProofWriter, IFEval, …)
  scripts/                        launch + benchmark scripts
    multi_gpu/train_qwen3_efficiency.sh    canonical training launcher
    infer/                                  microbenchmarks (medusa_e2e, sdpa_mask, …)
  prepare/                        tokenizer-extension pipeline
infer/                            paper-side inference scripts
  infer_qwen3.py                  Qwen3 evaluation runner
  dataset/                        SQuAD val sample
data_gen/                         wait-k synthetic data generation pipeline
  gen_data_metamath.py            MetaMath wait-k generator
  gen_data_hotpotqa.py            HotpotQA wait-k generator
  gen_data_multi.py               generic multi-stream generator
  multi_agent_complete_system.py  multi-agent orchestrator
  shared_cache/                   disk-backed LLM-response cache
```

## Setup

```bash
pip install -r train/train/requirements.txt
export SEC5_ROOT=$(pwd)
```

## Training (Sec 5: Efficiency, Qwen3-1.7B / 4B)

`train/scripts/multi_gpu/train_qwen3_efficiency.sh` is the canonical
launcher. Set `SEC5_ROOT`, `BASE_MODEL`, and `TRAIN_FILE`, then `bash` it.
Underlying command:

```bash
accelerate launch \
  --num_processes=4 \
  --config_file $SEC5_ROOT/train/train/deepspeed_zero2.yaml \
  $SEC5_ROOT/train/train/train_qwen3.py \
  --model_name /path/to/Qwen3-1.7B/snapshots/medusa \
  --train_file_path /path/to/processed_cache \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=3 \
  --learning_rate=1e-5 \
  --warmup_ratio=0.1 \
  --bf16=True \
  --output_dir /path/to/checkpoints
```

`train_qwen3.py` imports `Qwen3ForMedusa` from `qwen3/` and
`CustomizedTrainer` from `custom_trainer_optimized.py`.

## Data construction

`data_gen/` contains the wait-$k$ data construction pipeline described in
paper Sec 3.2:

- `gen_data_metamath.py` — wait-$k$ samples from MetaMath (Sec 5 math eval).
- `gen_data_hotpotqa.py` — same for HotpotQA.
- `gen_data_multi.py` — generic multi-stream generator from a source corpus.
- `multi_agent_complete_system.py` — multi-agent orchestrator that drives
  generation, causal verification, and quality filtering.
- `shared_cache/` — disk-backed cache for LLM responses during generation.

A 50-row sample of the produced data is included at
`train/custom_datasets/sample_50rows_aggregation_110_leader_8_test.jsonl`
to illustrate the on-disk row format. Full datasets are large
(multi-GB) and regenerable; not shipped.

## Evaluation (Sec 5 benchmarks)

Tasks reported in paper Table 1 / 2: **GSM8K**, **MATH500**, **LogicNLI**,
**SQuAD**, **ProofWriter**, **PubMedQA**.

Single-model evaluation:

```bash
python infer/infer_qwen3.py \
  --model /path/to/checkpoint \
  --tasks gsm8k,proofwriter,logicnli,mathqa,mmlu_redux \
  --n_samples 5 \
  --max_new_tokens 1024 \
  --out_dir output/eval_run
```

Multi-head-aware offline scoring (after inference produces `.jsonl`):

- `train/eval/eval_gsm8k_new26.py` — GSM8K scorer.
- `train/eval/eval_proofwriter_new26.py` — ProofWriter scorer.
- `train/eval/eval_multicheckpoint_3head.py` — wires all five Sec 5 tasks
  (SQuAD / ProofWriter / LogicNLI / PubMedQA / GSM8K) for a sweep over
  checkpoints; this is the primary multi-task driver.
- `train/eval/eval_ifeval_hf_lmeval_scoring.py` — IFEval scoring via
  lm-eval-harness (instruction-following capability check).
- `train/eval/medusa_eval_math_hard.py` — Math hard evaluation.
- `train/eval/medusa_eval_all.py` — multi-benchmark Medusa-aware driver.
- `train/eval/aggregate_ifeval_jsonl.py` — IFEval aggregator.

## Benchmarking the multi-head streaming path

`train/scripts/infer/benchmark_*.py` provide microbenchmarks of the
multi-head streaming inference path:

- `benchmark_medusa_e2e.py` — end-to-end latency / throughput.
- `benchmark_compare_origin_multi.py` — multi-head vs. original sequential.
- `benchmark_origin_generate.py` — baseline `model.generate()` throughput.
- `benchmark_sdpa_mask.py` — attention-mask construction overhead.

## Notes

- All cluster-specific paths are placeholder shell variables: `${SEC5_ROOT}`
  (this folder), `${MODELS_ROOT}` (HF snapshots), `${DATA_ROOT}`
  (pre-processed datasets), `${RESULTS_ROOT}` (training outputs). Set them
  in your shell or edit the scripts.
- The `WANDB_API_KEY` env wiring in `train_qwen3.py` has been redacted;
  set it in your shell instead. Default `wandb_project` is `multistream-efficiency`.
