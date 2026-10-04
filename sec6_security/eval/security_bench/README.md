# security_bench

Self-contained security / prompt-injection-robustness evaluation suite
(TensorTrust, Purple, Gandalf, RuLES, StruQ, NESSiE, IFEval) for the Vanilla
vs. Stream(Ours) Qwen3-4B multi-stream checkpoints from Table 3. No
dependency on the upstream ASIDE harness referenced in `../README.md` --
everything needed to run is vendored inside this folder.

## Layout

```
security_bench/
├── __init__.py
├── common.py              # seeding, string cleanup, mean/std, InferenceBackend protocol
├── backends.py             # VanillaBackend (plain AutoModelForCausalLM) + StreamBackend (Qwen3ForMultiStream)
├── datasets.py              # dataset loading/caching for every benchmark
├── benchmarks.py             # one eval_xxx() function per benchmark, backend-agnostic
├── cli.py                     # argparse entry point: --backend vanilla|stream ...
├── run_vanilla.py               # thin wrapper, fixes --backend vanilla
├── run_stream.py                  # thin wrapper, fixes --backend stream
├── run_security_eval.sh             # shell driver (PID 0=vanilla, PID 1=stream)
├── vendor/                            # vendored non-pip dependencies (see below)
├── data/                                # small static data files (see below)
└── cache/eval_cache/                      # auto-populated on first run (see below)
```

## How the data is organized

**`vendor/`** -- vendored code + the data that ships alongside it, so this
folder needs nothing outside itself to run:

- `vendor/qwen3_multistream/` -- the `Qwen3ForMultiStream` architecture
  (`__init__.py`, `configuration_qwen3.py`, `modular_qwen3.py`,
  `modeling_qwen3.py`). Loaded by `backends.py` via explicit file path,
  independent of `sys.path`.
- `vendor/rules.py` -- the RuLES benchmark scenarios/scoring logic,
  originally from the ASIDE harness (Zverev et al.), vendored here
  stdlib-only and imported as `from .vendor import rules`.

**`data/`** -- small static benchmark inputs:

- `data/struq_eval.json` -- StruQ's `tatsu-lab/alpaca_farm` eval split.
- `data/prompt_templates.json` -- system/user prompt template used to format
  StruQ examples.

**`cache/eval_cache/`** -- *not* checked in; populated automatically the
first time each benchmark runs, then reused on every subsequent run:

| Benchmark  | Source                                              | Cached as |
|---|---|---|
| TensorTrust | GitHub `HumanCompatibleAI/tensor-trust-data` (main) | `hijacking_processed.jsonl` |
| Purple      | GitHub `kothasuhas/purple-problem` (pinned commit)  | `purple_processed.json` |
| Gandalf     | HF `Lakera/gandalf_summarization`                   | `gandalf_processed/` |
| NESSiE      | HF `JByale/NESSiE`, split=test                      | `nessie_test.parquet` |

RuLES and StruQ don't hit this cache -- RuLES' scenarios are defined directly
in `vendor/rules.py`, and StruQ reads straight from `data/struq_eval.json`.
IFEval doesn't use this cache either; it goes through the `lm_eval` package's
own dataset loading/caching, external to this package.

Removed from the original benchmark set (not vendored, not run by default):
**BIPIA** and **HackerPrompt**. Neither is part of the Vanilla/Stream Table 3
comparison (not in `run_security_eval.sh`'s default `--datasets` list), and
BIPIA in particular pulled in a 6.5MB third-party benchmark package for
functionality nothing here uses.

## Checkpoints and training data

See `../README.md` ("Training data we released" / "Released checkpoints")
for the public SFT dataset and the Vanilla/Stream Qwen3-4B checkpoints this
suite evaluates.

## Usage

```bash
# via the shell driver (mirrors infer_security_4b.sh's PID convention)
./run_security_eval.sh _ 0   # Vanilla
./run_security_eval.sh _ 1   # Stream

# or directly
python3 run_vanilla.py --model_path /path/to/checkpoint --out_dir out/vanilla --datasets tensortrust purple gandalf rules struq ifeval
python3 run_stream.py  --model_path /path/to/checkpoint --out_dir out/stream  --datasets tensortrust purple gandalf rules struq ifeval
```

`VANILLA_MODEL` / `STREAM_MODEL` / `SEEDS_TO_RUN` / `OUT_DIR_OVERRIDE` env
vars override `run_security_eval.sh`'s defaults. Model checkpoints
themselves are not included here (not code/small data) -- point
`--model_path` at wherever you have them.
