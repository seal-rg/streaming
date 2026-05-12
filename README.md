# Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs

Code release for *"Multi-Stream LLMs: Unblocking Language Models with Parallel
Streams of Thoughts, Inputs and Outputs"* (Su, Yang, Li, Geiping; 2026).

The paper has three experimental sections, each with its own subfolder
under this directory. The subfolders are self-contained and each can be
set up and run independently.

## Layout

```
sec5_efficiency/      Paper Section 5: Efficiency
                      Qwen3-1.7B / 4B with 2- or 3-stream interleaved packing.
                      Trains "solving-while-reading" and "auditing-while-solving".
                      Eval: GSM8K, MATH500, LogicNLI, SQuAD, ProofWriter, PubMedQA.

sec6_security/        Paper Section 6: Security
                      Qwen2.5-7B / Qwen3-4B with interleaved packing.
                      Trains on multi-stream-reconstructed Alpaca.
                      Eval: TensorTrust, Gandalf, Purple, RuLES, StruQ-ID/OOD,
                      NESSiE, IFEval.

sec7_monitorability/  Paper Section 7: Monitorability
                      Stream-8B (Qwen3-8B) and Stream-27B (Qwen3.5-27B) with
                      10 cognitive streams. Qwen3.5 uses per-stream
                      Gated-DeltaNet states.
                      Eval: AF eval-aware/sub-vocalization, monitor accuracy
                      (Meinke/Schoen 6-class), concern sub-vocalization.
```

## Implementation Notes across subfolders


| Aspect              | Sec 5 / Sec 6                            | Sec 7                              |
| ------------------- | ---------------------------------------- | ---------------------------------- |
| Backbone            | Qwen2.5 / Qwen3                          | Qwen3 / Qwen3.5 (incl. DeltaNet)   |
| Dataset format      | `.jsonl`, single-row per timestep        | `.npz` packed shards, 10-channel   |
| Data construction   | wait-$k$ (MetaMath, HotpotQA, Alpaca)    | synthetic 10-stream tabular        |
| Number of streams   | 2 (solving-while-reading) / 3 (auditing) | 10 (User, Output, 8 thinking)      |
| Entry point         | `train/train/train{_qwen3}.py`           | `train/train/train_stream.py`      |

Note: Sec 5 / Sec 6 model classes are named `Qwen2ForMedusa` /
`Qwen3ForMedusa` and have a `medusa_num_heads` config attribute for historical reasons, but at
inference time the model functions as described in the paper with complete weight sharing between streams.

## Quick start

Pick the section you care about and follow its README:

- **Sec 5 (Efficiency)** — `sec5_efficiency/README.md`
- **Sec 6 (Security)** — `sec6_security/README.md`
- **Sec 7 (Monitorability)** — `sec7_monitorability/README.md`


## Citation

```bibtex
@article{su_2026_multi-stream,
  title={Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs},
  author={Su, Guinan and Yang, Yanwu and Li, Xueyan and Geiping, Jonas},
  year={2026}
}
```
