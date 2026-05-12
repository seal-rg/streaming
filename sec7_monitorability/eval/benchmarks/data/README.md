# Benchmark Data

Datasets for the three monitorability benchmarks reported in Section 7.

## Final datasets used by the benchmarks

| File                              | Benchmark module           | Items | Used for                                                  |
| --------------------------------- | -------------------------- | ----- | --------------------------------------------------------- |
| `alignment_faking_prompts_v2.json`| `alignment_faking.py`      | 50    | AF eval-awareness / AF sub-vocalization (Greenblatt 2024) |
| `monitor_accuracy_50.json`          | `monitor_accuracy.py`        | 50    | Monitor accuracy (6-class covert behaviour)               |
| `monitor_accuracy_50_labeled.json`  | `monitor_accuracy.py`        | 50    | Labeled variant used in paper Table 4                     |
| `monitor_accuracy_balanced_v1.json` | `monitor_accuracy.py`        | 50    | Class-balanced variant                                    |
| `concern_subvocalization_50.json`     | `concern_subvocalization.py`   | 50    | Concern sub-vocalization (7 categories)                   |

## Monitorability source data + build scripts

`monitor_accuracy_50.json` was assembled from three upstream sources by the
build scripts in this directory.

| Source file                          | Description                                                |
| ------------------------------------ | ---------------------------------------------------------- |
| `apollo_scheming.json`               | Apollo "scheming" scenarios                                |
| `instrumental_convergence_50.json`   | Instrumental convergence stress tests                      |
| `cot_faithfulness_scenarios.json`    | Chain-of-thought faithfulness scenarios (Chen / Lanham / Turpin) |

Rebuild from sources:

```bash
python build_monitor_accuracy.py           # → monitor_accuracy_50.json
python build_monitor_accuracy_labels.py    # → monitor_accuracy_50_labeled.json
python build_monitor_accuracy_balanced_v1.py # → monitor_accuracy_balanced_v1.json
```
