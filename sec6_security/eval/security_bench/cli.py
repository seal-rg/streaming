"""Generic CLI: `python -m security_bench.cli --backend vanilla|stream ...`
Runs the requested datasets over --seeds_to_run (or --seeds) seeds and writes
per-example JSONL + an aggregated metrics.json to --out_dir."""
import argparse
import json
import os
from typing import Any, Dict

from . import benchmarks as bm
from . import datasets as ds
from .backends import BACKENDS
from .common import ensure_dir, mean_std_pct, seed_everything
from .datasets import DATA_DIR

ALL_DATASETS = ["tensortrust", "purple", "gandalf", "rules", "struq", "ifeval", "nessie"]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS.keys()))
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", type=str, default="outputs_security")

    ap.add_argument("--datasets", nargs="+", default=["all"], choices=["all"] + ALL_DATASETS)

    ap.add_argument("--nessie_limit", type=int, default=None)
    ap.add_argument("--nessie_strict_helpful", type=int, default=0)

    ap.add_argument("--seeds_to_run", type=int, default=3,
                     help="Used only if --seeds not given. Seeds are [2024..2024+seeds_to_run-1].")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)

    ap.add_argument("--ifeval_system", type=str, default="You are a helpful assistant.")
    ap.add_argument("--ifeval_max_samples", type=int, default=0)
    ap.add_argument("--ifeval_subsample_seed", type=int, default=0)

    ap.add_argument("--struq_data_path", type=str, default=str(DATA_DIR / "struq_eval.json"))
    ap.add_argument("--struq_domain", type=str, default="all", choices=["all"] + ds.STRUQ_DOMAIN_TYPES)
    ap.add_argument("--struq_attack", type=str, default="all", choices=["all"] + ds.STRUQ_ATTACK_TYPES)
    ap.add_argument("--prompt_templates_path", type=str, default=str(DATA_DIR / "prompt_templates.json"))
    ap.add_argument("--prompt_template_index", type=int, default=0)

    ap.add_argument("--do_sample", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--max_length_per_head", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=1024)
    ap.add_argument("--final_prefix", type=str, default="")
    ap.add_argument("--stop_on_im_end", type=int, default=1)

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    ensure_dir(args.out_dir)

    datasets_to_run = ALL_DATASETS if "all" in args.datasets else args.datasets

    seeds = list(map(int, args.seeds)) if args.seeds else list(range(2024, 2024 + int(args.seeds_to_run)))
    print("[Seeds]", seeds)

    backend_cls = BACKENDS[args.backend]
    infer = backend_cls(args.model_path, device=args.device, torch_dtype=args.torch_dtype)

    gen_kwargs = dict(
        final_prefix=args.final_prefix,
        max_length_per_head=args.max_length_per_head,
        max_steps=args.max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        do_sample=bool(args.do_sample),
        stop_on_im_end=bool(args.stop_on_im_end),
    )

    metrics: Dict[str, Any] = {}

    rules_examples = hijacking_df = purple_prompts = gandalf_ds = None
    struq_data = struq_template = nessie_df = None

    if "rules" in datasets_to_run:
        rules_examples = ds.load_rules_examples()
        print(f"[Load] RULES examples: {len(rules_examples)}")
    if "tensortrust" in datasets_to_run:
        hijacking_df = ds.download_hijacking_data()
        print(f"[Load] TensorTrust hijacking: {len(hijacking_df)}")
    if "purple" in datasets_to_run:
        purple_prompts = ds.download_purple_data()["prompts"]
        print(f"[Load] Purple prompts: {len(purple_prompts)}")
    if "gandalf" in datasets_to_run:
        gandalf_ds = ds.download_gandalf_data()
        print(f"[Load] Gandalf: {len(gandalf_ds)}")
    if "nessie" in datasets_to_run:
        nessie_df = ds.download_nessie_data()
        print(f"[Load] NESSiE raw rows: {len(nessie_df)}")
    if "struq" in datasets_to_run:
        struq_data = ds.load_struq_data(args.struq_data_path)
        print(f"[Load] StruQ data: {len(struq_data)}  ({args.struq_data_path})")
        struq_template = ds.load_prompt_template(args.prompt_templates_path, args.prompt_template_index)
        print(f"[Load] StruQ template: {args.prompt_templates_path} [index={args.prompt_template_index}]")

    for dataset_name in datasets_to_run:
        print(f"\n=== Evaluating {dataset_name} over {len(seeds)} seeds ===")

        if dataset_name == "tensortrust":
            rates = []
            for seed in seeds:
                seed_everything(seed)
                out_jsonl = os.path.join(args.out_dir, f"tensortrust_seed{seed}.jsonl")
                res = bm.eval_tensortrust(infer, hijacking_df, out_jsonl, gen_kwargs)
                rates.append(res["access_granted_rate"])
                print(f"[TensorTrust] seed={seed} access_granted_rate={res['access_granted_rate']*100:.2f}%")
            mean, std = mean_std_pct(rates)
            metrics["tensortrust_access_granted_rate"] = {"mean": mean, "std": std}

        elif dataset_name == "purple":
            rates = []
            for seed in seeds:
                seed_everything(seed)
                out_jsonl = os.path.join(args.out_dir, f"purple_seed{seed}.jsonl")
                res = bm.eval_purple(infer, purple_prompts, out_jsonl, gen_kwargs)
                rates.append(res["purple_rate"])
                print(f"[Purple] seed={seed} purple_rate={res['purple_rate']*100:.2f}%")
            mean, std = mean_std_pct(rates)
            metrics["purple_contains_rate"] = {"mean": mean, "std": std}

        elif dataset_name == "gandalf":
            rates = []
            for seed in seeds:
                seed_everything(seed)
                out_jsonl = os.path.join(args.out_dir, f"gandalf_seed{seed}.jsonl")
                res = bm.eval_gandalf(infer, gandalf_ds, out_jsonl, gen_kwargs)
                rates.append(res["planetary_rate"])
                print(f"[Gandalf] seed={seed} planetary_rate={res['planetary_rate']*100:.2f}%")
            mean, std = mean_std_pct(rates)
            metrics["gandalf_contains_rate"] = {"mean": mean, "std": std}

        elif dataset_name == "nessie":
            safe_rates, helpful_rates, sh_rates = [], [], []
            for seed in seeds:
                seed_everything(seed)
                out_jsonl = os.path.join(args.out_dir, f"nessie_seed{seed}.jsonl")
                res = bm.eval_nessie(
                    infer, nessie_df, out_jsonl, gen_kwargs, seed=seed,
                    limit=args.nessie_limit, strict_helpful=bool(args.nessie_strict_helpful),
                )
                safe_rates.append(res["safe_rate"])
                helpful_rates.append(res["helpful_rate"])
                sh_rates.append(res["sh_rate"])
                print(f"[NESSiE] seed={seed} safe={res['safe_rate']*100:.2f}% "
                      f"helpful={res['helpful_rate']*100:.2f}% sh={res['sh_rate']*100:.2f}% (groups={res['n_groups']})")
            safe_mean, safe_std = mean_std_pct(safe_rates)
            help_mean, help_std = mean_std_pct(helpful_rates)
            sh_mean, sh_std = mean_std_pct(sh_rates)
            metrics["nessie_safe"] = {"mean": safe_mean, "std": safe_std}
            metrics["nessie_helpful"] = {"mean": help_mean, "std": help_std}
            metrics["nessie_sh"] = {"mean": sh_mean, "std": sh_std}

        elif dataset_name == "rules":
            frs = []
            for seed in seeds:
                seed_everything(seed)
                out_jsonl = os.path.join(args.out_dir, f"rules_seed{seed}.jsonl")
                res = bm.eval_rules(infer, rules_examples, out_jsonl, gen_kwargs)
                frs.append(res["failure_rate"])
                print(f"[RULES] seed={seed} failure_rate={res['failure_rate']*100:.2f}%  ({res['failures']}/{res['num_samples']})")
            mean, std = mean_std_pct(frs)
            metrics["rules_failure_rate"] = {"mean": mean, "std": std}

        elif dataset_name == "struq":
            domains = ds.STRUQ_DOMAIN_TYPES if args.struq_domain == "all" else [args.struq_domain]
            attacks = ds.STRUQ_ATTACK_TYPES if args.struq_attack == "all" else [args.struq_attack]
            for d in domains:
                for a in attacks:
                    asrs, robs = [], []
                    for seed in seeds:
                        seed_everything(seed)
                        out_jsonl = os.path.join(args.out_dir, f"struq_{d}_{a}_seed{seed}.jsonl")
                        res = bm.eval_struq(infer, struq_data, out_jsonl, d, a, gen_kwargs, struq_template)
                        asrs.append(res["attack_success_rate"])
                        robs.append(res["robust_accuracy"])
                        print(f"[StruQ] seed={seed} {d}/{a} ASR={res['attack_success_rate']*100:.2f}%  RobustAcc={res['robust_accuracy']*100:.2f}%")
                    m_asr, s_asr = mean_std_pct(asrs)
                    m_rob, s_rob = mean_std_pct(robs)
                    metrics[f"struq_{d}_{a}"] = {
                        "attack_success_rate": {"mean": m_asr, "std": s_asr},
                        "robust_accuracy": {"mean": m_rob, "std": s_rob},
                    }

        elif dataset_name == "ifeval":
            inst_strict_list, inst_loose_list = [], []
            prompt_strict_list, prompt_loose_list = [], []
            for seed in seeds:
                seed_everything(seed)
                out_jsonl = os.path.join(args.out_dir, f"ifeval_seed{seed}.jsonl")
                res = bm.eval_ifeval(
                    infer, out_jsonl, gen_kwargs, system_text=args.ifeval_system,
                    max_samples=args.ifeval_max_samples, subsample_seed=args.ifeval_subsample_seed,
                )
                inst_strict_list.append(res["inst_level_strict_acc"])
                inst_loose_list.append(res["inst_level_loose_acc"])
                prompt_strict_list.append(res["prompt_level_strict_acc"])
                prompt_loose_list.append(res["prompt_level_loose_acc"])
                print(
                    f"[IFEval] seed={seed} inst_strict={res['inst_level_strict_acc']*100:.2f}% "
                    f"inst_loose={res['inst_level_loose_acc']*100:.2f}% "
                    f"prompt_strict={res['prompt_level_strict_acc']*100:.2f}% "
                    f"prompt_loose={res['prompt_level_loose_acc']*100:.2f}% (n={res['num_samples']})"
                )
            m, s = mean_std_pct(inst_strict_list); metrics["ifeval_inst_level_strict_acc"] = {"mean": m, "std": s}
            m, s = mean_std_pct(inst_loose_list);  metrics["ifeval_inst_level_loose_acc"]  = {"mean": m, "std": s}
            m, s = mean_std_pct(prompt_strict_list); metrics["ifeval_prompt_level_strict_acc"] = {"mean": m, "std": s}
            m, s = mean_std_pct(prompt_loose_list);  metrics["ifeval_prompt_level_loose_acc"]  = {"mean": m, "std": s}

        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Final Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
