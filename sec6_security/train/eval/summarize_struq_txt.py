#!/usr/bin/env python3

import glob
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

# =============================
# Regex for StruQ metric line
# =============================
RE_STRUQ_METRIC = re.compile(
    r"\[StruQ\]\s*seed=(\d+)\s+([A-Za-z0-9_\-]+)\/([A-Za-z0-9_\-]+)\s+ASR=([\d.]+)%\s+RobustAcc=([\d.]+)%",
    re.M,
)


# =============================
# Parse one file
# =============================
def parse_one_file(file_path):
    text = Path(file_path).read_text(errors="ignore")
    rows = []
    for m in RE_STRUQ_METRIC.finditer(text):
        seed = int(m.group(1))
        domain = m.group(2)
        attack = m.group(3)
        asr = float(m.group(4))
        racc = float(m.group(5))
        rows.append((seed, domain, attack, asr, racc))
    return rows


# =============================
# Aggregate
# =============================
def aggregate(rows):
    by_cat = defaultdict(list)
    by_domain = defaultdict(list)

    for seed, domain, attack, asr, racc in rows:
        by_cat[(domain, attack)].append((asr, racc))
        by_domain[domain].append((asr, racc))

    def summarize(pairs):
        asr_vals = [x for x, _ in pairs]
        racc_vals = [y for _, y in pairs]

        return (
            mean(asr_vals),
            stdev(asr_vals) if len(asr_vals) > 1 else 0.0,
            mean(racc_vals),
            stdev(racc_vals) if len(racc_vals) > 1 else 0.0,
            len(pairs),
        )

    cat_summary = []
    for (domain, attack), pairs in sorted(by_cat.items()):
        cat_summary.append((f"{domain}/{attack}", *summarize(pairs)))

    domain_summary = []
    for domain, pairs in sorted(by_domain.items()):
        domain_summary.append((domain, *summarize(pairs)))

    return cat_summary, domain_summary


# =============================
# Collect files (robust glob)
# =============================
def collect_files(inputs):
    files = []
    for inp in inputs:
        matches = glob.glob(inp, recursive=True)

        for m in matches:
            p = Path(m)
            if p.is_file() and (p.suffix in [".out", ".txt"]):
                files.append(p)

        if Path(inp).is_dir():
            files.extend(Path(inp).rglob("*.out"))
            files.extend(Path(inp).rglob("*.txt"))

    return sorted(set(files))


# =============================
# Main
# =============================
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python summarize_struq_txt.py job.*.out")
        print("  python summarize_struq_txt.py /path/to/logs")
        sys.exit(1)

    files = collect_files(sys.argv[1:])

    if not files:
        print("No .out or .txt files found.")
        sys.exit(0)

    rows = []
    for f in files:
        rows.extend(parse_one_file(f))

    if not rows:
        print("Files found, but no StruQ metrics detected.")
        sys.exit(0)

    cat_summary, domain_summary = aggregate(rows)

    print("\n==============================")
    print(f"Parsed {len(rows)} metric rows from {len(files)} files")
    print("==============================\n")

    print("---- Category avg across seeds ----")
    print("category\tASR_mean\tASR_std\tRobustAcc_mean\tRobustAcc_std\tN")
    for cat, asr_m, asr_s, racc_m, racc_s, n in cat_summary:
        print(f"{cat}\t{asr_m:.4f}\t{asr_s:.4f}\t{racc_m:.4f}\t{racc_s:.4f}\t{n}")

    print("\n---- Domain avg (id / ood) ----")
    print("domain\tASR_mean\tASR_std\tRobustAcc_mean\tRobustAcc_std\tN")
    for domain, asr_m, asr_s, racc_m, racc_s, n in domain_summary:
        print(f"{domain}\t{asr_m:.4f}\t{asr_s:.4f}\t{racc_m:.4f}\t{racc_s:.4f}\t{n}")

    print("\nDone.")


if __name__ == "__main__":
    main()
