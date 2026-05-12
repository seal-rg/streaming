"""Benchmark evaluation suite for multi-stream models."""

import glob
import re
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"


def find_data_file(prefix: str, max_items: int | None = None,
                   data_dir: str | None = None) -> Path:
    """Find the best matching data file for a benchmark.

    Looks for files matching {prefix}_{N}.json where N is the item count.
    Returns the largest file that doesn't exceed max_items, or the largest
    available if max_items is None.

    Examples:
        find_data_file("harmbench", max_items=50)   -> harmbench_50.json
        find_data_file("harmbench", max_items=500)   -> harmbench_500.json (if exists)
        find_data_file("harmbench", max_items=None)  -> harmbench_500.json (largest)
    """
    base = Path(data_dir) if data_dir else _DATA_DIR
    pattern = str(base / f"{prefix}_*.json")
    candidates = []
    for path in glob.glob(pattern):
        m = re.search(rf"{prefix}_(\d+)", Path(path).stem)
        if m:
            n = int(m.group(1))
            candidates.append((n, Path(path)))

    if not candidates:
        # Fallback to hardcoded default
        return base / f"{prefix}_50.json"

    candidates.sort(key=lambda x: x[0])

    if max_items is None:
        # Return largest available
        return candidates[-1][1]

    # Return largest that doesn't exceed max_items
    best = candidates[0][1]  # smallest as fallback
    for n, path in candidates:
        if n <= max_items:
            best = path
        else:
            # This file is bigger than needed — only use it if nothing smaller exists
            break

    return best
