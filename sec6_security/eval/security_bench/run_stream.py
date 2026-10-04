#!/usr/bin/env python3
"""Entry point for the Stream (Qwen3ForMultiStream) backend.
Equivalent to: python -m security_bench.cli --backend stream ..."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from security_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["--backend", "stream"] + sys.argv[1:])
