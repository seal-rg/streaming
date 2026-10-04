#!/usr/bin/env python3
"""Entry point for the Vanilla (plain AutoModelForCausalLM) backend.
Equivalent to: python -m security_bench.cli --backend vanilla ..."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from security_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["--backend", "vanilla"] + sys.argv[1:])
