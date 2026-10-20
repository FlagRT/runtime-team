#!/usr/bin/env python3
"""Benchmark BM25/dense in either execution mode; defaults to sequential."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from benchmark_sequential import main


if __name__ == "__main__":
    main()
