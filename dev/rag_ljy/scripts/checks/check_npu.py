#!/usr/bin/env python3
"""Compatibility entry point for the shared device smoke check."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from check_device import main

if __name__ == "__main__":
    main()
