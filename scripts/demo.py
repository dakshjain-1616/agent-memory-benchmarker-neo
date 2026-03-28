#!/usr/bin/env python3
"""Thin wrapper — delegates to the root demo.py entry point."""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import main  # noqa: E402

if __name__ == "__main__":
    main()
