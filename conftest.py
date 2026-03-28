"""Root conftest.py — adds project root to sys.path for all tests."""

import sys
import os

# Ensure the project root (containing agent_memory_benchma/) is importable
sys.path.insert(0, os.path.dirname(__file__))
