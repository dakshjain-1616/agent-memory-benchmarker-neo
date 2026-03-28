"""Exponential-decay memory freshness / staleness tracker."""

import os
import math
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_HALFLIFE_SECONDS = float(os.getenv("MEMORY_DECAY_HALFLIFE", "3600"))
_LAMBDA = math.log(2) / _HALFLIFE_SECONDS  # decay constant


def compute_staleness(age_seconds: float, halflife: float = _HALFLIFE_SECONDS) -> float:
    """Return staleness ∈ [0, 1]; 0 = perfectly fresh, 1 = completely stale."""
    lam = math.log(2) / halflife
    freshness = math.exp(-lam * age_seconds)
    return 1.0 - freshness


class StalenessTracker:
    """Track memory-addition timestamps per backend and report freshness metrics."""

    def __init__(self, halflife: float = _HALFLIFE_SECONDS):
        self.halflife = halflife
        # backend_name -> list of addition timestamps (epoch seconds)
        self._additions: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_addition(self, backend_name: str, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self._additions.setdefault(backend_name, []).append(ts)

    def simulate_aging(self, backend_name: str, age_seconds: float) -> None:
        """Shift all timestamps backward to simulate passage of time."""
        if backend_name not in self._additions:
            return
        self._additions[backend_name] = [
            ts - age_seconds for ts in self._additions[backend_name]
        ]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_staleness_report(self, backend_name: str) -> Dict:
        """Return freshness metrics for a single backend."""
        timestamps = self._additions.get(backend_name, [])
        if not timestamps:
            return {"backend": backend_name, "count": 0,
                    "avg_freshness": 1.0, "avg_staleness": 0.0,
                    "min_freshness": 1.0, "max_staleness": 0.0}
        now = time.time()
        freshnesses = [
            math.exp(-math.log(2) / self.halflife * (now - ts))
            for ts in timestamps
        ]
        avg_fresh = sum(freshnesses) / len(freshnesses)
        return {
            "backend": backend_name,
            "count": len(timestamps),
            "avg_freshness": avg_fresh,
            "avg_staleness": 1.0 - avg_fresh,
            "min_freshness": min(freshnesses),
            "max_staleness": 1.0 - min(freshnesses),
        }

    def get_all_reports(self) -> Dict[str, Dict]:
        """Return staleness reports for every tracked backend."""
        return {name: self.get_staleness_report(name) for name in self._additions}

    def clear(self, backend_name: Optional[str] = None) -> None:
        if backend_name:
            self._additions.pop(backend_name, None)
        else:
            self._additions.clear()
