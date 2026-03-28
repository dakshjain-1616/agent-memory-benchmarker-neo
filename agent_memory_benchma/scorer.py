"""Accuracy scoring: exact-match + semantic similarity."""

import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_EXACT_WEIGHT = float(os.getenv("SCORER_EXACT_WEIGHT", "0.4"))
_SEMANTIC_WEIGHT = float(os.getenv("SCORER_SEMANTIC_WEIGHT", "0.6"))


def _tokenise(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenise(a), _tokenise(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


class Scorer:
    """Combines exact-match and semantic similarity into a single accuracy score."""

    def __init__(
        self,
        exact_weight: float = _EXACT_WEIGHT,
        semantic_weight: float = _SEMANTIC_WEIGHT,
        mock_mode: bool = False,
    ):
        self.exact_weight = exact_weight
        self.semantic_weight = semantic_weight
        self.mock_mode = mock_mode
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return
        if self.mock_mode:
            return
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning("Could not load SentenceTransformer (%s); falling back to Jaccard.", exc)
            self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exact_match(self, response: str, expected: str) -> float:
        """Word-overlap score in [0, 1] with substring bonus."""
        if not expected:
            return 1.0
        overlap = _jaccard(response, expected)
        substring_bonus = 0.2 if expected.lower() in response.lower() else 0.0
        return min(1.0, overlap + substring_bonus)

    def semantic_similarity(self, response: str, expected: str) -> float:
        """Cosine similarity via sentence-transformers, or Jaccard fallback."""
        self._load_model()
        if self._model is None:
            return _jaccard(response, expected)
        try:
            import numpy as np
            embs = self._model.encode([response, expected], normalize_embeddings=True)
            return float(np.dot(embs[0], embs[1]))
        except Exception as exc:
            logger.debug("Semantic similarity failed (%s); using Jaccard.", exc)
            return _jaccard(response, expected)

    def score_response(self, response: str, expected: str) -> float:
        """Weighted combination of exact_match and semantic_similarity."""
        exact = self.exact_match(response, expected)
        semantic = self.semantic_similarity(response, expected)
        return self.exact_weight * exact + self.semantic_weight * semantic
