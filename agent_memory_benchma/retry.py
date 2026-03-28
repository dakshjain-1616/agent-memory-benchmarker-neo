"""Exponential backoff retry utilities."""

import os
import time
import logging
import functools
from typing import Callable, Type, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_ATTEMPTS = int(os.getenv("BENCHMARK_RETRY_ATTEMPTS", "3"))
_DEFAULT_BASE_DELAY = float(os.getenv("BENCHMARK_RETRY_BASE_DELAY", "1.0"))


def with_retry(
    attempts: int = _DEFAULT_ATTEMPTS,
    base_delay: float = _DEFAULT_BASE_DELAY,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator that retries the wrapped function with exponential backoff."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return retry_call(fn, args=args, kwargs=kwargs,
                              attempts=attempts, base_delay=base_delay,
                              exceptions=exceptions)
        return wrapper
    return decorator


def retry_call(
    fn: Callable,
    args: tuple = (),
    kwargs: dict = None,
    attempts: int = _DEFAULT_ATTEMPTS,
    base_delay: float = _DEFAULT_BASE_DELAY,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Call *fn* with exponential backoff, raising the last exception on failure."""
    if kwargs is None:
        kwargs = {}
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:
            last_exc = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %d/%d failed (%s); retrying in %.1fs",
                           attempt, attempts, exc, delay)
            time.sleep(delay)
    raise last_exc
