"""Env vars, one place.

Deferred from M1 on purpose: one variable did not need a module. M2 has four.

No pydantic-settings — four `os.environ` reads do not justify a dependency, and
this keeps `config.py` importable without a web framework the way `model.py` is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .model import resolve_model_dir


@dataclass(frozen=True, slots=True)
class Settings:
    model_dir: Path
    num_threads: int | None
    max_batch_size: int
    log_level: str


@lru_cache
def get_settings() -> Settings:
    """Read once. Cached, so every caller sees the same object."""
    threads = os.environ.get("NUM_THREADS")
    return Settings(
        # The one resolution path. `resolve_model_dir` already does explicit arg
        # → $MODEL_DIR → checkout; a second copy here is how those drift.
        model_dir=resolve_model_dir(),
        # Unset means torch's default (every core). Pinning it to 1 helps p95
        # under concurrency, but M3 has to measure both, so there is no default.
        num_threads=int(threads) if threads else None,
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", 32)),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )
