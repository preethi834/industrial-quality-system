"""Named, independent random streams.

Every source of randomness in the simulator has its own stream, derived from the
scenario seed and a stable name. Two properties follow:

- **Determinism:** the same seed always gives the same numbers (no global state,
  no dependence on Python's per-process ``hash()`` salt).
- **Isolation:** a fault that changes how one signal is computed cannot shift the
  numbers drawn for any other signal, because each consumer draws a fixed number of
  values per event from its own stream. This is what makes the test "each fault
  changes only its intended signals" exact instead of statistical.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stream(seed: int, name: str) -> np.random.Generator:
    """Return a PCG64 generator for ``(seed, name)``; identical inputs give identical streams."""
    if seed < 0:
        raise ValueError("seed must be >= 0")
    key = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, key])))
