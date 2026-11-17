"""Identifiers.

uuid7()          - default for every database record and event (RFC 9562 version 7).
deterministic_id - reproducible IDs for simulator entities (same seed -> same IDs).
"""

from __future__ import annotations

import secrets
import time
import uuid

# Fixed project namespace for UUIDv5. Never change it: simulator IDs depend on it.
_IQS_NAMESPACE = uuid.UUID("6f1c2a52-8d0e-4f43-9a4e-2b7b0d6c9e11")
_SEPARATOR = "\x1f"  # unit separator: ("a", "bc") and ("ab", "c") give different IDs


def uuid7() -> uuid.UUID:
    """48-bit Unix-millisecond timestamp + version/variant bits + 74 random bits.

    IDs sort by creation time across milliseconds; order within one millisecond
    is random (acceptable here: no caller relies on sub-millisecond order).
    """
    unix_ms = time.time_ns() // 1_000_000
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    value = (unix_ms & 0xFFFF_FFFF_FFFF) << 80 | 0x7 << 76 | rand_a << 64 | 0b10 << 62 | rand_b
    return uuid.UUID(int=value)


def uuid7_timestamp_ms(value: uuid.UUID) -> int:
    """Creation time (Unix ms) embedded in a UUIDv7."""
    if value.version != 7:
        raise ValueError(f"not a UUIDv7: {value}")
    return value.int >> 80


def deterministic_id(*parts: str) -> uuid.UUID:
    """UUIDv5 from the project namespace and the given parts."""
    if not parts:
        raise ValueError("deterministic_id needs at least one part")
    if any(_SEPARATOR in part for part in parts):
        raise ValueError("deterministic_id parts must not contain the unit separator (\\x1f)")
    return uuid.uuid5(_IQS_NAMESPACE, _SEPARATOR.join(parts))
