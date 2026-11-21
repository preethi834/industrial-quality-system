"""Text description of a defect by a vision-language model — behind an interface.

Blueprint P2 Layer 1 asks for a VLM description; Rev 2 §3.9 requires approval for paid
APIs and model downloads, and none is approved. So the only provider is "none": no
describer exists, and the API omits the `description` field entirely. Nothing is faked.
A real provider (local Qwen-VL or an API) implements VLMDescriber and is added to
get_describer() once approved.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image


class VLMDescriber(Protocol):
    def describe(self, image: Image.Image, anomaly_map: np.ndarray, category: str) -> str: ...


def get_describer(provider: str) -> VLMDescriber | None:
    if provider == "none":
        return None
    raise ValueError(f"VLM provider {provider!r} is not available (needs approval)")
