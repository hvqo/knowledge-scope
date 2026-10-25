"""Process-wide guard for heavyweight local model construction.

Transformers/SentenceTransformers model construction is not thread-safe:
concurrent ``from_pretrained`` calls can leak an ``init_empty_weights``
context into another thread and leave that model on the meta device. Only
model construction needs to be serialized; inference remains parallel.
"""

from __future__ import annotations

import threading

MODEL_LOADING_LOCK = threading.RLock()

__all__ = ["MODEL_LOADING_LOCK"]
