"""Real image-pixel embeddings from Qwen3-VL-Embedding-2B.

The published checkpoint is a ``Qwen3VLForConditionalGeneration`` (weights are
prefixed with ``model.``), while the Sentence-Transformers wrapper loads the
base ``Qwen3VLModel`` without that prefix and silently leaves every weight
randomly initialized.  Loading the conditional-generation class and using its
base model directly keeps the real pretrained weights; pooling is the same
last-token pool + L2 normalization the model card specifies.

We also avoid the ``Qwen3VLProcessor`` video branch (it requires torchvision)
by composing the tokenizer and the Qwen2-VL image processor ourselves.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import ProcessorMixin

from knowledge_scope.shared.config import Settings

logger = logging.getLogger(__name__)

VISUAL_EMBEDDING_SCHEMA_VERSION = "1.0"
DEFAULT_VISUAL_DIMENSION = 2048
DEFAULT_VISUAL_INSTRUCTION = "Represent the user's input."
_TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class VisualEmbeddingError(RuntimeError):
    """Raised when the visual embedding model cannot be loaded or used."""


@dataclass(frozen=True)
class VisualEmbeddingContract:
    schema_version: str
    model_id: str
    model_path: str
    dtype: str
    normalize: bool
    dimension: int
    query_prompt: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "dtype": self.dtype,
            "normalize": self.normalize,
            "dimension": self.dimension,
            "query_prompt": self.query_prompt,
        }


@dataclass(frozen=True)
class VisualEncodeStats:
    batch_count: int
    latency_ms: float
    vram_peak_mb: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_count": self.batch_count,
            "latency_ms": round(self.latency_ms, 2),
            "vram_peak_mb": None if self.vram_peak_mb is None else round(self.vram_peak_mb, 1),
        }


class _ImageOnlyQwen3VLProcessor(Qwen3VLProcessor):
    """Qwen3-VL processor without the torchvision-dependent video branch."""

    attributes: ClassVar[list[str]] = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: Any = None,
        tokenizer: Any = None,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template)
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = "<|video_pad|>"
        image_token_id = getattr(tokenizer, "image_token_id", None)
        self.image_token_id = image_token_id or tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        self.vision_start_token_id = getattr(
            tokenizer, "vision_start_token_id", None
        ) or tokenizer.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = getattr(
            tokenizer, "vision_end_token_id", None
        ) or tokenizer.convert_tokens_to_ids(self.vision_end_token)


def _conversation(
    *,
    text: str | None,
    image: Image.Image | None,
    instruction: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]
    if image is not None:
        content.append(
            {
                "type": "image",
                "image": image,
                "min_pixels": 4096,
                "max_pixels": 1_843_200,
            }
        )
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": "NULL"})
    return messages


def _current_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / 1024**2


class VisualEmbeddingEncoder:
    """Lazy, thread-safe encoder producing normalized visual/text vectors."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._dimension = DEFAULT_VISUAL_DIMENSION
        self._load_ms = 0.0
        self._contract = VisualEmbeddingContract(
            schema_version=VISUAL_EMBEDDING_SCHEMA_VERSION,
            model_id=settings.visual_model_id,
            model_path=settings.visual_model_path,
            dtype=settings.visual_dtype,
            normalize=True,
            dimension=DEFAULT_VISUAL_DIMENSION,
            query_prompt=settings.visual_query_prompt,
        )

    @property
    def contract(self) -> VisualEmbeddingContract:
        return self._contract

    @property
    def load_latency_ms(self) -> float:
        return self._load_ms

    def _model_source(self) -> str:
        return self._settings.visual_model_path or self._settings.visual_model_id

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            started = time.perf_counter()
            try:
                dtype = _TORCH_DTYPES[self._settings.visual_dtype]
                source = self._model_source()
                tokenizer = AutoTokenizer.from_pretrained(source, padding_side="right")
                image_processor = AutoImageProcessor.from_pretrained(source)
                processor = _ImageOnlyQwen3VLProcessor(
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                )
                causal = Qwen3VLForConditionalGeneration.from_pretrained(
                    source,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                model = causal.model.to(self._settings.visual_device).eval()
                if len(model.language_model.layers) == 0:  # pragma: no cover - defensive
                    raise VisualEmbeddingError("visual embedding model has no language layers")
            except VisualEmbeddingError:
                raise
            except Exception as error:
                raise VisualEmbeddingError(
                    f"unable to load visual embedding model {source!r}: {error}"
                ) from error
            self._tokenizer = tokenizer
            self._processor = processor
            self._model = model
            self._dimension = int(model.language_model.config.hidden_size)
            self._load_ms = (time.perf_counter() - started) * 1000.0
            self._contract = VisualEmbeddingContract(
                schema_version=VISUAL_EMBEDDING_SCHEMA_VERSION,
                model_id=self._settings.visual_model_id,
                model_path=self._settings.visual_model_path,
                dtype=self._settings.visual_dtype,
                normalize=True,
                dimension=self._dimension,
                query_prompt=self._settings.visual_query_prompt,
            )
            logger.info(
                "visual embedding model loaded: source=%s dim=%s load_ms=%.1f",
                source,
                self._dimension,
                self._load_ms,
            )

    @torch.no_grad()
    def _encode_batch(
        self,
        *,
        texts: Sequence[str] | None,
        images: Sequence[Image.Image] | None,
        instruction: str,
    ) -> np.ndarray:
        if self._model is None:
            self._load()
        assert self._tokenizer is not None
        assert self._processor is not None
        assert self._model is not None
        if images is not None:
            conversations = [
                _conversation(text=None, image=image, instruction=instruction) for image in images
            ]
            image_arg = list(images)
        else:
            assert texts is not None
            conversations = [
                _conversation(text=text, image=None, instruction=instruction) for text in texts
            ]
            image_arg = None
        prompt = self._tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._processor(
            text=prompt,
            images=image_arg,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._settings.visual_device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }
        output = self._model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        hidden = output.last_hidden_state
        attention_mask = inputs["attention_mask"]
        flipped = attention_mask.flip(dims=[1])
        last_positions = flipped.argmax(dim=1)
        columns = attention_mask.shape[1] - last_positions - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        embeddings = hidden[rows, columns]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.float().cpu().numpy()

    def _vram_peak_mb(self) -> float | None:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / 1024**2

    def _batched(
        self,
        *,
        texts: Sequence[str] | None,
        images: Sequence[Image.Image] | None,
        instruction: str,
    ) -> tuple[list[list[float]], VisualEncodeStats]:
        items: Sequence[Any] = images if images is not None else texts or []
        if not items:
            return [], VisualEncodeStats(batch_count=0, latency_ms=0.0, vram_peak_mb=None)
        started = time.perf_counter()
        batch_size = max(1, int(self._settings.visual_batch_size))
        vectors: list[list[float]] = []
        batches = 0
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            if images is not None:
                array = self._encode_batch(texts=None, images=chunk, instruction=instruction)
            else:
                array = self._encode_batch(texts=list(chunk), images=None, instruction=instruction)
            vectors.extend(array.tolist())
            batches += 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        return vectors, VisualEncodeStats(
            batch_count=batches,
            latency_ms=latency_ms,
            vram_peak_mb=self._vram_peak_mb(),
        )

    def encode_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], VisualEncodeStats]:
        return self._batched(
            texts=texts,
            images=None,
            instruction=DEFAULT_VISUAL_INSTRUCTION,
        )

    def encode_query(self, query: str) -> tuple[list[float], VisualEncodeStats]:
        vectors, stats = self._batched(
            texts=[query],
            images=None,
            instruction=self._settings.visual_query_prompt,
        )
        return vectors[0], stats

    def encode_images(
        self,
        images: Sequence[bytes],
    ) -> tuple[list[list[float]], VisualEncodeStats]:
        max_pixels = int(self._settings.visual_max_image_pixels)
        prepared: list[Image.Image] = []
        for payload in images:
            with Image.open(io.BytesIO(payload)) as handle:
                image = handle.convert("RGB")
            if image.width * image.height > max_pixels:
                scale = (max_pixels / float(image.width * image.height)) ** 0.5
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                )
            prepared.append(image)
        return self._batched(
            texts=None,
            images=prepared,
            instruction=DEFAULT_VISUAL_INSTRUCTION,
        )

    def vram_peak_mb(self) -> float | None:
        return self._vram_peak_mb()

    def embedding_dimension(self) -> int:
        if self._model is None:
            self._load()
        return self._dimension

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "DEFAULT_VISUAL_DIMENSION",
    "DEFAULT_VISUAL_INSTRUCTION",
    "VISUAL_EMBEDDING_SCHEMA_VERSION",
    "VisualEmbeddingContract",
    "VisualEmbeddingEncoder",
    "VisualEmbeddingError",
    "VisualEncodeStats",
]
