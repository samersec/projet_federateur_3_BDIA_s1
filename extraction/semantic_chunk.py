"""Data structures and JSONL loading for the semantic chunking pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = BASE_DIR / "extraction" / "extracted_pages.jsonl"
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "semantic_chunking.yaml"


@dataclass(frozen=True, slots=True)
class PageRecord:
    source: str
    page: int
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("PageRecord.source must be a non-empty string")
        if isinstance(self.page, bool) or not isinstance(self.page, int):
            raise ValueError("PageRecord.page must be an integer")
        if self.page < 1:
            raise ValueError("PageRecord.page must be greater than or equal to 1")
        if not isinstance(self.text, str):
            raise ValueError("PageRecord.text must be a string")


@dataclass(frozen=True, slots=True)
class SentenceRecord:
    source: str
    page: int
    sentence_index: int
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("SentenceRecord.source must be a non-empty string")
        if isinstance(self.page, bool) or not isinstance(self.page, int):
            raise ValueError("SentenceRecord.page must be an integer")
        if self.page < 1:
            raise ValueError("SentenceRecord.page must be greater than or equal to 1")
        if (
            isinstance(self.sentence_index, bool)
            or not isinstance(self.sentence_index, int)
        ):
            raise ValueError("SentenceRecord.sentence_index must be an integer")
        if self.sentence_index < 0:
            raise ValueError(
                "SentenceRecord.sentence_index must be greater than or equal to 0"
            )
        if not isinstance(self.text, str):
            raise ValueError("SentenceRecord.text must be a string")


@dataclass(frozen=True, slots=True)
class SentenceEmbedding:
    sentence: SentenceRecord
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class SentenceSimilarity:
    first_sentence: SentenceRecord
    second_sentence: SentenceRecord
    score: float


@dataclass(frozen=True, slots=True)
class SemanticBoundary:
    first_sentence: SentenceRecord
    second_sentence: SentenceRecord
    score: float
    threshold: float


def read_page_records(
    input_path: str | Path = DEFAULT_INPUT_PATH,
) -> Iterator[PageRecord]:
    """Read extracted pages from JSONL and yield validated PageRecord objects."""
    path = Path(input_path)
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.is_file():
        raise FileNotFoundError(f"Extracted pages file not found: {path}")

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                data: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc

            if not isinstance(data, dict):
                raise ValueError(
                    f"Line {line_number} of {path} must contain a JSON object"
                )

            missing_fields = [
                field
                for field in ("source", "page", "text")
                if field not in data
            ]
            if missing_fields:
                raise ValueError(
                    f"Line {line_number} of {path} is missing fields: "
                    f"{', '.join(missing_fields)}"
                )

            try:
                yield PageRecord(
                    source=data["source"],
                    page=data["page"],
                    text=data["text"],
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid PageRecord on line {line_number} of {path}: {exc}"
                ) from exc


def split_into_sentences(page: PageRecord) -> list[SentenceRecord]:
    """Split one page into sentences while preserving page provenance."""
    if not page.text.strip():
        return []

    closing_punctuation = {"\"", "'", ")", "]", "}", "»"}
    sentence_starts = 0
    sentence_index = 0
    sentences: list[SentenceRecord] = []

    def add_sentence(start: int, end: int) -> None:
        nonlocal sentence_index
        text = page.text[start:end].strip()
        if not text:
            return

        sentences.append(
            SentenceRecord(
                source=page.source,
                page=page.page,
                sentence_index=sentence_index,
                text=text,
            )
        )
        sentence_index += 1

    index = 0
    while index < len(page.text):
        character = page.text[index]

        if character not in ".?!؟؛":
            index += 1
            continue

        end = index + 1

        while end < len(page.text) and page.text[end] in closing_punctuation:
            end += 1

        if character == ".":
            previous_character = page.text[index - 1] if index > 0 else ""
            next_character = page.text[end] if end < len(page.text) else ""
            isolated_dot = previous_character.isalnum() and (
                not next_character or next_character.isspace()
            )

            if (
                previous_character.isdigit()
                and next_character.isdigit()
            ) or (
                previous_character.isalpha()
                and next_character.isalpha()
            ):
                isolated_dot = False

            if not isolated_dot:
                index += 1
                continue

        elif character in "?!":
            next_character = page.text[end] if end < len(page.text) else ""
            if next_character and not next_character.isspace():
                index += 1
                continue

        add_sentence(sentence_starts, end)
        sentence_starts = end
        index = end

    if sentence_starts < len(page.text):
        add_sentence(sentence_starts, len(page.text))

    return sentences


def load_semantic_chunking_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the semantic chunking YAML configuration."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the chunking configuration") from exc

    path = Path(config_path)
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.is_file():
        raise FileNotFoundError(f"Semantic chunking configuration not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Semantic chunking configuration must be a YAML mapping")

    required_fields = (
        "model_name",
        "similarity_threshold",
        "min_size",
        "max_size",
        "overlap",
        "batch_size",
        "device",
        "input_path",
        "output_path",
    )
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(
            "Semantic chunking configuration is missing fields: "
            + ", ".join(missing_fields)
        )

    return config


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _load_sentence_transformer_class() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required; install it before loading BGE-M3"
        ) from exc
    return SentenceTransformer


def resolve_device(device: str, cuda_available: bool | None = None) -> str:
    """Resolve an auto/cpu/cuda device setting."""
    normalized_device = device.strip().lower()
    if normalized_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "device must be one of 'auto', 'cpu', or 'cuda'"
        )

    available = _cuda_is_available() if cuda_available is None else cuda_available

    if normalized_device == "cuda":
        if not available:
            raise RuntimeError(
                "CUDA was requested but CUDA is not available on this system"
            )
        return "cuda"

    if normalized_device == "auto":
        return "cuda" if available else "cpu"

    return "cpu"


class BGE3Embedder:
    """Load BGE-M3 once and encode SentenceRecord objects in batches."""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        model: Any | None = None,
        cuda_available: bool | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than or equal to 1")

        self.model_name = model_name
        self.device = resolve_device(device, cuda_available=cuda_available)
        self.batch_size = batch_size

        if model is not None:
            self.model = model
            return

        SentenceTransformer = _load_sentence_transformer_class()
        self.model = SentenceTransformer(model_name, device=self.device)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        model: Any | None = None,
        cuda_available: bool | None = None,
    ) -> "BGE3Embedder":
        """Create an embedder from a semantic-chunking configuration mapping."""
        model_name = config.get("model_name")
        device = config.get("device")
        batch_size = config.get("batch_size")

        if not isinstance(model_name, str):
            raise ValueError("model_name must be a string")
        if not isinstance(device, str):
            raise ValueError("device must be a string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer")

        return cls(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model=model,
            cuda_available=cuda_available,
        )

    def embed_sentences(
        self,
        sentences: list[SentenceRecord],
    ) -> list[SentenceEmbedding]:
        """Encode sentences in batches and preserve each sentence association."""
        if not sentences:
            return []

        sentence_embeddings: list[SentenceEmbedding] = []

        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start:start + self.batch_size]
            batch_text = [sentence.text for sentence in batch]

            encoded = self.model.encode(
                batch_text,
                batch_size=len(batch),
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            embeddings = np.asarray(encoded)

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if embeddings.shape[0] != len(batch):
                raise RuntimeError(
                    "The embedding model returned a row count that does "
                    "not match the sentence batch"
                )

            for sentence, embedding in zip(batch, embeddings, strict=True):
                sentence_embeddings.append(
                    SentenceEmbedding(
                        sentence=sentence,
                        embedding=np.asarray(embedding, dtype=np.float32),
                    )
                )

        return sentence_embeddings


def cosine_similarity_between(
    first_embedding: np.ndarray,
    second_embedding: np.ndarray,
) -> float:
    """Calculate cosine similarity between two compatible embeddings."""
    first = np.asarray(first_embedding, dtype=np.float32)
    second = np.asarray(second_embedding, dtype=np.float32)

    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Embeddings must be one-dimensional vectors")
    if first.shape != second.shape:
        raise ValueError(
            "Embedding dimensions must match: "
            f"{first.shape[0]} != {second.shape[0]}"
        )
    if first.size == 0:
        raise ValueError("Embeddings must not be empty")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Embeddings must contain only finite values")

    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    return float(np.dot(first, second) / (first_norm * second_norm))


def calculate_consecutive_similarities(
    sentence_embeddings: list[SentenceEmbedding],
) -> list[SentenceSimilarity]:
    """Calculate cosine similarity only between consecutive sentence embeddings."""
    if len(sentence_embeddings) < 2:
        return []

    similarities: list[SentenceSimilarity] = []

    for first, second in zip(
        sentence_embeddings,
        sentence_embeddings[1:],
        strict=False,
    ):
        score = cosine_similarity_between(first.embedding, second.embedding)
        similarities.append(
            SentenceSimilarity(
                first_sentence=first.sentence,
                second_sentence=second.sentence,
                score=score,
            )
        )

    return similarities


def detect_semantic_boundaries(
    similarities: list[SentenceSimilarity],
    similarity_threshold: float,
) -> list[SemanticBoundary]:
    """Detect candidate semantic boundaries below a similarity threshold."""
    if isinstance(similarity_threshold, bool) or not isinstance(
        similarity_threshold,
        (int, float),
    ):
        raise ValueError("similarity_threshold must be a number")

    threshold = float(similarity_threshold)
    if not np.isfinite(threshold):
        raise ValueError("similarity_threshold must be finite")
    if threshold < -1.0 or threshold > 1.0:
        raise ValueError("similarity_threshold must be between -1.0 and 1.0")

    boundaries: list[SemanticBoundary] = []

    for similarity in similarities:
        if not np.isfinite(similarity.score):
            raise ValueError("Similarity scores must contain only finite values")

        if similarity.score < threshold:
            boundaries.append(
                SemanticBoundary(
                    first_sentence=similarity.first_sentence,
                    second_sentence=similarity.second_sentence,
                    score=similarity.score,
                    threshold=threshold,
                )
            )

    return boundaries
