"""Data structures and JSONL loading for the semantic chunking pipeline."""

from __future__ import annotations

import json
import logging
import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = BASE_DIR / "extraction" / "extracted_pages.jsonl"
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "semantic_chunking.yaml"
LOGGER = logging.getLogger("semantic_chunk")


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


@dataclass(frozen=True, slots=True)
class SemanticChunk:
    chunk_id: str
    source: str
    text: str
    pages: list[int]
    start_page: int
    end_page: int
    char_count: int
    sentence_count: int


@dataclass(frozen=True, slots=True)
class PipelineStats:
    documents: int = 0
    sentences: int = 0
    chunks: int = 0
    empty_documents: int = 0
    errors: int = 0


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


def _joined_sentence_length(sentences: list[SentenceRecord]) -> int:
    return len(" ".join(sentence.text.strip() for sentence in sentences))


def _boundary_positions(
    sentences: list[SentenceRecord],
    boundaries: list[SemanticBoundary],
) -> dict[int, SemanticBoundary]:
    positions: dict[int, SemanticBoundary] = {}

    for sentence_index in range(1, len(sentences)):
        first = sentences[sentence_index - 1]
        second = sentences[sentence_index]

        for boundary in boundaries:
            if (
                boundary.first_sentence == first
                and boundary.second_sentence == second
            ):
                positions[sentence_index] = boundary
                break

    return positions


def _overlap_sentences(
    previous_sentences: list[SentenceRecord],
    overlap: int,
) -> list[SentenceRecord]:
    if overlap <= 0 or not previous_sentences:
        return []

    selected_reversed: list[SentenceRecord] = []
    selected_length = 0

    for sentence in reversed(previous_sentences):
        sentence_length = len(sentence.text.strip())
        separator_length = 1 if selected_reversed else 0
        if selected_length + separator_length + sentence_length > overlap:
            break

        selected_reversed.append(sentence)
        selected_length += separator_length + sentence_length

    return list(reversed(selected_reversed))


def build_semantic_chunks(
    sentences: list[SentenceRecord],
    boundaries: list[SemanticBoundary],
    min_size: int,
    max_size: int,
    overlap: int,
) -> list[SemanticChunk]:
    """Build semantic chunks without splitting individual sentences."""
    if not sentences:
        return []

    for name, value in (
        ("min_size", min_size),
        ("max_size", max_size),
        ("overlap", overlap),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be greater than or equal to 0")

    if min_size > max_size:
        raise ValueError("min_size must be less than or equal to max_size")
    if overlap >= max_size:
        raise ValueError("overlap must be smaller than max_size")

    source = sentences[0].source
    if any(sentence.source != source for sentence in sentences):
        raise ValueError("All sentences must belong to the same source document")

    boundary_by_position = _boundary_positions(sentences, boundaries)
    core_chunks: list[tuple[int, int]] = []
    start = 0

    while start < len(sentences):
        end = start + 1

        while end <= len(sentences):
            candidate_sentences = sentences[start:end]
            candidate_length = _joined_sentence_length(candidate_sentences)
            candidate_count = end - start

            if candidate_count == 1 and candidate_length > max_size:
                break

            if candidate_length >= max_size:
                available_boundaries = [
                    position
                    for position in boundary_by_position
                    if start < position <= end
                ]
                valid_boundaries = [
                    position
                    for position in available_boundaries
                    if position - start > 0
                    and _joined_sentence_length(
                        sentences[start:position]
                    ) >= min_size
                    and _joined_sentence_length(
                        sentences[start:position]
                    ) <= max_size
                ]

                if valid_boundaries:
                    split_at = min(
                        valid_boundaries,
                        key=lambda position: (
                            boundary_by_position[position].score,
                            -position,
                        ),
                    )
                else:
                    split_at = None
                    for position in range(end - 1, start, -1):
                        if _joined_sentence_length(
                            sentences[start:position]
                        ) >= min_size:
                            split_at = position
                            break

                if split_at is None:
                    split_at = end

                end = split_at
                break

            if (
                end in boundary_by_position
                and candidate_length >= min_size
                and end < len(sentences)
            ):
                break

            if end == len(sentences):
                break

            end += 1

        if end <= start:
            end = start + 1

        core_chunks.append((start, end))
        start = end

    core_chunks = [
        chunk
        for chunk in core_chunks
        if chunk[0] < chunk[1]
    ]

    if len(core_chunks) > 1:
        final_start, final_end = core_chunks[-1]
        final_length = _joined_sentence_length(
            sentences[final_start:final_end]
        )

        if final_length < min_size:
            previous_start, previous_end = core_chunks[-2]
            merged_length = _joined_sentence_length(
                sentences[previous_start:final_end]
            )

            if merged_length <= max_size:
                core_chunks[-2:] = [(previous_start, final_end)]

    document_stem = Path(source).stem
    output: list[SemanticChunk] = []

    for chunk_index, (start, end) in enumerate(core_chunks, start=1):
        core_sentences = sentences[start:end]
        emitted_sentences = list(core_sentences)

        if chunk_index > 1:
            previous_start, previous_end = core_chunks[chunk_index - 2]
            previous_sentences = sentences[previous_start:previous_end]
            emitted_sentences = (
                _overlap_sentences(previous_sentences, overlap)
                + emitted_sentences
            )

        text = " ".join(
            sentence.text.strip()
            for sentence in emitted_sentences
            if sentence.text.strip()
        )
        pages = sorted({
            sentence.page
            for sentence in emitted_sentences
        })

        output.append(
            SemanticChunk(
                chunk_id=f"{document_stem}-semantic-{chunk_index:06d}",
                source=source,
                text=text,
                pages=pages,
                start_page=pages[0],
                end_page=pages[-1],
                char_count=len(text),
                sentence_count=len(emitted_sentences),
            )
        )

    return output


def _semantic_chunk_to_record(chunk: SemanticChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "text": chunk.text,
        "pages": chunk.pages,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "char_count": chunk.char_count,
        "sentence_count": chunk.sentence_count,
    }


def _validate_pipeline_config(config: dict[str, Any]) -> tuple[
    str,
    str,
    float,
    int,
    int,
    int,
    int,
]:
    model_name = config.get("model_name")
    device = config.get("device")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(device, str):
        raise ValueError("device must be a string")

    similarity_threshold = config.get("similarity_threshold")
    min_size = config.get("min_size")
    max_size = config.get("max_size")
    overlap = config.get("overlap")
    batch_size = config.get("batch_size")

    numeric_config = {
        "similarity_threshold": similarity_threshold,
        "min_size": min_size,
        "max_size": max_size,
        "overlap": overlap,
        "batch_size": batch_size,
    }

    for name, value in numeric_config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")

    threshold = float(similarity_threshold)
    if not np.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
        raise ValueError("similarity_threshold must be a finite value from -1 to 1")

    min_size_int = int(min_size)
    max_size_int = int(max_size)
    overlap_int = int(overlap)
    batch_size_int = int(batch_size)

    if float(min_size) != min_size_int:
        raise ValueError("min_size must be an integer")
    if float(max_size) != max_size_int:
        raise ValueError("max_size must be an integer")
    if float(overlap) != overlap_int:
        raise ValueError("overlap must be an integer")
    if float(batch_size) != batch_size_int:
        raise ValueError("batch_size must be an integer")
    if min_size_int < 0 or max_size_int <= 0:
        raise ValueError("min_size and max_size must be positive or zero")
    if min_size_int > max_size_int:
        raise ValueError("min_size must be less than or equal to max_size")
    if overlap_int < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap_int >= max_size_int:
        raise ValueError("overlap must be smaller than max_size")
    if batch_size_int < 1:
        raise ValueError("batch_size must be greater than or equal to 1")

    return (
        model_name,
        device,
        threshold,
        min_size_int,
        max_size_int,
        overlap_int,
        batch_size_int,
    )


def _split_document_pages(
    source_pages: list[PageRecord],
) -> list[SentenceRecord]:
    page_order = sorted(source_pages, key=lambda record: record.page)
    sentences: list[SentenceRecord] = []

    for page in page_order:
        page_sentences = split_into_sentences(page)

        for sentence in page_sentences:
            sentences.append(
                SentenceRecord(
                    source=sentence.source,
                    page=sentence.page,
                    sentence_index=len(sentences),
                    text=sentence.text,
                )
            )

    return sentences


def _write_chunk(output_file: Any, chunk: SemanticChunk) -> None:
    record = _semantic_chunk_to_record(chunk)
    output_file.write(
        json.dumps(record, ensure_ascii=False) + "\n"
    )
    output_file.flush()


def run_semantic_chunking(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    embedder: BGE3Embedder | None = None,
) -> PipelineStats:
    """Run the complete semantic chunking pipeline incrementally."""
    config = load_semantic_chunking_config(config_path)
    (
        model_name,
        device,
        threshold,
        min_size,
        max_size,
        overlap,
        batch_size,
    ) = _validate_pipeline_config(config)

    resolved_input = (
        input_path
        if input_path is not None
        else config["input_path"]
    )
    resolved_output = (
        output_path
        if output_path is not None
        else config["output_path"]
    )

    input_file = Path(resolved_input)
    if not input_file.is_absolute():
        input_file = BASE_DIR / input_file

    output_file = Path(resolved_output)
    if not output_file.is_absolute():
        output_file = BASE_DIR / output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    documents = 0
    sentence_count = 0
    chunk_count = 0
    empty_documents = 0
    error_count = 0

    seen_sources: set[str] = set()
    active_source: str | None = None
    active_pages: list[PageRecord] = []

    def process_active_document(
        output_stream: Any,
    ) -> tuple[int, int, int]:
        nonlocal embedder

        if active_source is None:
            return 0, 0, 0

        sentences = _split_document_pages(active_pages)
        if not sentences:
            LOGGER.warning(
                "Document '%s' contains no usable sentences",
                active_source,
            )
            return 0, 0, 1

        if embedder is None:
            embedder = BGE3Embedder(
                model_name=model_name,
                device=device,
                batch_size=batch_size,
            )

        sentence_embeddings = embedder.embed_sentences(sentences)
        similarities = calculate_consecutive_similarities(sentence_embeddings)
        boundaries = detect_semantic_boundaries(similarities, threshold)
        chunks = build_semantic_chunks(
            sentences,
            boundaries,
            min_size,
            max_size,
            overlap,
        )

        for chunk in chunks:
            _write_chunk(output_stream, chunk)

        LOGGER.info(
            "Document '%s': %d sentences, %d chunks",
            active_source,
            len(sentences),
            len(chunks),
        )

        return len(sentences), len(chunks), 0

    with output_file.open("w", encoding="utf-8") as output_stream:
        try:
            for page in read_page_records(input_file):
                if active_source is None:
                    active_source = page.source

                if page.source != active_source:
                    documents += 1

                    try:
                        sentences_added, chunks_added, empty_added = (
                            process_active_document(output_stream)
                        )
                        sentence_count += sentences_added
                        chunk_count += chunks_added
                        empty_documents += empty_added
                    except Exception as exc:
                        error_count += 1
                        LOGGER.exception(
                            "Error processing document '%s': %s",
                            active_source,
                            exc,
                        )

                    seen_sources.add(active_source)
                    active_source = page.source
                    active_pages = []

                    if page.source in seen_sources:
                        raise ValueError(
                            "Extracted pages must be grouped contiguously "
                            f"by source; source '{page.source}' appeared again"
                        )

                active_pages.append(page)
        except Exception as exc:
            error_count += 1
            LOGGER.exception("Pipeline input error: %s", exc)
            raise

        if active_source is not None:
            documents += 1

            try:
                sentences_added, chunks_added, empty_added = (
                    process_active_document(output_stream)
                )
                sentence_count += sentences_added
                chunk_count += chunks_added
                empty_documents += empty_added
            except Exception as exc:
                error_count += 1
                LOGGER.exception(
                    "Error processing document '%s': %s",
                    active_source,
                    exc,
                )

    if documents == 0:
        LOGGER.warning("No extracted page records were found")

    stats = PipelineStats(
        documents=documents,
        sentences=sentence_count,
        chunks=chunk_count,
        empty_documents=empty_documents,
        errors=error_count,
    )

    LOGGER.info(
        "Semantic chunking complete: %d documents, %d sentences, "
        "%d chunks, %d empty documents, %d errors",
        stats.documents,
        stats.sentences,
        stats.chunks,
        stats.empty_documents,
        stats.errors,
    )

    return stats


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build semantic chunks with BGE-M3 embeddings",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the semantic chunking YAML configuration",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional override for the extracted-page JSONL path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional override for the semantic chunk JSONL path",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = build_argument_parser().parse_args()

    try:
        stats = run_semantic_chunking(
            config_path=args.config,
            input_path=args.input,
            output_path=args.output,
        )
        return 1 if stats.errors > 0 else 0
    except Exception as exc:
        LOGGER.exception("Semantic chunking failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
