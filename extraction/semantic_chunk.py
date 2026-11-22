"""Data structures and JSONL loading for the semantic chunking pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = BASE_DIR / "extraction" / "extracted_pages.jsonl"


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
