import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from extraction.semantic_chunk import (
    BGE3Embedder,
    PageRecord,
    SentenceRecord,
    SentenceEmbedding,
    SentenceSimilarity,
    calculate_consecutive_similarities,
    cosine_similarity_between,
    detect_semantic_boundaries,
    resolve_device,
    SemanticBoundary,
    SemanticChunk,
    build_semantic_chunks,
    _overlap_sentences,
    run_semantic_chunking,
    split_into_sentences,
)


class SplitIntoSentencesTests(unittest.TestCase):
    def test_splits_french_sentences_and_preserves_punctuation(self):
        page = PageRecord(
            source="decision.pdf",
            page=3,
            text="Le contrat est valide. La Cour confirme la responsabilité!",
        )

        sentences = split_into_sentences(page)

        self.assertEqual(
            [sentence.text for sentence in sentences],
            [
                "Le contrat est valide.",
                "La Cour confirme la responsabilité!",
            ],
        )
        self.assertEqual(
            [sentence.sentence_index for sentence in sentences],
            [0, 1],
        )

    def test_splits_arabic_sentences_and_preserves_punctuation(self):
        page = PageRecord(
            source="قرار.pdf",
            page=2,
            text="تم قبول الاستئناف؟ ال قرار صحيح؛ المحكمة تؤيد الحكم.",
        )

        sentences = split_into_sentences(page)

        self.assertEqual(
            [sentence.text for sentence in sentences],
            [
                "تم قبول الاستئناف؟",
                "ال قرار صحيح؛",
                "المحكمة تؤيد الحكم.",
            ],
        )

    def test_preserves_document_and_page_for_every_sentence(self):
        page = PageRecord(
            source="20624.pdf",
            page=7,
            text="المبدأ الأول. المبدأ الثاني.",
        )

        sentences = split_into_sentences(page)

        for sentence in sentences:
            self.assertEqual(sentence.source, "20624.pdf")
            self.assertEqual(sentence.page, 7)

    def test_does_not_split_decimal_numbers(self):
        page = PageRecord(
            source="decision.pdf",
            page=1,
            text="بلغت القيمة 5534.857 دينا ا.",
        )

        sentences = split_into_sentences(page)

        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0].text, page.text)

    def test_does_not_split_decimal_dates(self):
        page = PageRecord(
            source="decision.pdf",
            page=1,
            text="التقرير مؤرخ في 06.07.2020.",
        )

        sentences = split_into_sentences(page)

        self.assertEqual(len(sentences), 1)

    def test_does_not_split_abbreviated_legal_references(self):
        page = PageRecord(
            source="decision.pdf",
            page=1,
            text="طبق أحكام الفصل 123 م.م.م.ت.",
        )

        sentences = split_into_sentences(page)

        self.assertEqual(len(sentences), 1)

    def test_ignores_empty_text_and_blank_fragments(self):
        empty_page = PageRecord(source="decision.pdf", page=1, text="   ")
        blank_fragments_page = PageRecord(
            source="decision.pdf",
            page=2,
            text=" \n ",
        )

        self.assertEqual(split_into_sentences(empty_page), [])
        self.assertEqual(split_into_sentences(blank_fragments_page), [])


class MockSentenceTransformer:
    def __init__(self):
        self.encode_calls = []

    def encode(
        self,
        texts,
        batch_size,
        normalize_embeddings,
        convert_to_numpy,
    ):
        self.encode_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        return np.asarray(
            [[float(len(text)), 1.0] for text in texts],
            dtype=np.float32,
        )


class BGE3EmbedderTests(unittest.TestCase):
    def make_sentences(self, count):
        return [
            SentenceRecord(
                source="decision.pdf",
                page=4,
                sentence_index=index,
                text=f"Phrase numéro {index}",
            )
            for index in range(count)
        ]

    def test_resolves_auto_device_to_cpu_by_default(self):
        self.assertEqual(resolve_device("auto", cuda_available=False), "cpu")

    def test_resolves_auto_device_to_cuda_when_available(self):
        self.assertEqual(resolve_device("auto", cuda_available=True), "cuda")

    def test_cpu_device_always_uses_cpu(self):
        self.assertEqual(resolve_device("cpu", cuda_available=True), "cpu")

    def test_cuda_device_fails_when_cuda_is_unavailable(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA.*not available"):
            resolve_device("cuda", cuda_available=False)

    def test_rejects_unknown_device(self):
        with self.assertRaisesRegex(ValueError, "device must be"):
            resolve_device("gpu", cuda_available=False)

    def test_loads_mock_model_once_and_reuses_it(self):
        mock_model = MockSentenceTransformer()
        model_class_calls = []

        class MockModelClass:
            def __new__(cls, model_name, device):
                model_class_calls.append((model_name, device))
                return mock_model

        embedder = BGE3Embedder(
            model_name="BAAI/bge-m3",
            device="cpu",
            batch_size=2,
            model=mock_model,
        )

        with patch(
            "extraction.semantic_chunk._load_sentence_transformer_class",
            return_value=MockModelClass,
        ):
            embedder_loaded = BGE3Embedder(
                model_name="BAAI/bge-m3",
                device="cpu",
                batch_size=2,
            )

        embedder_loaded.embed_sentences(self.make_sentences(3))
        embedder_loaded.embed_sentences(self.make_sentences(1))

        self.assertEqual(model_class_calls, [("BAAI/bge-m3", "cpu")])
        self.assertEqual(len(mock_model.encode_calls), 3)

    def test_processes_sentences_in_batches_and_preserves_associations(self):
        mock_model = MockSentenceTransformer()
        embedder = BGE3Embedder(
            model_name="BAAI/bge-m3",
            device="cpu",
            batch_size=2,
            model=mock_model,
        )
        sentences = self.make_sentences(5)

        embeddings = embedder.embed_sentences(sentences)

        self.assertEqual(len(embeddings), 5)
        self.assertEqual(
            [result.sentence for result in embeddings],
            sentences,
        )
        self.assertEqual(
            [call["texts"] for call in mock_model.encode_calls],
            [
                ["Phrase numéro 0", "Phrase numéro 1"],
                ["Phrase numéro 2", "Phrase numéro 3"],
                ["Phrase numéro 4"],
            ],
        )
        for call in mock_model.encode_calls:
            self.assertTrue(call["normalize_embeddings"])
            self.assertTrue(call["convert_to_numpy"])

    def test_empty_sentence_list_does_not_encode(self):
        mock_model = MockSentenceTransformer()
        embedder = BGE3Embedder(
            model_name="BAAI/bge-m3",
            device="cpu",
            batch_size=2,
            model=mock_model,
        )

        embeddings = embedder.embed_sentences([])

        self.assertEqual(embeddings, [])
        self.assertEqual(mock_model.encode_calls, [])


class ConsecutiveSimilarityTests(unittest.TestCase):
    def make_sentence_embedding(self, sentence_index, embedding):
        return SentenceEmbedding(
            sentence=SentenceRecord(
                source="decision.pdf",
                page=5,
                sentence_index=sentence_index,
                text=f"Phrase {sentence_index}",
            ),
            embedding=np.asarray(embedding, dtype=np.float32),
        )

    def test_calculates_similarity_between_identical_normalized_vectors(self):
        score = cosine_similarity_between(
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
        )

        self.assertAlmostEqual(score, 1.0)

    def test_calculates_similarity_between_opposite_vectors(self):
        score = cosine_similarity_between(
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([-1.0, 0.0], dtype=np.float32),
        )

        self.assertAlmostEqual(score, -1.0)

    def test_rejects_incompatible_embedding_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimensions must match"):
            cosine_similarity_between(
                np.asarray([1.0, 0.0]),
                np.asarray([1.0, 0.0, 0.0]),
            )

    def test_rejects_two_dimensional_embedding_arrays(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            cosine_similarity_between(
                np.asarray([[1.0, 0.0]]),
                np.asarray([[1.0, 0.0]]),
            )

    def test_rejects_zero_vectors(self):
        with self.assertRaisesRegex(ValueError, "zero vectors"):
            cosine_similarity_between(
                np.asarray([0.0, 0.0]),
                np.asarray([1.0, 0.0]),
            )

    def test_empty_embedding_list_returns_no_results(self):
        similarities = calculate_consecutive_similarities([])

        self.assertEqual(similarities, [])

    def test_single_sentence_returns_no_results(self):
        sentence_embedding = self.make_sentence_embedding(
            0,
            [1.0, 0.0],
        )

        similarities = calculate_consecutive_similarities([sentence_embedding])

        self.assertEqual(similarities, [])

    def test_calculates_only_consecutive_pairs_and_preserves_records(self):
        first = self.make_sentence_embedding(0, [1.0, 0.0])
        second = self.make_sentence_embedding(1, [0.0, 1.0])
        third = self.make_sentence_embedding(2, [1.0, 0.0])

        similarities = calculate_consecutive_similarities([
            first,
            second,
            third,
        ])

        self.assertEqual(len(similarities), 2)
        self.assertIsInstance(similarities[0], SentenceSimilarity)
        self.assertIs(similarities[0].first_sentence, first.sentence)
        self.assertIs(similarities[0].second_sentence, second.sentence)
        self.assertAlmostEqual(similarities[0].score, 0.0)
        self.assertIs(similarities[1].first_sentence, second.sentence)
        self.assertIs(similarities[1].second_sentence, third.sentence)
        self.assertAlmostEqual(similarities[1].score, 0.0)


class SemanticBoundaryTests(unittest.TestCase):
    def make_similarity(self, first_index, second_index, score):
        return SentenceSimilarity(
            first_sentence=SentenceRecord(
                source="20624.pdf",
                page=first_index + 1,
                sentence_index=first_index,
                text=f"Phrase {first_index}",
            ),
            second_sentence=SentenceRecord(
                source="20624.pdf",
                page=second_index + 1,
                sentence_index=second_index,
                text=f"Phrase {second_index}",
            ),
            score=score,
        )

    def test_detects_scores_below_threshold_in_original_order(self):
        similarities = [
            self.make_similarity(0, 1, 0.85),
            self.make_similarity(1, 2, 0.60),
            self.make_similarity(2, 3, 0.71),
            self.make_similarity(3, 4, 0.40),
        ]

        boundaries = detect_semantic_boundaries(similarities, 0.72)

        self.assertEqual(len(boundaries), 3)
        self.assertIsInstance(boundaries[0], SemanticBoundary)
        self.assertEqual(boundaries[0].score, 0.60)
        self.assertEqual(boundaries[1].score, 0.71)
        self.assertEqual(boundaries[2].score, 0.40)
        self.assertEqual(
            [boundary.second_sentence.sentence_index for boundary in boundaries],
            [2, 3, 4],
        )

    def test_exact_threshold_is_not_a_boundary(self):
        similarities = [self.make_similarity(0, 1, 0.72)]

        boundaries = detect_semantic_boundaries(similarities, 0.72)

        self.assertEqual(boundaries, [])

    def test_preserves_source_and_page_provenance(self):
        similarity = self.make_similarity(0, 1, 0.30)

        boundaries = detect_semantic_boundaries([similarity], 0.72)

        self.assertEqual(boundaries[0].first_sentence.source, "20624.pdf")
        self.assertEqual(boundaries[0].second_sentence.source, "20624.pdf")
        self.assertEqual(boundaries[0].first_sentence.page, 1)
        self.assertEqual(boundaries[0].second_sentence.page, 2)

    def test_empty_similarity_list_returns_no_boundaries(self):
        boundaries = detect_semantic_boundaries([], 0.72)

        self.assertEqual(boundaries, [])

    def test_rejects_invalid_threshold_type(self):
        with self.assertRaisesRegex(ValueError, "must be a number"):
            detect_semantic_boundaries([], "0.72")

    def test_rejects_threshold_outside_cosine_range(self):
        with self.assertRaisesRegex(ValueError, "between -1.0 and 1.0"):
            detect_semantic_boundaries([], 1.5)

    def test_rejects_non_finite_similarity_score(self):
        similarity = self.make_similarity(0, 1, float("nan"))

        with self.assertRaisesRegex(ValueError, "finite values"):
            detect_semantic_boundaries([similarity], 0.72)


class SemanticChunkBuilderTests(unittest.TestCase):
    def make_sentence(self, index, text, page=None):
        return SentenceRecord(
            source="20624.pdf",
            page=page if page is not None else (1 + index // 2),
            sentence_index=index,
            text=text,
        )

    def make_boundary(self, first, second, score=0.50):
        return SemanticBoundary(
            first_sentence=first,
            second_sentence=second,
            score=score,
            threshold=0.72,
        )

    def test_empty_sentences_return_no_chunks(self):
        chunks = build_semantic_chunks([], [], 400, 700, 100)

        self.assertEqual(chunks, [])

    def test_boundary_is_delayed_below_minimum_size(self):
        sentences = [
            self.make_sentence(0, "A" * 100),
            self.make_sentence(1, "B" * 100),
            self.make_sentence(2, "C" * 100),
        ]

        boundary = self.make_boundary(sentences[0], sentences[1])
        chunks = build_semantic_chunks(
            sentences,
            [boundary],
            min_size=250,
            max_size=400,
            overlap=0,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].text,
            "A" * 100 + " " + "B" * 100 + " " + "C" * 100,
        )

    def test_semantic_boundary_splits_a_sufficient_chunk(self):
        sentences = [
            self.make_sentence(0, "A" * 200),
            self.make_sentence(1, "B" * 200),
            self.make_sentence(2, "C" * 400),
        ]

        boundary = self.make_boundary(sentences[1], sentences[2])
        chunks = build_semantic_chunks(
            sentences,
            [boundary],
            min_size=400,
            max_size=700,
            overlap=0,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].char_count, 401)
        self.assertEqual(chunks[1].char_count, 400)

    def test_max_size_uses_the_best_available_boundary(self):
        sentences = [
            self.make_sentence(0, "A" * 250),
            self.make_sentence(1, "B" * 250),
            self.make_sentence(2, "C" * 250),
        ]

        boundaries = [
            self.make_boundary(sentences[0], sentences[1], 0.40),
            self.make_boundary(sentences[1], sentences[2], 0.20),
        ]
        chunks = build_semantic_chunks(
            sentences,
            boundaries,
            min_size=400,
            max_size=700,
            overlap=0,
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertLessEqual(chunks[0].char_count, 700)
        self.assertLessEqual(chunks[1].char_count, 700)

    def test_no_boundary_falls_back_to_phrase_boundary(self):
        sentences = [
            self.make_sentence(0, "A" * 250),
            self.make_sentence(1, "B" * 250),
            self.make_sentence(2, "C" * 100),
        ]

        chunks = build_semantic_chunks(
            sentences,
            [],
            min_size=400,
            max_size=500,
            overlap=0,
        )

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertNotIn("\n", chunk.text)

    def test_individual_sentence_over_max_size_is_not_split(self):
        sentence = self.make_sentence(0, "A" * 800)

        chunks = build_semantic_chunks(
            [sentence],
            [],
            min_size=400,
            max_size=700,
            overlap=0,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "A" * 800)
        self.assertEqual(chunks[0].char_count, 800)

    def test_preserves_pages_and_creates_unique_deterministic_ids(self):
        sentences = [
            self.make_sentence(0, "A" * 200, page=1),
            self.make_sentence(1, "B" * 200, page=2),
            self.make_sentence(2, "C" * 400, page=3),
        ]
        boundary = self.make_boundary(sentences[1], sentences[2])

        first_run = build_semantic_chunks(
            sentences,
            [boundary],
            min_size=400,
            max_size=700,
            overlap=50,
        )
        second_run = build_semantic_chunks(
            sentences,
            [boundary],
            min_size=400,
            max_size=700,
            overlap=50,
        )

        first_ids = [chunk.chunk_id for chunk in first_run]
        second_ids = [chunk.chunk_id for chunk in second_run]

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertTrue(all(
            chunk.source == "20624.pdf"
            for chunk in first_run
        ))
        self.assertEqual(first_run[0].start_page, 1)
        self.assertEqual(first_run[0].end_page, 2)

    def test_overlap_uses_complete_sentences(self):
        sentences = [
            self.make_sentence(0, "A" * 120),
            self.make_sentence(1, "B" * 120),
            self.make_sentence(2, "C" * 120),
        ]
        overlap = _overlap_sentences(sentences[:2], overlap=150)

        self.assertEqual(len(overlap), 1)
        self.assertEqual(overlap[0].text, "B" * 120)

    def test_invalid_size_configuration_is_rejected(self):
        sentence = self.make_sentence(0, "Phrase")

        with self.assertRaisesRegex(ValueError, "min_size"):
            build_semantic_chunks([sentence], [], 700, 400, 100)


class MockPipelineEmbedder:
    def __init__(self):
        self.calls = []

    def embed_sentences(self, sentences):
        self.calls.append(sentences)
        embeddings = []

        for sentence in sentences:
            embeddings.append(
                SentenceEmbedding(
                    sentence=sentence,
                    embedding=np.asarray(
                        [1.0 if sentence.sentence_index % 2 == 0 else 0.0,
                         0.0 if sentence.sentence_index % 2 == 0 else 1.0],
                        dtype=np.float32,
                    ),
                )
            )

        return embeddings


class SemanticPipelineIntegrationTests(unittest.TestCase):
    def make_config(self, directory):
        config_path = directory / "semantic.yaml"
        config_path.write_text(
            "\n".join([
                "model_name: BAAI/bge-m3",
                "similarity_threshold: 0.72",
                "min_size: 10",
                "max_size: 100",
                "overlap: 10",
                "batch_size: 2",
                "device: cpu",
                "input_path: pages.jsonl",
                "output_path: semantic.jsonl",
            ]),
            encoding="utf-8",
        )
        return config_path

    def test_pipeline_writes_valid_jsonl_with_document_separation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config_path = self.make_config(directory)
            input_path = directory / "pages.jsonl"
            output_path = directory / "semantic.jsonl"

            input_path.write_text(
                "\n".join([
                    json.dumps({
                        "source": "a.pdf",
                        "page": 1,
                        "text": "Première phrase. Deuxième phrase.",
                    }, ensure_ascii=False),
                    json.dumps({
                        "source": "a.pdf",
                        "page": 2,
                        "text": "Troisième phrase!",
                    }, ensure_ascii=False),
                    json.dumps({
                        "source": "b.pdf",
                        "page": 1,
                        "text": "تم قبول الاستئناف؟ الحكم صالح.",
                    }, ensure_ascii=False),
                ]),
                encoding="utf-8",
            )

            embedder = MockPipelineEmbedder()
            stats = run_semantic_chunking(
                config_path=config_path,
                input_path=input_path,
                output_path=output_path,
                embedder=embedder,
            )

            lines = [
                json.loads(line)
                for line in output_path.read_text(
                    encoding="utf-8",
                ).splitlines()
                if line.strip()
            ]

            self.assertEqual(stats.documents, 2)
            self.assertGreater(stats.sentences, 0)
            self.assertEqual(stats.chunks, len(lines))
            self.assertEqual(stats.errors, 0)
            self.assertEqual(
                len(embedder.calls),
                2,
                "Each document should be embedded separately",
            )

            sources = {record["source"] for record in lines}
            self.assertEqual(sources, {"a.pdf", "b.pdf"})
            for record in lines:
                self.assertIn("chunk_id", record)
                self.assertIn("pages", record)
                self.assertIn("start_page", record)
                self.assertIn("end_page", record)
                self.assertGreaterEqual(
                    record["start_page"],
                    min(record["pages"]),
                )
                self.assertLessEqual(
                    record["end_page"],
                    max(record["pages"]),
                )

    def test_pipeline_rejects_interleaved_sources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config_path = self.make_config(directory)
            input_path = directory / "pages.jsonl"
            output_path = directory / "semantic.jsonl"

            input_path.write_text(
                "\n".join([
                    json.dumps({
                        "source": "a.pdf",
                        "page": 1,
                        "text": "Première phrase.",
                    }),
                    json.dumps({
                        "source": "b.pdf",
                        "page": 1,
                        "text": "Autre phrase.",
                    }),
                    json.dumps({
                        "source": "a.pdf",
                        "page": 2,
                        "text": "Dernière phrase.",
                    }),
                ]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "must be grouped contiguously",
            ):
                run_semantic_chunking(
                    config_path=config_path,
                    input_path=input_path,
                    output_path=output_path,
                    embedder=MockPipelineEmbedder(),
                )


if __name__ == "__main__":
    unittest.main()
