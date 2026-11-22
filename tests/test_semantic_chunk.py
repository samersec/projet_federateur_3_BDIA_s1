import unittest

from extraction.semantic_chunk import PageRecord, split_into_sentences


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


if __name__ == "__main__":
    unittest.main()
