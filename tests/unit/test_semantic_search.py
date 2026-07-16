"""Test semantic search utility."""

import unittest


class TestSemanticSearch(unittest.TestCase):
    def test_import(self):
        from utils.semantic_search import is_available

        self.assertIsInstance(is_available(), bool)

    def test_empty_query(self):
        from utils.semantic_search import semantic_search

        self.assertEqual(semantic_search("", []), [])

    def test_empty_entries(self):
        from utils.semantic_search import semantic_search

        self.assertEqual(semantic_search("hello", []), [])

    def test_substring_fallback(self):
        from utils.history_service import HistoryEntry
        from utils.semantic_search import semantic_search

        entries = [
            HistoryEntry(source_path="meeting.wav", full_text="quarterly review"),
            HistoryEntry(source_path="podcast.wav", full_text="tech discussion"),
        ]
        results = semantic_search("quarterly", entries)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].source_path, "meeting.wav")

    def test_title_match_higher_score(self):
        from utils.history_service import HistoryEntry
        from utils.semantic_search import semantic_search

        entries = [
            HistoryEntry(source_path="review.wav", full_text="something else"),
            HistoryEntry(source_path="other.wav", full_text="review of the quarter"),
        ]
        results = semantic_search("review", entries)
        self.assertEqual(len(results), 2)
        # Title match should score higher
        self.assertGreater(results[0][1], results[1][1])


if __name__ == "__main__":
    unittest.main()
