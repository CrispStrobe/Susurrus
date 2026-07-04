"""Test transcription history service."""

import os
import shutil
import tempfile
import unittest


class TestHistoryEntry(unittest.TestCase):
    def test_roundtrip(self):
        from utils.history_service import HistoryEntry

        entry = HistoryEntry(
            source_path="/tmp/audio.wav",
            backend="crispasr:parakeet",
            model="auto",
            language="en",
            segments=[(0.0, 2.5, "Hello"), (2.5, 5.0, "World")],
            duration=5.0,
        )
        d = entry.to_dict()
        restored = HistoryEntry.from_dict(d)
        self.assertEqual(restored.id, entry.id)
        self.assertEqual(restored.backend, "crispasr:parakeet")
        self.assertEqual(len(restored.segments), 2)
        self.assertAlmostEqual(restored.segments[0][0], 0.0)
        self.assertEqual(restored.segments[1][2], "World")

    def test_title_from_path(self):
        from utils.history_service import HistoryEntry

        entry = HistoryEntry(source_path="/home/user/audio/recording.wav")
        self.assertEqual(entry.title, "recording.wav")

    def test_title_fallback(self):
        from utils.history_service import HistoryEntry

        entry = HistoryEntry()
        self.assertTrue(entry.title.startswith("Transcription"))

    def test_created_at_str(self):
        from utils.history_service import HistoryEntry

        entry = HistoryEntry(created_at=1700000000.0)
        self.assertIn("2023", entry.created_at_str)

    def test_dict_segments(self):
        from utils.history_service import HistoryEntry

        entry = HistoryEntry(segments=[{"start": 1.0, "end": 2.0, "text": "dict seg"}])
        d = entry.to_dict()
        self.assertEqual(d["segments"][0]["text"], "dict seg")
        restored = HistoryEntry.from_dict(d)
        self.assertEqual(restored.segments[0][2], "dict seg")

    def test_speaker_names(self):
        from utils.history_service import HistoryEntry

        entry = HistoryEntry(speaker_names={"Speaker 1": "Alice"})
        d = entry.to_dict()
        restored = HistoryEntry.from_dict(d)
        self.assertEqual(restored.speaker_names["Speaker 1"], "Alice")


class TestHistoryService(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _svc(self):
        from utils.history_service import HistoryService

        return HistoryService(history_dir=self.tmpdir)

    def test_save_and_load(self):
        from utils.history_service import HistoryEntry

        svc = self._svc()
        entry = HistoryEntry(
            source_path="test.wav",
            segments=[(0.0, 1.0, "Hello")],
        )
        svc.save(entry)
        loaded = svc.load(entry.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.source_path, "test.wav")
        self.assertEqual(len(loaded.segments), 1)

    def test_list_entries(self):
        from utils.history_service import HistoryEntry

        svc = self._svc()
        for i in range(3):
            svc.save(HistoryEntry(source_path=f"file{i}.wav", created_at=1000 + i))
        entries = svc.list_entries()
        self.assertEqual(len(entries), 3)
        # Newest first
        self.assertEqual(entries[0].source_path, "file2.wav")

    def test_delete(self):
        from utils.history_service import HistoryEntry

        svc = self._svc()
        entry = HistoryEntry(source_path="delete_me.wav")
        svc.save(entry)
        self.assertIsNotNone(svc.load(entry.id))
        svc.delete(entry.id)
        self.assertIsNone(svc.load(entry.id))

    def test_search(self):
        from utils.history_service import HistoryEntry

        svc = self._svc()
        svc.save(HistoryEntry(source_path="meeting.wav", full_text="quarterly review"))
        svc.save(HistoryEntry(source_path="podcast.wav", full_text="tech discussion"))
        results = svc.search("quarterly")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_path, "meeting.wav")

    def test_search_by_title(self):
        from utils.history_service import HistoryEntry

        svc = self._svc()
        svc.save(HistoryEntry(source_path="interview_2024.wav"))
        results = svc.search("interview")
        self.assertEqual(len(results), 1)

    def test_search_case_insensitive(self):
        from utils.history_service import HistoryEntry

        svc = self._svc()
        svc.save(HistoryEntry(source_path="test.wav", full_text="Hello World"))
        self.assertEqual(len(svc.search("hello")), 1)
        self.assertEqual(len(svc.search("WORLD")), 1)

    def test_clear_all(self):
        from utils.history_service import HistoryEntry

        svc = self._svc()
        for i in range(5):
            svc.save(HistoryEntry())
        self.assertEqual(len(svc.list_entries()), 5)
        svc.clear_all()
        self.assertEqual(len(svc.list_entries()), 0)

    def test_load_nonexistent(self):
        svc = self._svc()
        self.assertIsNone(svc.load("nonexistent-id"))

    def test_corrupt_file_skipped(self):
        svc = self._svc()
        # Write a corrupt JSON file
        with open(os.path.join(self.tmpdir, "bad.json"), "w") as f:
            f.write("{invalid json")
        entries = svc.list_entries()
        self.assertEqual(len(entries), 0)  # Corrupt file skipped

    def test_empty_dir(self):
        svc = self._svc()
        self.assertEqual(svc.list_entries(), [])
        self.assertEqual(svc.search("anything"), [])


if __name__ == "__main__":
    unittest.main()
