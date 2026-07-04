"""Test segment model and transcription result."""

import unittest


class TestSegment(unittest.TestCase):
    def test_basic(self):
        from utils.segment_model import Segment

        s = Segment(1.0, 2.5, "Hello world")
        self.assertAlmostEqual(s.start, 1.0)
        self.assertAlmostEqual(s.end, 2.5)
        self.assertEqual(s.text, "Hello world")
        self.assertFalse(s.edited)

    def test_roundtrip_dict(self):
        from utils.segment_model import Segment

        s = Segment(1.0, 2.0, "Test", speaker="Speaker 1", confidence=0.95, edited=True)
        d = s.to_dict()
        s2 = Segment.from_dict(d)
        self.assertEqual(s2.text, "Test")
        self.assertEqual(s2.speaker, "Speaker 1")
        self.assertAlmostEqual(s2.confidence, 0.95)
        self.assertTrue(s2.edited)

    def test_from_tuple(self):
        from utils.segment_model import Segment

        s = Segment.from_tuple((3.0, 5.0, "From tuple"))
        self.assertAlmostEqual(s.start, 3.0)
        self.assertEqual(s.text, "From tuple")

    def test_to_tuple(self):
        from utils.segment_model import Segment

        s = Segment(1.0, 2.0, "Tuple out")
        self.assertEqual(s.to_tuple(), (1.0, 2.0, "Tuple out"))

    def test_repr(self):
        from utils.segment_model import Segment

        s = Segment(0.0, 1.0, "Hi", speaker="Alice")
        r = repr(s)
        self.assertIn("Alice", r)
        self.assertIn("Hi", r)

    def test_dict_without_optional(self):
        from utils.segment_model import Segment

        s = Segment(0.0, 1.0, "Plain")
        d = s.to_dict()
        self.assertNotIn("speaker", d)
        self.assertNotIn("confidence", d)
        self.assertNotIn("edited", d)


class TestTranscriptionResult(unittest.TestCase):
    def test_add_segment(self):
        from utils.segment_model import TranscriptionResult

        tr = TranscriptionResult()
        tr.add_segment(0.0, 1.0, "Hello")
        tr.add_segment(1.0, 2.0, "World")
        self.assertEqual(len(tr.segments), 2)

    def test_edit_segment(self):
        from utils.segment_model import TranscriptionResult

        tr = TranscriptionResult()
        tr.add_segment(0.0, 1.0, "Original")
        tr.edit_segment(0, "Edited")
        self.assertEqual(tr.segments[0].text, "Edited")
        self.assertTrue(tr.segments[0].edited)

    def test_edit_out_of_range(self):
        from utils.segment_model import TranscriptionResult

        tr = TranscriptionResult()
        tr.add_segment(0.0, 1.0, "Only one")
        tr.edit_segment(5, "Nothing")  # Should not crash
        self.assertEqual(tr.segments[0].text, "Only one")

    def test_rename_speaker(self):
        from utils.segment_model import TranscriptionResult

        tr = TranscriptionResult()
        tr.add_segment(0.0, 1.0, "Hello", speaker="Speaker 1")
        tr.rename_speaker("Speaker 1", "Alice")
        self.assertEqual(tr.display_speaker("Speaker 1"), "Alice")
        self.assertIsNone(tr.display_speaker(None))

    def test_display_speaker_unmapped(self):
        from utils.segment_model import TranscriptionResult

        tr = TranscriptionResult()
        self.assertEqual(tr.display_speaker("Speaker 2"), "Speaker 2")

    def test_full_text(self):
        from utils.segment_model import TranscriptionResult

        tr = TranscriptionResult()
        tr.add_segment(0.0, 1.0, "First")
        tr.add_segment(1.0, 2.0, "Second")
        self.assertEqual(tr.full_text(), "First\nSecond")

    def test_to_tuples(self):
        from utils.segment_model import TranscriptionResult

        tr = TranscriptionResult()
        tr.add_segment(0.0, 1.0, "A")
        tr.add_segment(1.0, 2.0, "B")
        tuples = tr.to_tuples()
        self.assertEqual(len(tuples), 2)
        self.assertEqual(tuples[0], (0.0, 1.0, "A"))

    def test_from_tuples(self):
        from utils.segment_model import TranscriptionResult

        tr = TranscriptionResult.from_tuples([(0.0, 1.0, "X"), (1.0, 2.0, "Y")])
        self.assertEqual(len(tr.segments), 2)
        self.assertEqual(tr.segments[1].text, "Y")


if __name__ == "__main__":
    unittest.main()
