"""Test the dependency-free AI-generation marker (EU AI Act Art. 50(2))."""

import os
import struct
import tempfile
import unittest
import wave

from utils.ai_marking import (
    AI_DISCLOSURE_TEXT,
    embed_wav_ai_marker,
    is_ai_marked,
    read_wav_ai_marker,
)

_FRAMES = b"\x01\x00" * 1000


def _write_wav(path, frames=_FRAMES):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(frames)


class TestAiMarking(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "out.wav")
        _write_wav(self.path)

    def tearDown(self):
        self._dir.cleanup()

    def test_unmarked_by_default(self):
        self.assertFalse(is_ai_marked(self.path))
        self.assertIsNone(read_wav_ai_marker(self.path))

    def test_embed_then_read_round_trip(self):
        self.assertTrue(embed_wav_ai_marker(self.path, model="piper"))
        self.assertTrue(is_ai_marked(self.path))

        tags = read_wav_ai_marker(self.path)
        self.assertEqual(tags["ICMT"], AI_DISCLOSURE_TEXT)
        self.assertEqual(tags["IENG"], "piper")
        self.assertEqual(tags["ISFT"], "Susurrus")
        self.assertEqual(tags["ITCH"], "EU-AI-Act-Art50-2")

    def test_audio_payload_unchanged(self):
        """Marking must not alter a single audio sample."""
        embed_wav_ai_marker(self.path)
        with wave.open(self.path, "rb") as w:
            self.assertEqual(w.readframes(w.getnframes()), _FRAMES)
            self.assertEqual(w.getframerate(), 16000)
            self.assertEqual(w.getnchannels(), 1)

    def test_riff_size_header_updated(self):
        """A stale RIFF size field would make the file unreadable to parsers."""
        embed_wav_ai_marker(self.path)
        with open(self.path, "rb") as f:
            data = f.read()
        (declared,) = struct.unpack("<I", data[4:8])
        self.assertEqual(declared, len(data) - 8)

    def test_idempotent(self):
        """Re-marking must not stack duplicate chunks."""
        self.assertTrue(embed_wav_ai_marker(self.path))
        size_once = os.path.getsize(self.path)
        self.assertTrue(embed_wav_ai_marker(self.path))
        self.assertEqual(os.path.getsize(self.path), size_once)

    def test_odd_length_payload_stays_aligned(self):
        """RIFF chunks are word-aligned; an odd payload needs a pad byte."""
        odd = os.path.join(self._dir.name, "odd.wav")
        _write_wav(odd, frames=b"\x01\x00" * 1000 + b"\x02\x00")
        self.assertTrue(embed_wav_ai_marker(odd))
        self.assertIsNotNone(read_wav_ai_marker(odd))
        with wave.open(odd, "rb") as w:
            self.assertEqual(w.getnframes(), 1001)

    def test_missing_file(self):
        self.assertFalse(embed_wav_ai_marker(os.path.join(self._dir.name, "nope.wav")))
        self.assertIsNone(read_wav_ai_marker(os.path.join(self._dir.name, "nope.wav")))

    def test_non_riff_file(self):
        bogus = os.path.join(self._dir.name, "bogus.wav")
        with open(bogus, "wb") as f:
            f.write(b"not a riff file at all")
        self.assertFalse(embed_wav_ai_marker(bogus))
        self.assertIsNone(read_wav_ai_marker(bogus))

    def test_no_temp_file_left_behind(self):
        embed_wav_ai_marker(self.path)
        leftovers = [n for n in os.listdir(self._dir.name) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
