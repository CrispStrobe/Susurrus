"""End-to-end CLI tests — actually transcribe audio and verify output.

Requires: crispasr binary (functional) + test audio file.
Auto-skips when either is missing. CPU-only, ≤120s timeout.
"""

import json
import os
import subprocess
import sys
import unittest

# Reuse the binary check from test_crispasr_live
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _binary_works():
    from utils.crispasr_utils import find_crispasr

    exe = find_crispasr()
    if not exe:
        return None
    try:
        result = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return exe
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


_exe = _binary_works()
skip_no_binary = unittest.skipUnless(_exe, "crispasr binary not available or not runnable")

_test_audio = os.environ.get("SUSURRUS_TEST_AUDIO")
if not _test_audio:
    for candidate in [
        os.path.expanduser("~/code/CrispASR/samples/jfk.wav"),
        os.path.expanduser("~/code/CrispASR/tests/fixtures/jfk.wav"),
    ]:
        if os.path.isfile(candidate):
            _test_audio = candidate
            break
skip_no_audio = unittest.skipUnless(_test_audio, "No test audio file found")

_cli = os.path.join(os.path.dirname(__file__), "..", "..", "cli.py")


def _run_cli(args, timeout=120):
    """Run cli.py with args, return (returncode, stdout, stderr)."""
    cmd = [sys.executable, _cli] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


@skip_no_binary
class TestCLIListBackends(unittest.TestCase):
    """Test --list-backends works."""

    def test_list_backends(self):
        rc, out, err = _run_cli(["--list-backends"])
        self.assertEqual(rc, 0, f"--list-backends failed: {err}")
        self.assertIn("crispasr", out)
        self.assertIn("Transcription backends", out)
        self.assertIn("TTS backends", out)


@skip_no_binary
@skip_no_audio
class TestCLITranscribe(unittest.TestCase):
    """Test actual transcription via CLI."""

    def test_transcribe_default(self):
        """Transcribe with default backend (whisper tiny)."""
        rc, out, err = _run_cli(
            [
                "--backend",
                "crispasr",
                "--model",
                "auto:q5_0",
                "--file",
                _test_audio,
                "--auto-download",
            ]
        )
        self.assertEqual(rc, 0, f"transcribe failed: {err}")
        self.assertGreater(len(out.strip()), 0, "Empty output")

    def test_transcribe_output_srt(self):
        """Transcribe and output as SRT."""
        rc, out, err = _run_cli(
            [
                "--backend",
                "crispasr",
                "--model",
                "auto:q5_0",
                "--file",
                _test_audio,
                "--auto-download",
                "--output-format",
                "srt",
            ]
        )
        self.assertEqual(rc, 0, f"SRT output failed: {err}")
        # SRT has numbered entries and --> arrows
        self.assertIn("-->", out)

    def test_transcribe_output_json(self):
        """Transcribe and output as JSON."""
        rc, out, err = _run_cli(
            [
                "--backend",
                "crispasr",
                "--model",
                "auto:q5_0",
                "--file",
                _test_audio,
                "--auto-download",
                "--output-format",
                "json",
            ]
        )
        self.assertEqual(rc, 0, f"JSON output failed: {err}")
        data = json.loads(out)
        self.assertIn("segments", data)
        self.assertGreater(len(data["segments"]), 0)

    def test_transcribe_output_csv(self):
        """Transcribe and output as CSV."""
        rc, out, err = _run_cli(
            [
                "--backend",
                "crispasr",
                "--model",
                "auto:q5_0",
                "--file",
                _test_audio,
                "--auto-download",
                "--output-format",
                "csv",
            ]
        )
        self.assertEqual(rc, 0, f"CSV output failed: {err}")
        lines = out.strip().splitlines()
        self.assertGreater(len(lines), 1)  # header + at least one row
        self.assertIn("start", lines[0])

    def test_transcribe_output_vtt(self):
        """Transcribe and output as VTT."""
        rc, out, err = _run_cli(
            [
                "--backend",
                "crispasr",
                "--model",
                "auto:q5_0",
                "--file",
                _test_audio,
                "--auto-download",
                "--output-format",
                "vtt",
            ]
        )
        self.assertEqual(rc, 0, f"VTT output failed: {err}")
        self.assertTrue(out.startswith("WEBVTT"))


@skip_no_binary
class TestCLIMissingFile(unittest.TestCase):
    """Test error handling for missing files."""

    def test_missing_file_error(self):
        rc, out, err = _run_cli(["--file", "/nonexistent/audio.wav"])
        self.assertNotEqual(rc, 0)

    def test_no_file_error(self):
        rc, out, err = _run_cli([])
        self.assertNotEqual(rc, 0)


@skip_no_binary
class TestCLIModes(unittest.TestCase):
    """Test CLI modes exist and don't crash on --help."""

    def test_help(self):
        rc, out, err = _run_cli(["--help"])
        self.assertEqual(rc, 0)
        self.assertIn("transcribe", out)
        self.assertIn("tts", out)
        self.assertIn("translate", out)
        self.assertIn("align", out)


if __name__ == "__main__":
    unittest.main()
