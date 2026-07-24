"""Live integration tests for CrispASR binary.

All tests require the crispasr binary on PATH or at CRISPASR_EXECUTABLE.
They auto-skip when the binary is not found. Designed for a small VPS
(8 GB RAM, CPU-only, no GPU).

Tests use --dry-run-resolve (no model download/load) where possible,
and whisper-tiny (~75 MB) for the minimal real-inference tests.

Timeout: 60 s per test. No test should hang.
"""

import os
import subprocess
import tempfile
import unittest

from utils.crispasr_utils import find_crispasr


def _binary_works():
    """Check if the crispasr binary exists AND can actually execute."""
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

# Look for a short test audio file: env var, sibling CrispASR sample, or skip.
_test_audio = os.environ.get("SUSURRUS_TEST_AUDIO")
if not _test_audio:
    for candidate in [
        os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "jfk.wav"),
        os.path.expanduser("~/code/CrispASR/samples/jfk.wav"),
        os.path.expanduser("~/code/CrispASR/tests/fixtures/jfk.wav"),
    ]:
        if os.path.isfile(candidate):
            _test_audio = candidate
            break

skip_no_audio = unittest.skipUnless(_test_audio, "No test audio file found")


def _run(args, timeout=60):
    """Run crispasr with args, return (returncode, stdout, stderr)."""
    cmd = [_exe] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


@skip_no_binary
class TestBinaryBasics(unittest.TestCase):
    """Test that the binary starts and responds to basic flags."""

    def test_version(self):
        rc, out, _ = _run(["--version"])
        self.assertEqual(rc, 0)
        self.assertRegex(out.strip(), r"\d+\.\d+")

    def test_list_backends_json(self):
        rc, out, _ = _run(["--list-backends-json"])
        self.assertEqual(rc, 0)
        import json

        data = json.loads(out.strip())
        # Handle both formats: bare list or {"backends": [...]}
        if isinstance(data, dict) and "backends" in data:
            entries = data["backends"]
        else:
            entries = data
        self.assertIsInstance(entries, list)
        self.assertGreater(len(entries), 0)
        names = [e["name"] for e in entries if isinstance(e, dict)]
        # whisper should always be compiled in
        self.assertIn("whisper", names)


@skip_no_binary
class TestBackendProbe(unittest.TestCase):
    """Test that new CrispASR backends appear in --list-backends-json."""

    def setUp(self):
        rc, out, _ = _run(["--list-backends-json"])
        import json

        self.backends = []
        if rc == 0:
            data = json.loads(out.strip())
            # Handle both formats: bare list or {"backends": [...]}
            if isinstance(data, dict) and "backends" in data:
                entries = data["backends"]
            else:
                entries = data
            self.backends = [e["name"] for e in entries if isinstance(e, dict)]

    def test_whisper_present(self):
        self.assertIn("whisper", self.backends)

    def test_parakeet_present(self):
        # parakeet is compiled in most builds
        if "parakeet" not in self.backends:
            self.skipTest("parakeet not compiled in this build")

    def test_new_087_backends_if_compiled(self):
        """Check new backends — advisory only (they may not be compiled in)."""
        new_backends = [
            "ark-asr",
            "higgs-stt",
            "moss-transcribe",
            "dots-tts",
            "bananamind-tts",
            "tada",
        ]
        found = [b for b in new_backends if b in self.backends]
        # At least log what we found
        if not found:
            self.skipTest(f"None of the new 0.8.7 backends compiled in: {new_backends}")

    def test_new_0822_backends_if_compiled(self):
        """Check CrispASR 0.8.22 backends — advisory only."""
        new_backends = [
            "granite-4.1-plus",
            "granite-4.1-nar",
            "omniasr-300m",
            "omniasr-llm",
            "fun-asr-mlt-nano",
            "miotts",
            "qwen3-tts-customvoice",
            "chatterbox-turbo",
            "kartoffelbox-turbo",
            "lahgtna-chatterbox",
        ]
        found = [b for b in new_backends if b in self.backends]
        if not found:
            self.skipTest(f"None of the new 0.8.22 backends compiled in: {new_backends}")


@skip_no_binary
class TestDryRunResolve(unittest.TestCase):
    """Test --dry-run-resolve for model resolution without loading."""

    def test_whisper_dry_run(self):
        rc, out, err = _run(["--backend", "whisper", "-m", "auto", "--dry-run-resolve", "--no-gpu"])
        # Should succeed and print a model path
        self.assertEqual(rc, 0, f"dry-run failed: {err}")

    def test_parakeet_dry_run(self):
        rc, out, err = _run(
            ["--backend", "parakeet", "-m", "auto", "--dry-run-resolve", "--no-gpu"]
        )
        self.assertEqual(rc, 0, f"dry-run failed: {err}")


@skip_no_binary
@skip_no_audio
class TestTranscribeSmall(unittest.TestCase):
    """Transcribe with whisper-tiny (75 MB, fits 8 GB RAM, CPU-only)."""

    def test_whisper_tiny_transcribe(self):
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto:q5_0",
                "-f",
                _test_audio,
                "--no-gpu",
                "-t",
                "2",
                "--auto-download",
                "-np",
            ],
            timeout=120,
        )
        self.assertEqual(rc, 0, f"transcribe failed: {err}")
        # Should produce some output
        self.assertGreater(len(out.strip()), 0, "Empty transcription output")


@skip_no_binary
@skip_no_audio
class TestAlignOnly(unittest.TestCase):
    """Test --align-only mode (requires aligner model)."""

    def test_align_only_flag_accepted(self):
        """Verify --align-only is a recognized flag (dry-run)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("And so my fellow Americans")
            f.flush()
            txt_path = f.name

        try:
            rc, out, err = _run(
                [
                    "--backend",
                    "whisper",
                    "-m",
                    "auto:q5_0",
                    "-f",
                    _test_audio,
                    "--align-only",
                    "--text-file",
                    txt_path,
                    "--no-gpu",
                    "-t",
                    "2",
                    "--auto-download",
                    "-np",
                ],
                timeout=120,
            )
            # The flag should be accepted (rc=0); output may vary by binary version
            if rc != 0 and "align-only" in err.lower():
                self.skipTest("Binary does not support --align-only yet")
            self.assertEqual(rc, 0, f"align-only failed: {err}")
        finally:
            os.unlink(txt_path)


@skip_no_binary
class TestDiarizeFlags(unittest.TestCase):
    """Test new diarization flags are accepted."""

    def test_diarize_speakers_flag_accepted(self):
        """--diarize-speakers should be a recognized flag."""
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto",
                "--dry-run-resolve",
                "--diarize-speakers",
                "--no-gpu",
            ]
        )
        if rc != 0 and "diarize-speakers" in err:
            self.skipTest("Binary does not support --diarize-speakers yet")
        self.assertEqual(rc, 0, f"flag rejected: {err}")


@skip_no_binary
class TestNewCLIFlags(unittest.TestCase):
    """Test that new CLI flags are accepted by the binary (dry-run)."""

    def test_prefix_text_accepted(self):
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto",
                "--dry-run-resolve",
                "--prefix-text",
                "meeting notes",
                "--no-gpu",
            ]
        )
        if rc != 0 and "prefix-text" in err:
            self.skipTest("Binary does not support --prefix-text yet")
        self.assertEqual(rc, 0, f"flag rejected: {err}")

    def test_speaker_db_consent_accepted(self):
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto",
                "--dry-run-resolve",
                "--speaker-db-consent",
                "--no-gpu",
            ]
        )
        if rc != 0 and "speaker-db-consent" in err:
            self.skipTest("Binary does not support --speaker-db-consent yet")
        self.assertEqual(rc, 0, f"flag rejected: {err}")


@skip_no_binary
class TestProvenanceFlags(unittest.TestCase):
    """Test EU AI Act compliance flags are accepted by the binary."""

    def test_no_watermark_accepted(self):
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto",
                "--dry-run-resolve",
                "--no-watermark",
                "--no-gpu",
            ]
        )
        if rc != 0 and "no-watermark" in err:
            self.skipTest("Binary does not support --no-watermark yet")
        self.assertEqual(rc, 0, f"flag rejected: {err}")

    def test_i_have_rights_accepted(self):
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto",
                "--dry-run-resolve",
                "--i-have-rights",
                "--no-gpu",
            ]
        )
        if rc != 0 and "i-have-rights" in err:
            self.skipTest("Binary does not support --i-have-rights yet")
        self.assertEqual(rc, 0, f"flag rejected: {err}")

    def test_no_spoken_disclaimer_accepted(self):
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto",
                "--dry-run-resolve",
                "--no-spoken-disclaimer",
                "--no-gpu",
            ]
        )
        if rc != 0 and "no-spoken-disclaimer" in err:
            self.skipTest("Binary does not support --no-spoken-disclaimer yet")
        self.assertEqual(rc, 0, f"flag rejected: {err}")

    def test_no_c2pa_requires_responsibility_ack(self):
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto",
                "--dry-run-resolve",
                "--no-c2pa",
                "--accept-marking-responsibility",
                "--no-gpu",
            ]
        )
        if rc != 0 and "no-c2pa" in err:
            self.skipTest("Binary does not support --no-c2pa yet")
        self.assertEqual(rc, 0, f"flag rejected: {err}")

    def test_tts_speed_accepted(self):
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto",
                "--dry-run-resolve",
                "--tts-speed",
                "1.5",
                "--no-gpu",
            ]
        )
        if rc != 0 and "tts-speed" in err:
            self.skipTest("Binary does not support --tts-speed yet")
        self.assertEqual(rc, 0, f"flag rejected: {err}")

    def test_att_context_accepted(self):
        rc, out, err = _run(
            [
                "--backend",
                "whisper",
                "-m",
                "auto",
                "--dry-run-resolve",
                "--att-context",
                "512",
                "--no-gpu",
            ]
        )
        if rc != 0 and "att-context" in err:
            self.skipTest("Binary does not support --att-context yet")
        self.assertEqual(rc, 0, f"flag rejected: {err}")

    def test_vad_export_import_flags_accepted(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"segments":[]}')
            f.flush()
            vad_path = f.name

        try:
            rc, out, err = _run(
                [
                    "--backend",
                    "whisper",
                    "-m",
                    "auto",
                    "--dry-run-resolve",
                    "--vad-import",
                    vad_path,
                    "--vad-import-strict",
                    "--vad-export",
                    vad_path,
                    "--no-gpu",
                ]
            )
            if rc != 0 and "vad-import" in err:
                self.skipTest("Binary does not support VAD import/export yet")
            self.assertEqual(rc, 0, f"flag rejected: {err}")
        finally:
            os.unlink(vad_path)


if __name__ == "__main__":
    unittest.main()
