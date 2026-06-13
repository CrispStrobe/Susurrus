"""Unit tests for the speaker-diarization backend.

Regression coverage for issue #12: the ``backends.diarization`` package was
excluded by ``.gitignore`` (a bare ``diarization/`` rule) so fresh clones were
missing the module and ``DiarizationManager`` entirely, and even when present
the import crashed on torchaudio >= 2.1 because pyannote.audio references
symbols (``AudioMetaData``, ``info``, ``list_audio_backends``) that newer
torchaudio removed.

These tests are offline: they never download a model or contact Hugging Face.
The live, network-dependent diarization run lives in the integration suite.
"""

import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIARIZATION_DIR = os.path.join(REPO_ROOT, "backends", "diarization")


def _diarization_available():
    """True only when the full diarization API (pyannote.audio etc.) is usable.

    The package itself imports even without the heavy optional deps (it degrades
    gracefully), so checking ``DiarizationManager is not None`` is what tells us
    the real API is present.
    """
    try:
        import backends.diarization as diarization

        return diarization.DiarizationManager is not None
    except Exception:
        return False


skip_no_diarization = unittest.skipUnless(
    _diarization_available(), "diarization API (pyannote.audio) not available"
)


class TestDiarizationPackagePresent(unittest.TestCase):
    """The source package must exist on disk and be tracked by git (issue #12)."""

    def test_package_files_exist(self):
        for fname in ("__init__.py", "manager.py", "compat.py", "progress.py"):
            path = os.path.join(DIARIZATION_DIR, fname)
            self.assertTrue(os.path.isfile(path), f"missing source file: {path}")

    def test_package_not_gitignored(self):
        """A bare ``diarization/`` ignore rule must not swallow the source package."""
        import subprocess

        result = subprocess.run(
            ["git", "check-ignore", "backends/diarization/manager.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        # check-ignore exits 0 (and prints the path) only when the file IS ignored.
        self.assertNotEqual(
            result.returncode,
            0,
            "backends/diarization/ is git-ignored — fresh clones will miss it (issue #12)",
        )


class TestTorchaudioCompatShim(unittest.TestCase):
    """The shim must restore torchaudio symbols pyannote imports at load time."""

    def setUp(self):
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            self.skipTest("torchaudio not installed")
        from backends.diarization.compat import apply_torchaudio_compat

        self.assertTrue(apply_torchaudio_compat())

    def test_symbols_restored(self):
        import torchaudio

        self.assertTrue(hasattr(torchaudio, "AudioMetaData"))
        self.assertTrue(hasattr(torchaudio, "info"))
        self.assertTrue(hasattr(torchaudio, "list_audio_backends"))

    def test_list_audio_backends_nonempty_when_soundfile_present(self):
        import torchaudio

        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")
        self.assertIn("soundfile", torchaudio.list_audio_backends())

    def test_info_reports_correct_metadata(self):
        import tempfile

        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            self.skipTest("numpy/soundfile not installed")
        import torchaudio

        sr = 16000
        samples = (0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, sr, endpoint=False))).astype(
            "float32"
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            sf.write(wav_path, samples, sr)
            meta = torchaudio.info(wav_path)
            self.assertEqual(meta.sample_rate, sr)
            self.assertEqual(meta.num_frames, sr)
            self.assertEqual(meta.num_channels, 1)
        finally:
            os.remove(wav_path)


@skip_no_diarization
class TestDiarizationImport(unittest.TestCase):
    """The public API must import — this is the exact failure from issue #12."""

    def test_import_public_api(self):
        from backends.diarization import (
            DiarizationManager,
            default_device,
            verify_authentication,
        )

        self.assertTrue(callable(verify_authentication))
        self.assertIsNotNone(DiarizationManager)
        self.assertIsNotNone(default_device)

    def test_worker_modules_import(self):
        import workers.diarize_audio  # noqa: F401
        import workers.diarize_worker  # noqa: F401


@skip_no_diarization
class TestDiarizationManager(unittest.TestCase):
    """Offline behaviour of DiarizationManager (no pipeline init, no network)."""

    def test_construct_cpu(self):
        from backends.diarization import DiarizationManager

        m = DiarizationManager(hf_token="dummy", device="cpu")
        self.assertEqual(m.device, "cpu")

    def test_list_available_models(self):
        from backends.diarization import DiarizationManager

        models = DiarizationManager.list_available_models()
        self.assertIn("Default", models)

    def test_unknown_model_falls_back_to_default(self):
        from backends.diarization import DiarizationManager

        m = DiarizationManager(hf_token="dummy", device="cpu", model_name="does-not-exist")
        self.assertEqual(m.model_name, "Default")

    def test_get_model_id(self):
        from backends.diarization import DiarizationManager

        m = DiarizationManager(hf_token="dummy", device="cpu")
        self.assertTrue(m.get_model_id().startswith("pyannote/"))

    def test_invalid_device_falls_back_to_cpu(self):
        # Requesting cuda on a machine without it must fall back, not crash.
        import torch

        from backends.diarization import DiarizationManager

        if not torch.cuda.is_available():
            m = DiarizationManager(hf_token="dummy", device="cuda")
            self.assertEqual(m.device, "cpu")


if __name__ == "__main__":
    unittest.main()
