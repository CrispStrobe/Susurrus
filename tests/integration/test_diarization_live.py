"""Live integration test for speaker diarization.

This test actually loads the pyannote pipeline and runs diarization, so it
downloads the model from Hugging Face and requires an accepted model licence.
It is skipped by default; opt in with::

    RUN_LIVE_TESTS=1 HF_TOKEN=hf_xxx pytest tests/integration/test_diarization_live.py

The audio is a synthetic two-speaker clip (two distinct tones with a gap), so
the run is self-contained and needs no external fixtures.
"""

import os
import tempfile
import unittest


def _diarization_available():
    try:
        import backends.diarization as diarization

        return diarization.DiarizationManager is not None
    except Exception:
        return False


def _live_enabled():
    return os.environ.get("RUN_LIVE_TESTS") == "1" and bool(os.environ.get("HF_TOKEN"))


skip_no_live = unittest.skipUnless(
    _live_enabled() and _diarization_available(),
    "live diarization test — set RUN_LIVE_TESTS=1 and HF_TOKEN to run",
)


@skip_no_live
class TestDiarizationLive(unittest.TestCase):
    def _make_two_speaker_wav(self, path):
        import numpy as np
        import soundfile as sf

        sr = 16000

        def tone(freq, seconds):
            t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
            return 0.3 * np.sin(2 * np.pi * freq * t)

        silence = np.zeros(int(sr * 0.5))
        # "speaker A" then a gap then "speaker B"
        signal = np.concatenate(
            [tone(180, 2.0), silence, tone(330, 2.0), silence, tone(180, 2.0)]
        ).astype("float32")
        sf.write(path, signal, sr)

    def _real_speech_clip(self):
        """Return a path to a real multi-speaker speech clip if one is available.

        Set SUSURRUS_TEST_AUDIO to point at your own clip; otherwise we probe a
        couple of well-known sample locations (e.g. a sibling CrispASR checkout).
        Synthetic tones don't trigger pyannote's voice-activity detection, so we
        can only assert on speaker counts when real speech is available.
        """
        env = os.environ.get("SUSURRUS_TEST_AUDIO")
        if env and os.path.isfile(env):
            return env
        for cand in (
            os.path.expanduser("~/code/CrispASR/samples/multispeaker.wav"),
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "CrispASR",
                "samples",
                "multispeaker.wav",
            ),
        ):
            if os.path.isfile(cand):
                return os.path.abspath(cand)
        return None

    def test_diarize_returns_segments(self):
        from backends.diarization import DiarizationManager

        manager = DiarizationManager(hf_token=os.environ["HF_TOKEN"], device="cpu")

        real = self._real_speech_clip()
        if real:
            # Real speech → the pipeline must actually find speakers/segments.
            segments = manager.diarize(real)
            self.assertIsInstance(segments, list)
            self.assertGreater(len(segments), 0, "diarization produced no segments")
            for seg in segments:
                self.assertIn("speaker", seg)
                self.assertIn("start", seg)
                self.assertIn("end", seg)
                self.assertLessEqual(seg["start"], seg["end"])
            return

        # No real clip available — verify the pipeline loads and runs end to end
        # on synthetic audio (it will legitimately find 0 speech segments).
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            self._make_two_speaker_wav(wav_path)
            segments = manager.diarize(wav_path)
            self.assertIsInstance(segments, list)
        finally:
            os.remove(wav_path)


if __name__ == "__main__":
    unittest.main()
