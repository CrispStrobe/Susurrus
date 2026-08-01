"""Audio watermarking for synthetic speech (EU AI Act Art. 50(2)).

Two in-sample watermarks, tried strongest first:

  1. AudioSeal (Meta, MIT) — learned, robust to deliberate removal. Optional:
     it pulls in torch plus model weights.
  2. A spread-spectrum comb in :mod:`utils.spread_spectrum` — pure numpy, so
     it is always available, and byte-compatible with CrispASR and CrispTTS.

Tier 2 exists because tier 1 being absent used to mean no in-sample mark at
all, leaving only the declarative RIFF chunk — which any transcode removes.
Art. 50(2) has no "unless an optional dependency is missing" clause.

The declarative RIFF marker in ``utils.ai_marking`` makes synthetic audio
machine-readable, but it is metadata: strip the chunk, or transcode to MP3,
and it is gone. A neural watermark is embedded in the samples themselves and
survives re-encoding, resampling and clipping.

    pip install audioseal          # or: pip install 'susurrus[watermark]'

The CrispASR backends watermark inside the binary and do not use this module.
"""

import logging

logger = logging.getLogger(__name__)

#: AudioSeal model cards. The 16-bit variant carries a short payload we do
#: not use; presence of the watermark is the signal that matters here.
_GENERATOR_MODEL = "audioseal_wm_16bits"
_DETECTOR_MODEL = "audioseal_detector_16bits"

#: Detection probability at or above which audio counts as watermarked.
#: AudioSeal's own guidance treats 0.5 as the decision boundary.
DETECTION_THRESHOLD = 0.5

_generator = None
_detector = None
_unavailable = False


def _load(kind):
    """Lazily load an AudioSeal model. Returns None when unavailable."""
    global _generator, _detector, _unavailable

    if _unavailable:
        return None

    cached = _generator if kind == "generator" else _detector
    if cached is not None:
        return cached

    try:
        from audioseal import AudioSeal
    except ImportError:
        logger.debug("audioseal not installed; neural watermarking unavailable")
        _unavailable = True
        return None

    try:
        if kind == "generator":
            _generator = AudioSeal.load_generator(_GENERATOR_MODEL)
            return _generator
        _detector = AudioSeal.load_detector(_DETECTOR_MODEL)
        return _detector
    except Exception as e:
        # Model download failures, torch version mismatches, no network —
        # all mean "no neural watermark", none mean "crash the synthesis".
        logger.warning("Could not load AudioSeal %s: %s", kind, e)
        _unavailable = True
        return None


def neural_available():
    """Return True if the AudioSeal (neural) watermark can run.

    Loads the model, so this is the expensive question. Ask it only when the
    distinction between the two tiers actually matters.
    """
    return _load("generator") is not None


def is_available():
    """Return True if *any* in-sample watermark can be applied or detected.

    Checks the dependency-free tier first: it is a plain numpy/soundfile
    import, whereas the neural check loads torch and may download weights.
    Callers use this to decide whether watermarking is worth attempting at
    all, and since the spread-spectrum tier needs neither, the answer is
    usually yes without touching torch.
    """
    try:
        import numpy  # noqa: F401
        import soundfile  # noqa: F401

        return True
    except ImportError:
        pass
    return neural_available()


def _read_wav(path):
    """Read a WAV as a (1, 1, samples) float tensor plus its sample rate."""
    import torch
    import torchaudio

    wav, sample_rate = torchaudio.load(path)
    if wav.dim() == 2:
        # AudioSeal expects a batch dimension: (batch, channels, samples).
        wav = wav.unsqueeze(0)
    return wav.to(torch.float32), sample_rate


def embed_watermark(wav_path):
    """Embed an in-sample watermark into a WAV file in place.

    Uses AudioSeal when available and falls back to the spread-spectrum comb,
    which needs only numpy. Returns True if either applied.
    """
    generator = _load("generator")
    if generator is None:
        return _embed_spread_spectrum(wav_path)

    try:
        import torchaudio

        wav, sample_rate = _read_wav(wav_path)
        watermarked = generator.get_watermarked_audio(wav, sample_rate=sample_rate)
        # Drop the batch dimension torchaudio.save does not expect.
        torchaudio.save(wav_path, watermarked.squeeze(0).detach().cpu(), sample_rate)
    except Exception as e:
        logger.warning(
            "Neural watermarking failed for %s: %s — "
            "falling back to the spread-spectrum watermark.",
            wav_path,
            e,
        )
        return _embed_spread_spectrum(wav_path)

    logger.info("AudioSeal watermark embedded: %s", wav_path)
    return True


def _embed_spread_spectrum(wav_path):
    """Embed the dependency-free watermark. Returns True on success.

    Every channel is marked, and the file's channel count and sample format
    are preserved. Watermarking channel 0 and writing the result back as mono
    would silently discard the other channels — a marking step must not
    destroy part of the audio it is marking.
    """
    try:
        import numpy as np
        import soundfile as sf

        from utils import spread_spectrum

        info = sf.info(wav_path)
        data, rate = sf.read(wav_path, dtype="float32", always_2d=True)

        marked = np.empty_like(data)
        for channel in range(data.shape[1]):
            marked[:, channel] = spread_spectrum.embed(data[:, channel])

        if info.channels == 1:
            marked = marked[:, 0]

        # Keep the original subtype: forcing PCM_16 would quantise 24-bit or
        # float output on its way through the watermarker.
        subtype = info.subtype if info.subtype else "PCM_16"
        sf.write(wav_path, marked, rate, subtype=subtype)
    except Exception as e:
        logger.warning("Spread-spectrum watermarking failed for %s: %s", wav_path, e)
        return False
    logger.info("Spread-spectrum watermark embedded: %s", wav_path)
    return True


def _detect_spread_spectrum(wav_path):
    """Detect the dependency-free watermark. Returns a result dict or None."""
    try:
        import soundfile as sf

        from utils import spread_spectrum

        data, _rate = sf.read(wav_path, dtype="float32")
        mono = data[:, 0] if data.ndim > 1 else data
        confidence = spread_spectrum.detect(mono)
    except Exception as e:
        logger.warning("Spread-spectrum detection failed for %s: %s", wav_path, e)
        return None
    return {
        "watermarked": confidence >= spread_spectrum.DETECTION_THRESHOLD,
        "confidence": confidence,
        "threshold": spread_spectrum.DETECTION_THRESHOLD,
        "backend": "spread-spectrum",
    }


def detect_watermark(wav_path):
    """Detect an AudioSeal watermark in a WAV file.

    Returns:
        dict with ``watermarked`` (bool) and ``confidence`` (float), or None
        if AudioSeal is unavailable or the file could not be processed.
    """
    detector = _load("detector")
    if detector is None:
        return _detect_spread_spectrum(wav_path)

    try:
        wav, sample_rate = _read_wav(wav_path)
        result, _message = detector.detect_watermark(wav, sample_rate=sample_rate)
        confidence = float(result)
    except Exception as e:
        logger.warning("Watermark detection failed for %s: %s", wav_path, e)
        return _detect_spread_spectrum(wav_path)

    if confidence >= DETECTION_THRESHOLD:
        return {
            "watermarked": True,
            "confidence": confidence,
            "threshold": DETECTION_THRESHOLD,
            "backend": "audioseal",
        }
    # AudioSeal says no — the file may still carry a spread-spectrum mark from
    # a build without torch, or from CrispASR/CrispTTS.
    fallback = _detect_spread_spectrum(wav_path)
    if fallback and fallback["watermarked"]:
        return fallback
    return {
        "watermarked": False,
        "confidence": confidence,
        "threshold": DETECTION_THRESHOLD,
        "backend": "audioseal",
    }


def _reset_cache_for_tests():
    """Clear the module-level model cache (test helper)."""
    global _generator, _detector, _unavailable
    _generator = None
    _detector = None
    _unavailable = False
