"""EU AI Act Art. 50 provenance, applied to a synthesized audio file.

This is the single implementation of the marking pipeline. It lives outside
``TTSBackend`` because not every route that produces synthetic audio is a
``TTSBackend``: ``CrispasrFFIBackend`` is a *transcription* backend that also
exposes ``synthesize()`` and ``speech_to_speech()``, and before this module
existed those routes wrote audio with no marking at all and then crashed on a
missing ``apply_provenance``.

Layer order is load-bearing:

1. **Spoken disclosure** (Art. 50(4)) — prepends audible speech, so it must
   run before anything that depends on the sample data.
2. **Neural watermark** (AudioSeal) — mutates samples; survives re-encoding.
3. **Declarative marker** — container-native metadata (RIFF INFO / ID3),
   dependency-free, so a default install still emits marked audio.
4. **C2PA Content Credentials** — hashes the finished file, so it must be
   last or the manifest describes audio that no longer exists.
"""

import logging
import os

logger = logging.getLogger(__name__)

#: Keys present in every result dict returned by :func:`apply_provenance`.
_EMPTY = {
    "spoken": False,
    "watermark": False,
    "marker": False,
    "c2pa": False,
    "opted_out": False,
    "unsupported_format": False,
}

#: Containers the declarative marker understands. Anything else can still be
#: watermarked or C2PA-signed, but has no dependency-free fallback.
_MARKABLE = (".wav", ".mp3")


def new_result(**overrides):
    """Return a fresh provenance-result dict."""
    result = dict(_EMPTY)
    result.update(overrides)
    return result


def marking_applied(result):
    """True if at least one machine-readable layer landed on the file."""
    return bool(result.get("watermark") or result.get("marker") or result.get("c2pa"))


def apply_provenance(
    output_path,
    options=None,
    model=None,
    backend=None,
    is_cloning=False,
    locale=None,
):
    """Apply the Art. 50 layers to *output_path* and report what landed.

    Args:
        output_path: The synthesized audio file, modified in place.
        options: Mapping of provenance switches — ``no_watermark``,
            ``no_c2pa``, ``no_spoken_disclaimer``,
            ``accept_marking_responsibility``, ``c2pa_cert``, ``c2pa_key``.
        model: Model identifier recorded in the marker and manifest.
        backend: The backend that produced the audio, used to synthesize the
            spoken disclosure in a matching format. Optional.
        is_cloning: Whether this synthesis cloned a voice, which is what
            engages the Art. 50(4) audible disclosure.
        locale: Language for the spoken disclosure.

    Returns:
        dict with ``spoken``, ``watermark``, ``marker``, ``c2pa``,
        ``opted_out`` and ``unsupported_format``.
    """
    options = options or {}
    result = new_result()

    if not output_path or not os.path.isfile(output_path):
        return result

    if options.get("accept_marking_responsibility"):
        logger.warning(
            "AI-content marking skipped (--accept-marking-responsibility). "
            "Responsibility for marking this output rests with the operator "
            "per EU AI Act Art. 50."
        )
        result["opted_out"] = True
        return result

    ext = os.path.splitext(output_path)[1].lower()
    result["unsupported_format"] = ext not in _MARKABLE

    # 1. Spoken disclosure — cloning only, matching CrispASR's behaviour.
    if is_cloning and backend is not None and not options.get("no_spoken_disclaimer"):
        try:
            from utils.spoken_disclosure import prepend_spoken_disclosure

            result["spoken"] = prepend_spoken_disclosure(backend, output_path, locale=locale)
        except ImportError:
            pass

    # 2. Neural watermark — robust to re-encoding, optional dependency.
    if not options.get("no_watermark"):
        try:
            from utils.audio_watermark import embed_watermark

            result["watermark"] = embed_watermark(output_path)
        except ImportError:
            pass

    # 3. Declarative marker — the layer that needs no dependencies.
    try:
        from utils.ai_marking import embed_ai_marker

        result["marker"] = embed_ai_marker(output_path, model=model)
    except ImportError:
        pass

    # 4. C2PA last: it hashes the final bytes.
    if not options.get("no_c2pa"):
        try:
            from utils.c2pa_signing import sign_audio_file

            result["c2pa"] = sign_audio_file(
                output_path,
                cert_pem=options.get("c2pa_cert"),
                key_pem=options.get("c2pa_key"),
                model=model,
            )
        except ImportError:
            pass

    if not marking_applied(result):
        logger.warning(
            "Could not mark %s as AI-generated. EU AI Act Art. 50(2) requires "
            "machine-readable marking of synthetic audio.",
            output_path,
        )

    return result


def verify_marking(output_path):
    """Inspect a file and report which Art. 50 layers are actually present.

    This asks the *file*, not the flags that were passed to whatever produced
    it. Used where marking is performed by something Susurrus does not
    control — notably the CrispASR binary, whose provenance support depends on
    build options, model capability and version.

    Returns:
        dict with ``marker``, ``c2pa`` and ``watermark``. A value of None
        means "cannot tell" (the detector for that layer is unavailable),
        which is deliberately distinct from False.
    """
    found = {"marker": False, "c2pa": None, "watermark": None}

    if not output_path or not os.path.isfile(output_path):
        return found

    try:
        from utils.ai_marking import read_ai_marker

        found["marker"] = read_ai_marker(output_path) is not None
    except ImportError:
        found["marker"] = None

    try:
        from utils.c2pa_signing import is_available as c2pa_available
        from utils.c2pa_signing import verify_audio_file

        if c2pa_available():
            report = verify_audio_file(output_path)
            found["c2pa"] = bool(report and report.get("valid"))
    except ImportError:
        pass

    # Neural detection last: it is the only probe that loads a model and runs
    # inference, so it must not gate on a name shadowed by the import above.
    try:
        from utils.audio_watermark import detect_watermark
        from utils.audio_watermark import is_available as watermark_available

        if watermark_available():
            report = detect_watermark(output_path)
            found["watermark"] = bool(report and report.get("watermarked"))
    except ImportError:
        pass

    return found
