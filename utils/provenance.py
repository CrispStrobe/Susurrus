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

**This pipeline fails closed.** If no machine-readable layer lands, or a
cloning run cannot deliver its audible disclosure, the output file is deleted
and :class:`ProvenanceError` is raised. Art. 50(2) has no "unless a dependency
is missing" clause, so an install that cannot mark must not synthesize: the one
outcome the Regulation rules out is unmarked synthetic audio reaching a person,
and warning about it while leaving the file on disk does not prevent that.

The single way past it is ``accept_marking_responsibility``, which is an
attestation that the operator is taking the Art. 50 duty on. Reusing the
existing opt-out rather than adding a second switch keeps one rule: either the
software marked the output, or a named human said they would.
"""

import logging
import os

logger = logging.getLogger(__name__)


class ProvenanceError(RuntimeError):
    """Raised when synthetic audio cannot be marked or disclosed.

    Carries the path that was refused so callers can name it. The file itself
    is already gone by the time this is raised — see :func:`enforce_marking`.
    """

    def __init__(self, message, output_path=None):
        super().__init__(message)
        self.output_path = output_path


#: Keys present in every result dict returned by :func:`apply_provenance`.
#:
#: ``spoken_required`` is deliberately separate from ``spoken``: whether the
#: disclosure was *owed* is not the same question as whether it landed. Without
#: the distinction a run whose disclosure failed looked identical to one that
#: never needed a disclosure, and callers reported success for both. Art. 50(4)
#: engages on cloning *or* on a preset voice that belongs to a real person —
#: see :mod:`utils.speaker_identity`.
_EMPTY = {
    "spoken": False,
    "spoken_required": False,
    "suppressed_spoken": False,
    "watermark": False,
    "marker": False,
    "c2pa": False,
    "opted_out": False,
    "unsupported_format": False,
    # Whose voice this was, per utils.speaker_identity. Reported so an operator
    # can see *why* a disclosure was or was not owed, rather than inferring it.
    "speaker_identity": "unknown",
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


def disclosure_missing(result):
    """True if an Art. 50(4) audible disclosure was owed and did not happen.

    Distinct from :func:`marking_applied`, which answers the Art. 50(2)
    machine-readable question. The two obligations are separate, and a run can
    satisfy one while failing the other — so callers that only checked for
    marking reported a confident success over cloned audio that announced
    nothing to whoever hears it.

    An operator who passed ``--no-spoken-disclaimer`` (which itself requires
    the responsibility attestation) has made that choice knowingly, so it is
    not reported as a shortfall.
    """
    return bool(
        result.get("spoken_required")
        and not result.get("spoken")
        and not result.get("suppressed_spoken")
        and not result.get("opted_out")
    )


#: Human-readable install hint, appended to every refusal. A refusal that does
#: not say how to satisfy it just reads as breakage.
_REMEDY = (
    "Install the marking stack with \"pip install 'susurrus[tts]'\" (adds "
    "soundfile + C2PA), or write .wav / .mp3 output, which needs no optional "
    "dependency. To ship unmarked audio deliberately, pass "
    "--accept-marking-responsibility (CLI) or tick 'I accept marking "
    "responsibility' (GUI), which attests that the EU AI Act Art. 50 duty "
    "rests with you."
)


def _discard(output_path):
    """Delete audio that may not be released. Returns True if it is gone.

    Refusing while leaving the file behind would be theatre: the unmarked
    audio would still be on disk under the name the user asked for, ready to
    be shipped by anyone who ignores an exit code. Deleting it undoes this
    run's own write — synthesis just created or overwrote this path, so there
    is no pre-existing user data here to lose.
    """
    try:
        os.unlink(output_path)
        return True
    except OSError as e:
        logger.error(
            "Could not delete unmarked audio %s: %s — this file is synthetic "
            "and carries no EU AI Act Art. 50 marking. Delete it manually.",
            output_path,
            e,
        )
        return False


def enforce_marking(result, output_path):
    """Delete and refuse *output_path* if it may not be released. Fails closed.

    Called at the end of every ``apply_provenance`` implementation — the base
    one here and the two CrispASR overrides that verify rather than apply — so
    no synthesis route can opt out of the check by having its own marking
    logic.

    Returns *result* unchanged when the output is releasable.

    Raises:
        ProvenanceError: if no machine-readable layer landed (Art. 50(2)) or a
            cloning run produced no audible disclosure (Art. 50(4)).
    """
    if result.get("opted_out"):
        return result

    if not marking_applied(result):
        detail = ""
        if result.get("unsupported_format"):
            ext = os.path.splitext(output_path or "")[1].lower() or "this container"
            detail = (
                f" {ext} has no declarative marker, so marking it needs "
                "C2PA or the in-sample watermark, and neither is available."
            )
        _discard(output_path)
        raise ProvenanceError(
            f"Refusing to write unmarked synthetic audio to {output_path}. EU "
            "AI Act Art. 50(2) requires machine-readable marking of synthetic "
            f"audio and no layer could be applied.{detail} {_REMEDY}",
            output_path,
        )

    if disclosure_missing(result):
        # Name *why* it was owed. "Cloned voice" was the only reason once, and
        # reading that over a stock-voice run would send the operator looking
        # for a reference file they never passed.
        because = (
            "this preset voice belongs to an identifiable person"
            if result.get("speaker_identity") == "real_person"
            else "this audio clones a voice"
        )
        _discard(output_path)
        raise ProvenanceError(
            f"Refusing to write undisclosed deepfake audio to {output_path}: "
            f"{because}. EU AI Act Art. 50(4) requires disclosure that deepfake "
            "content is artificially generated, and the audible disclosure "
            "could not be produced. Machine-readable marking does not reach a "
            f"listener. {_REMEDY}",
            output_path,
        )

    return result


def marking_available(output_path, is_cloning=False):
    """Preflight: can this install mark *output_path*? Returns (ok, reason).

    Advisory, and deliberately cheap — it inspects the extension and what
    imports, never the file, and loads no models. :func:`enforce_marking` is
    the authoritative check because only it can see what actually landed.

    The point of asking early is that a refusal should cost nothing. Without
    it, an install that cannot mark FLAC still loads the model, synthesizes,
    and only then throws the result away — the user waits minutes to be told
    something knowable in milliseconds.
    """
    ext = os.path.splitext(output_path or "")[1].lower()

    if ext in _MARKABLE:
        declarative = True
    else:
        declarative = False

    if not declarative:
        if not (_c2pa_installed() or _soundfile_installed()):
            return False, (
                f"No marking layer can be applied to {ext or 'this container'}: "
                "it has no declarative marker, and neither C2PA nor the "
                "in-sample watermark is installed."
            )

    # Art. 50(4) needs to concatenate audio. WAV goes through the stdlib;
    # every other container needs a decoder.
    if is_cloning and ext != ".wav" and not _soundfile_installed():
        return False, (
            f"Voice cloning to {ext or 'this container'} needs an audible "
            "disclosure, and concatenating non-WAV audio requires soundfile, "
            "which is not installed."
        )

    return True, ""


def _c2pa_installed():
    try:
        from utils.c2pa_signing import is_available

        return bool(is_available())
    except ImportError:
        return False


def _soundfile_installed():
    try:
        import soundfile  # noqa: F401

        return True
    except ImportError:
        return False


def apply_provenance(
    output_path,
    options=None,
    model=None,
    backend=None,
    is_cloning=False,
    locale=None,
    speaker_backend=None,
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
        is_cloning: Whether this synthesis cloned a voice, which is one of the
            two things that engages the Art. 50(4) audible disclosure.
        locale: Language for the spoken disclosure.
        speaker_backend: TTS backend name, used to decide whether the *preset*
            voice belongs to a real person — the other thing that engages it.
            See :mod:`utils.speaker_identity`.

    Returns:
        dict with ``spoken``, ``spoken_required``, ``suppressed_spoken``,
        ``watermark``, ``marker``, ``c2pa``, ``opted_out`` and
        ``unsupported_format``.

    Raises:
        ProvenanceError: if the output cannot be marked or disclosed. The file
            is deleted first — see :func:`enforce_marking`.
    """
    options = options or {}
    result = new_result()

    # Nothing to mark and nothing to delete. Callers pass paths that a failed
    # synthesis never created, and refusing over a file that does not exist
    # would turn a backend error into a confusing compliance error.
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

    # 1. Spoken disclosure. Owed when the voice is cloned *or* when the preset
    #    belongs to an identifiable person — Art. 3(60) turns on the output
    #    resembling someone, not on how the resemblance was obtained, and a
    #    Piper voice trained on one speaker's corpus resembles that speaker
    #    whether or not a reference WAV was passed. This used to key on
    #    is_cloning alone, which let every real-person preset out undisclosed.
    from utils.speaker_identity import requires_spoken_disclosure, resolve_speaker_identity

    identity = resolve_speaker_identity(
        backend=speaker_backend or getattr(backend, "backend_name", None),
        override=options.get("speaker_identity"),
    )
    needs_spoken = requires_spoken_disclosure(bool(is_cloning), identity, backend=speaker_backend)

    result["speaker_identity"] = identity
    result["spoken_required"] = needs_spoken
    result["suppressed_spoken"] = bool(needs_spoken and options.get("no_spoken_disclaimer"))
    if needs_spoken and backend is not None and not options.get("no_spoken_disclaimer"):
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

    # Fails closed: deletes the file and raises rather than returning a result
    # that says "unmarked" and trusting every caller to act on it.
    return enforce_marking(result, output_path)


def complete_marking(output_path, options=None, model=None):
    """Mark audio that something else produced: verify first, fill only gaps.

    For audio Susurrus did not synthesize itself — the marking proxy's
    responses from the CrispASR server, for instance. :func:`apply_provenance`
    is the wrong entry point there: it applies every layer unconditionally,
    which is correct for a Python-native backend that marked nothing, and wrong
    for output that arrives already marked. Measured on real Piper output from
    the server, the binary's own in-sample watermark reads at 0.815 and
    re-embedding over it costs ~41 dB SNR to raise a mark that already cleared
    the threshold. C2PA would stack a second manifest on the first.

    So the policy is the one the CrispASR backends already use:

    * the declarative marker is applied **unconditionally** — it is idempotent,
      dependency-free, does not touch the samples, and must not depend on a
      detector being right;
    * the in-sample watermark and C2PA are applied only when verification says
      they are *absent*, never when it says "cannot tell". An unknown answer is
      not a reason to damage the samples on the chance it helps, and the
      declarative floor has already satisfied the machine-readable duty.

    Raises:
        ProvenanceError: if nothing landed. The file is deleted first.
    """
    options = options or {}
    result = new_result()

    if not output_path or not os.path.isfile(output_path):
        return result

    ext = os.path.splitext(output_path)[1].lower()
    result["unsupported_format"] = ext not in _MARKABLE

    found = verify_marking(output_path)
    result["watermark"] = bool(found.get("watermark"))
    result["c2pa"] = bool(found.get("c2pa"))

    # Layer order is the same as apply_provenance's, and for the same reason.
    # Marking first and watermarking second looks harmless and is not: the
    # watermarker round-trips the file through soundfile, which rewrites it
    # from the samples and drops the RIFF chunk the marker had just appended.
    # The result claimed a marker that was no longer on disk.

    # 1. In-sample watermark, only where verification says there is none.
    if found.get("watermark") is False and not options.get("no_watermark"):
        try:
            from utils.audio_watermark import embed_watermark

            result["watermark"] = embed_watermark(output_path)
        except ImportError:
            pass

    # 2. The declarative floor, always — after anything that rewrites samples.
    try:
        from utils.ai_marking import embed_ai_marker

        result["marker"] = bool(embed_ai_marker(output_path, model=model))
    except ImportError:
        result["marker"] = bool(found.get("marker"))

    # 3. C2PA last: it hashes the finished bytes, so everything above is done.
    if found.get("c2pa") is False and not options.get("no_c2pa"):
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

    return enforce_marking(result, output_path)


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
