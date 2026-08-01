"""Append-only audit log for biometric events (EU AI Act Art. 12).

Art. 12 requires high-risk AI systems to automatically record events over
their lifetime. A deployment that enrolls named speakers and identifies them
from audio may fall under Annex III(1)(a), in which case the deployer has to
be able to show *who* was enrolled, *when*, and under what attestation.

Design constraints that follow from what this log is for:

* **Append-only, one JSON object per line.** Survives partial writes; a
  truncated final line costs one entry, not the file.
* **Hash-chained.** Each entry carries the SHA-256 of the previous one, so
  deleting or editing history is detectable. This is tamper-*evidence*, not
  tamper-proofing — anyone who can write the file can rebuild the chain. For
  stronger guarantees ship the log somewhere append-only.
* **No biometric payload, ever.** The log records that an enrollment happened,
  never the embedding or the audio. A record-keeping mechanism must not become
  a second copy of the special-category data it documents.
"""

import hashlib
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

#: Event types recorded by this module.
EVENT_ENROLL = "speaker_enroll"
EVENT_IDENTIFY = "speaker_identify"

#: Keys that must never appear in an audit entry — see the module docstring.
_FORBIDDEN_KEYS = frozenset(
    {"embedding", "embeddings", "xvector", "audio", "audio_data", "samples", "waveform"}
)

_GENESIS = "0" * 64


def audit_dir():
    """Return the audit log directory (XDG-compliant), creating it."""
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    d = os.path.join(xdg, "susurrus", "audit")
    os.makedirs(d, exist_ok=True)
    return d


def audit_log_path():
    """Return the path to the biometric audit log."""
    return os.path.join(audit_dir(), "biometric.jsonl")


def _entry_hash(entry):
    """Hash an entry's content deterministically, excluding its own hash."""
    payload = {k: v for k, v in entry.items() if k != "entry_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _last_hash(path):
    """Return the hash of the final entry, or the genesis value if empty."""
    if not os.path.isfile(path):
        return _GENESIS
    last = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
    except OSError as e:
        logger.warning("Could not read audit log %s: %s", path, e)
        return _GENESIS
    if not last:
        return _GENESIS
    try:
        return json.loads(last).get("entry_hash", _GENESIS)
    except json.JSONDecodeError:
        # A truncated tail must not silently restart the chain.
        logger.warning("Audit log tail is malformed; chain continuity broken")
        return _GENESIS


def record_event(event_type, speaker=None, database=None, consent=False, model=None, path=None):
    """Append a biometric event to the audit log.

    Args:
        event_type: One of ``EVENT_ENROLL`` / ``EVENT_IDENTIFY``.
        speaker: Speaker name or label involved, if any.
        database: Path to the speaker database in use.
        consent: Whether a consent attestation was supplied.
        model: Embedding model identifier.
        path: Override the log path (tests).

    Returns:
        The written entry as a dict, or None if writing failed.
    """
    log_path = path or audit_log_path()

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event_type,
        "speaker": speaker,
        "database": database,
        "consent_attested": bool(consent),
        "model": model,
        "software": f"Susurrus {_version()}",
        "prev_hash": _last_hash(log_path),
    }

    leaked = _FORBIDDEN_KEYS.intersection(entry)
    if leaked:  # pragma: no cover — guards against future edits to this dict
        raise ValueError(f"audit entry must not carry biometric payload: {sorted(leaked)}")

    entry["entry_hash"] = _entry_hash(entry)

    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError as e:
        # Never fail the user's transcription because logging failed, but do
        # say so — a silently absent audit trail is worse than a noisy one.
        logger.warning("Could not write audit log %s: %s", log_path, e)
        return None

    logger.info("Audit: %s speaker=%s consent=%s", event_type, speaker, bool(consent))
    return entry


def read_events(path=None):
    """Read all audit entries. Malformed lines are skipped with a warning."""
    log_path = path or audit_log_path()
    if not os.path.isfile(log_path):
        return []

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed audit entry at line %d", lineno)
    return entries


def verify_chain(path=None):
    """Verify the hash chain over the audit log.

    Returns:
        dict with ``valid`` (bool), ``entries`` (int) and ``errors`` (list of
        human-readable strings naming the first failure of each kind).
    """
    entries = read_events(path)
    errors = []
    expected_prev = _GENESIS

    for index, entry in enumerate(entries):
        if entry.get("prev_hash") != expected_prev:
            errors.append(
                f"entry {index} breaks the chain: expected prev_hash "
                f"{expected_prev[:12]}…, found {str(entry.get('prev_hash'))[:12]}…"
            )
        recomputed = _entry_hash(entry)
        if entry.get("entry_hash") != recomputed:
            errors.append(f"entry {index} was modified after it was written")
        expected_prev = entry.get("entry_hash", _GENESIS)

    return {"valid": not errors, "entries": len(entries), "errors": errors}


def _version():
    try:
        from __init__ import __version__

        return __version__
    except ImportError:
        return "unknown"
