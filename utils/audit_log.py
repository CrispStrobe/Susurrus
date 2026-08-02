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
* **Anchored.** A hash chain cannot detect truncation of its own tail: drop the
  last *n* entries and what remains is a shorter chain that still verifies.
  Detecting that needs a reference held outside the chain, so the entry count
  and head hash are mirrored into a sibling file after every append. Deleting
  the tail now contradicts the anchor. See :func:`verify_chain`.
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


def anchor_path(log_path=None):
    """Return the path to the anchor for *log_path*.

    A sibling rather than a section of the log itself: an anchor stored inside
    the thing it is anchoring moves with it, and would be truncated by the same
    operation it is supposed to detect.
    """
    log_path = log_path or audit_log_path()
    return f"{log_path}.anchor"


def _read_anchor(log_path):
    """Return the recorded ``{"entries": int, "head": str}``, or None."""
    try:
        with open(anchor_path(log_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "entries" not in data:
        return None
    return data


def _write_anchor(log_path, entries, head):
    """Mirror the chain head outside the chain, atomically.

    Written after the entry it describes, and replaced rather than rewritten in
    place: a torn anchor would report a truncation that never happened, and a
    record-keeping control that cries wolf gets switched off.
    """
    payload = {
        "entries": entries,
        "head": head,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    target = anchor_path(log_path)
    tmp = f"{target}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except OSError as e:
        logger.warning("Could not update audit anchor %s: %s", target, e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


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

    # After the append, never before: an anchor written first would claim an
    # entry that a failed write never produced, turning the truncation check
    # into a permanent false alarm.
    _write_anchor(log_path, _count_entries(log_path), entry["entry_hash"])

    logger.info("Audit: %s speaker=%s consent=%s", event_type, speaker, bool(consent))
    return entry


def _count_entries(path):
    """Count non-blank lines without parsing them."""
    total = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    except OSError:
        return 0
    return total


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
    """Verify the hash chain over the audit log, and check it against the anchor.

    The chain catches modification, reordering and deletion from the middle.
    It cannot catch deletion from the *end* — a truncated chain is still a
    valid one — so the recorded entry count and head hash are compared against
    the anchor written beside the log.

    Returns:
        dict with ``valid`` (bool), ``entries`` (int), ``anchor`` (the recorded
        anchor or None) and ``errors`` (human-readable strings).
    """
    log_path = path or audit_log_path()
    entries = read_events(log_path)
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

    anchor = _read_anchor(log_path)
    if anchor is not None:
        recorded = anchor.get("entries", 0)
        if len(entries) < recorded:
            errors.append(
                f"{recorded - len(entries)} entries are missing from the end of "
                f"the log: the anchor records {recorded}, the file has "
                f"{len(entries)}. The hash chain cannot see this on its own — "
                "entries were removed from the tail."
            )
        elif (
            len(entries) == recorded
            and entries
            and anchor.get("head")
            not in (
                None,
                entries[-1].get("entry_hash"),
            )
        ):
            errors.append(
                "the final entry does not match the anchored head: the last "
                "entry was replaced after it was written"
            )
    elif entries:
        # Not an error. Logs written before anchoring existed have none, and
        # calling that tampering would flag every existing deployment.
        logger.info(
            "Audit log has no anchor, so truncation of the tail cannot be "
            "checked. It will be anchored on the next recorded event."
        )

    return {
        "valid": not errors,
        "entries": len(entries),
        "anchor": anchor,
        "errors": errors,
    }


def _version():
    try:
        from __init__ import __version__

        return __version__
    except ImportError:
        return "unknown"
