"""Transcription history persistence — JSON file-based storage."""

import json
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)


def _history_dir():
    """Get the history storage directory (XDG-compliant)."""
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    d = os.path.join(xdg, "susurrus", "history")
    os.makedirs(d, exist_ok=True)
    return d


class HistoryEntry:
    """A single transcription history entry."""

    def __init__(
        self,
        entry_id=None,
        created_at=None,
        source_path=None,
        backend=None,
        model=None,
        language=None,
        segments=None,
        duration=None,
        speaker_names=None,
        full_text=None,
    ):
        self.id = entry_id or str(uuid.uuid4())
        self.created_at = created_at or time.time()
        self.source_path = source_path
        self.backend = backend
        self.model = model
        self.language = language
        self.segments = segments or []  # list of (start, end, text) or dicts
        self.duration = duration
        self.speaker_names = speaker_names or {}
        self.full_text = full_text

    def to_dict(self):
        segs = []
        for s in self.segments:
            if isinstance(s, dict):
                segs.append(s)
            else:
                segs.append({"start": s[0], "end": s[1], "text": s[2]})
        return {
            "id": self.id,
            "created_at": self.created_at,
            "source_path": self.source_path,
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "segments": segs,
            "duration": self.duration,
            "speaker_names": self.speaker_names,
            "full_text": self.full_text,
        }

    @classmethod
    def from_dict(cls, d):
        segs = []
        for s in d.get("segments", []):
            if isinstance(s, dict):
                segs.append((s.get("start", 0.0), s.get("end", 0.0), s.get("text", "")))
            else:
                segs.append(tuple(s))
        return cls(
            entry_id=d.get("id"),
            created_at=d.get("created_at"),
            source_path=d.get("source_path"),
            backend=d.get("backend"),
            model=d.get("model"),
            language=d.get("language"),
            segments=segs,
            duration=d.get("duration"),
            speaker_names=d.get("speaker_names", {}),
            full_text=d.get("full_text"),
        )

    @property
    def title(self):
        if self.source_path:
            return os.path.basename(self.source_path)
        return f"Transcription {self.id[:8]}"

    @property
    def created_at_str(self):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created_at))


class HistoryService:
    """Manages transcription history on disk."""

    def __init__(self, history_dir=None):
        self._dir = history_dir or _history_dir()

    def save(self, entry):
        """Save a HistoryEntry to disk."""
        path = os.path.join(self._dir, f"{entry.id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info("History saved: %s", entry.id)
        except Exception as e:
            logger.warning("Failed to save history %s: %s", entry.id, e)

    def load(self, entry_id):
        """Load a single HistoryEntry by ID."""
        path = os.path.join(self._dir, f"{entry_id}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return HistoryEntry.from_dict(json.load(f))
        except Exception as e:
            logger.warning("Failed to load history %s: %s", entry_id, e)
            return None

    def list_entries(self):
        """List all history entries, newest first."""
        entries = []
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entries.append(HistoryEntry.from_dict(json.load(f)))
            except Exception as e:
                logger.warning("Skipping corrupt history file %s: %s", fname, e)
        entries.sort(key=lambda e: e.created_at or 0, reverse=True)
        return entries

    def delete(self, entry_id):
        """Delete a history entry."""
        path = os.path.join(self._dir, f"{entry_id}.json")
        if os.path.isfile(path):
            os.remove(path)
            logger.info("History deleted: %s", entry_id)

    def search(self, query):
        """Search history entries by substring match on title and full_text."""
        query_lower = query.lower()
        results = []
        for entry in self.list_entries():
            title = (entry.title or "").lower()
            text = (entry.full_text or "").lower()
            if query_lower in title or query_lower in text:
                results.append(entry)
        return results

    def clear_all(self):
        """Delete all history entries."""
        for fname in os.listdir(self._dir):
            if fname.endswith(".json"):
                os.remove(os.path.join(self._dir, fname))
        logger.info("History cleared")
