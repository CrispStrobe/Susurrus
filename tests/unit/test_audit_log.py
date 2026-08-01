"""Test the Art. 12 biometric audit log."""

import argparse
import json
import os
import tempfile
import unittest

from utils.audit_log import (
    EVENT_ENROLL,
    EVENT_IDENTIFY,
    read_events,
    record_event,
    verify_chain,
)


class TestAuditLog(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "biometric.jsonl")

    def tearDown(self):
        self._dir.cleanup()

    def _record(self, **kw):
        kw.setdefault("event_type", EVENT_ENROLL)
        kw.setdefault("speaker", "alice")
        return record_event(path=self.path, **kw)

    def test_records_required_fields(self):
        entry = self._record(database="/db.json", consent=True, model="titanet-l")
        self.assertEqual(entry["event"], EVENT_ENROLL)
        self.assertEqual(entry["speaker"], "alice")
        self.assertEqual(entry["database"], "/db.json")
        self.assertTrue(entry["consent_attested"])
        self.assertEqual(entry["model"], "titanet-l")
        self.assertIn("timestamp", entry)
        self.assertIn("software", entry)

    def test_timestamp_is_utc_iso8601(self):
        entry = self._record()
        self.assertRegex(entry["timestamp"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_consent_defaults_to_false(self):
        """An unattested enrollment must be recorded as unattested."""
        self.assertFalse(self._record()["consent_attested"])

    def test_append_only(self):
        self._record(speaker="alice")
        self._record(speaker="bob")
        self._record(event_type=EVENT_IDENTIFY, speaker=None)

        events = read_events(self.path)
        self.assertEqual(len(events), 3)
        self.assertEqual([e["speaker"] for e in events], ["alice", "bob", None])
        self.assertEqual(events[2]["event"], EVENT_IDENTIFY)

    def test_never_stores_biometric_payload(self):
        """The log documents biometric processing; it must not duplicate it."""
        self._record(database="/db.json", model="titanet-l")
        raw = open(self.path, encoding="utf-8").read().lower()
        for forbidden in ("embedding", "xvector", "waveform", "samples", "audio_data"):
            self.assertNotIn(forbidden, raw)

    def test_missing_log_reads_as_empty(self):
        self.assertEqual(read_events(os.path.join(self._dir.name, "nope.jsonl")), [])

    def test_malformed_line_is_skipped_not_fatal(self):
        self._record(speaker="alice")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("{not json\n")
        self._record(speaker="bob")
        self.assertEqual(len(read_events(self.path)), 2)


class TestChainVerification(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "biometric.jsonl")
        for name in ("alice", "bob", "carol"):
            record_event(EVENT_ENROLL, speaker=name, path=self.path)

    def tearDown(self):
        self._dir.cleanup()

    def _rewrite(self, lines):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _lines(self):
        return open(self.path, encoding="utf-8").read().strip().split("\n")

    def test_intact_chain_verifies(self):
        result = verify_chain(self.path)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["entries"], 3)

    def test_entries_are_linked(self):
        events = read_events(self.path)
        self.assertEqual(events[0]["prev_hash"], "0" * 64)
        self.assertEqual(events[1]["prev_hash"], events[0]["entry_hash"])
        self.assertEqual(events[2]["prev_hash"], events[1]["entry_hash"])

    def test_detects_modified_entry(self):
        self._rewrite([line.replace('"alice"', '"mallory"') for line in self._lines()])
        result = verify_chain(self.path)
        self.assertFalse(result["valid"])
        self.assertTrue(any("modified" in e for e in result["errors"]))

    def test_detects_deleted_entry(self):
        lines = self._lines()
        self._rewrite([lines[0], lines[2]])  # drop the middle
        result = verify_chain(self.path)
        self.assertFalse(result["valid"])
        self.assertTrue(any("breaks the chain" in e for e in result["errors"]))

    def test_detects_truncated_head(self):
        self._rewrite(self._lines()[1:])
        self.assertFalse(verify_chain(self.path)["valid"])

    def test_detects_reordered_entries(self):
        lines = self._lines()
        self._rewrite([lines[1], lines[0], lines[2]])
        self.assertFalse(verify_chain(self.path)["valid"])

    def test_appending_after_verification_keeps_chain_valid(self):
        record_event(EVENT_IDENTIFY, speaker="dave", path=self.path)
        result = verify_chain(self.path)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["entries"], 4)

    def test_empty_log_is_valid(self):
        empty = os.path.join(self._dir.name, "empty.jsonl")
        result = verify_chain(empty)
        self.assertTrue(result["valid"])
        self.assertEqual(result["entries"], 0)


class TestCliAuditWiring(unittest.TestCase):
    """The CLI must record enrollment *and* identification."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("XDG_DATA_HOME")
        os.environ["XDG_DATA_HOME"] = self._dir.name

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("XDG_DATA_HOME", None)
        else:
            os.environ["XDG_DATA_HOME"] = self._prev
        self._dir.cleanup()

    def _ns(self, **over):
        base = dict(
            speaker_db=None,
            enroll_speaker=None,
            expect_speakers=None,
            speaker_db_consent=False,
            titanet_model=None,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_enrollment_is_recorded(self):
        import cli

        written = cli._audit_speaker_biometrics(
            self._ns(enroll_speaker="alice", speaker_db="/db.json", speaker_db_consent=True)
        )
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["event"], EVENT_ENROLL)
        self.assertTrue(written[0]["consent_attested"])

    def test_identification_is_recorded(self):
        """Art. 12 covers use, not only setup — matching must be logged too."""
        import cli

        written = cli._audit_speaker_biometrics(self._ns(expect_speakers="alice,bob"))
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["event"], EVENT_IDENTIFY)

    def test_nothing_recorded_without_biometrics(self):
        import cli

        self.assertEqual(cli._audit_speaker_biometrics(self._ns()), [])

    def test_written_entries_land_in_the_real_log(self):
        import cli
        from utils.audit_log import audit_log_path

        cli._audit_speaker_biometrics(self._ns(enroll_speaker="alice"))
        self.assertTrue(os.path.isfile(audit_log_path()))
        with open(audit_log_path(), encoding="utf-8") as f:
            self.assertEqual(json.loads(f.readline())["speaker"], "alice")


if __name__ == "__main__":
    unittest.main()
