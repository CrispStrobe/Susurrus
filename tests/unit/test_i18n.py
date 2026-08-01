"""Test the i18n catalog and lookup.

The parity tests are the point: a missing German string silently falls back to
English, which is a cosmetic bug for ordinary labels and a compliance problem
for consent text. Failing the suite is how a translator finds out.
"""

import ast
import glob
import os
import re
import unittest

from utils.i18n import (
    TRANSLATIONS,
    available_locales,
    detect_system_locale,
    get_locale,
    locale_name,
    set_locale,
    t,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


class TestLookup(unittest.TestCase):
    def setUp(self):
        set_locale("en")

    def tearDown(self):
        set_locale("en")

    def test_english_default(self):
        self.assertEqual(get_locale(), "en")
        self.assertEqual(t("btn.transcribe"), "Transcribe")

    def test_german(self):
        set_locale("de")
        self.assertEqual(t("btn.transcribe"), "Transkribieren")
        self.assertEqual(t("tab.history"), "Verlauf")

    def test_unknown_key_returns_the_key(self):
        set_locale("de")
        self.assertEqual(t("nonexistent.key"), "nonexistent.key")

    def test_unknown_locale_falls_back(self):
        set_locale("xx")
        self.assertEqual(get_locale(), "en")

    def test_per_call_locale_override(self):
        """A spoken disclosure picks its language per call, not per session."""
        self.assertEqual(t("btn.save", locale="de"), "Speichern")
        self.assertEqual(get_locale(), "en", "override must not leak")

    def test_available_locales(self):
        self.assertIn("en", available_locales())
        self.assertIn("de", available_locales())

    def test_locale_names(self):
        self.assertEqual(locale_name("de"), "Deutsch")
        self.assertEqual(locale_name("zz"), "zz")

    def test_detect_system_locale_returns_known_code(self):
        self.assertIn(detect_system_locale(), available_locales())

    def test_detect_system_locale_honours_env(self):
        prev = os.environ.get("LANGUAGE")
        try:
            os.environ["LANGUAGE"] = "de_DE.UTF-8"
            self.assertEqual(detect_system_locale(), "de")
        finally:
            if prev is None:
                os.environ.pop("LANGUAGE", None)
            else:
                os.environ["LANGUAGE"] = prev


class TestCatalogParity(unittest.TestCase):
    """Every locale must define every key — no silent English fallback."""

    def test_all_locales_have_all_keys(self):
        reference = set(TRANSLATIONS["en"])
        for code, strings in TRANSLATIONS.items():
            missing = reference - set(strings)
            self.assertEqual(missing, set(), f"locale '{code}' is missing: {sorted(missing)}")

    def test_no_locale_has_orphan_keys(self):
        """A key only in German means a typo, or a string English forgot."""
        reference = set(TRANSLATIONS["en"])
        for code, strings in TRANSLATIONS.items():
            orphans = set(strings) - reference
            self.assertEqual(orphans, set(), f"locale '{code}' has orphans: {sorted(orphans)}")

    def test_no_empty_values(self):
        for code, strings in TRANSLATIONS.items():
            for key, value in strings.items():
                self.assertTrue(str(value).strip(), f"{code}:{key} is empty")

    def test_translations_actually_differ_from_english(self):
        """Guards against a catalog copy-pasted and never translated."""
        english = TRANSLATIONS["en"]
        for code, strings in TRANSLATIONS.items():
            if code == "en":
                continue
            identical = [k for k, v in strings.items() if v == english.get(k)]
            # Proper nouns and codes legitimately match ("Susurrus", "de").
            self.assertLess(
                len(identical) / len(english),
                0.2,
                f"locale '{code}' looks untranslated: {len(identical)} identical values",
            )

    def test_format_placeholders_match_across_locales(self):
        """A dropped {count} raises KeyError at runtime, in that locale only."""

        def placeholders(value):
            return set(re.findall(r"\{(\w+)\}", str(value)))

        english = TRANSLATIONS["en"]
        for code, strings in TRANSLATIONS.items():
            for key, value in strings.items():
                self.assertEqual(
                    placeholders(value),
                    placeholders(english[key]),
                    f"{code}:{key} has mismatched format placeholders",
                )


class TestComplianceStrings(unittest.TestCase):
    """Consent and Art. 50 text must exist and be localized in every locale."""

    COMPLIANCE_KEYS = [
        "consent.clone_checkbox",
        "consent.clone_detail",
        "warn.no_watermark",
        "warn.marking_opted_out",
        "warn.marking_failed",
        "warn.speaker_db_consent",
        "disclosure.spoken",
    ]

    def test_present_in_every_locale(self):
        for code, strings in TRANSLATIONS.items():
            for key in self.COMPLIANCE_KEYS:
                self.assertIn(key, strings, f"{code} is missing {key}")

    def test_compliance_text_is_translated_not_english(self):
        for code, strings in TRANSLATIONS.items():
            if code == "en":
                continue
            for key in self.COMPLIANCE_KEYS:
                self.assertNotEqual(
                    strings[key],
                    TRANSLATIONS["en"][key],
                    f"{code}:{key} is still English — consent text must be localized",
                )

    def test_eu_ai_act_is_cited_in_every_locale(self):
        for code in TRANSLATIONS:
            self.assertIn("Art. 50", TRANSLATIONS[code]["warn.no_watermark"])
            self.assertIn("Art. 50", TRANSLATIONS[code]["consent.clone_detail"])


class TestGuiIsMigrated(unittest.TestCase):
    """No user-visible string may be hardcoded in the GUI layer.

    This is the regression guard for the migration itself: without it, the
    next widget someone adds quietly reintroduces English-only text.
    """

    #: Widget constructors and text setters whose string arguments are shown.
    #:
    #: This list started as the seven entries that end at ``setWindowTitle``,
    #: which let the About dialog, the diarization help and every TTS status
    #: line stay hardcoded English while the suite reported the GUI fully
    #: migrated. Text setters are where long-form user-facing prose lives.
    UI_CALLS = {
        "QLabel",
        "QPushButton",
        "QCheckBox",
        "QAction",
        "QGroupBox",
        "QRadioButton",
        "setPlaceholderText",
        "setToolTip",
        "setWindowTitle",
        "setText",
        "setInformativeText",
        "setDetailedText",
        "appendPlainText",
    }

    #: Dialog methods, matched only on a ``QMessageBox``/``QInputDialog``
    #: receiver. Matching the bare attribute name would sweep up every
    #: ``logging.warning()`` and ``logger.critical()`` in the GUI layer —
    #: log lines are not user-visible strings, and drowning the report in
    #: them is how a scan stops being read.
    DIALOG_RECEIVERS = {"QMessageBox", "QInputDialog", "QFileDialog"}
    DIALOG_CALLS = {
        "about",
        "information",
        "warning",
        "critical",
        "question",
        "getText",
    }

    #: Strings that are deliberately not translated (identifiers, filenames,
    #: proper nouns) or are assembled from translated parts elsewhere.
    ALLOWED = {
        "auto",
        "Auto",
        "CPU",
        "GPU",
        "MPS",
        "Susurrus",
        "tts_output.wav",
    }

    #: Words allowed to appear literally inside an f-string — markup and
    #: units, not prose.
    FSTRING_ALLOWED_WORDS = frozenset(
        {"br", "div", "span", "code", "pre", "h1", "h2", "h3", "h4", "ul", "li", "ol", "table"}
    )

    def _ui_calls(self):
        """Yield (relative_path, node, call_name) for every UI-text call."""
        pattern = os.path.join(REPO_ROOT, "gui", "**", "*.py")
        for path in sorted(glob.glob(pattern, recursive=True)):
            if "__pycache__" in path:
                continue
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            rel = os.path.relpath(path, REPO_ROOT)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in self.UI_CALLS:
                    yield rel, node, name
                elif name in self.DIALOG_CALLS and isinstance(node.func, ast.Attribute):
                    receiver = getattr(node.func.value, "id", None)
                    if receiver in self.DIALOG_RECEIVERS:
                        yield rel, node, f"{receiver}.{name}"

    def test_no_hardcoded_ui_strings(self):
        offenders = []
        for rel, node, name in self._ui_calls():
            for arg in node.args:
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    continue
                text = arg.value.strip()
                if len(text) <= 2 or not any(c.isalpha() for c in text):
                    continue
                if text in self.ALLOWED:
                    continue
                offenders.append(f"{rel}:{node.lineno} {name}({text[:60]!r})")

        self.assertEqual(
            offenders,
            [],
            "hardcoded user-visible strings found; route them through t():\n"
            + "\n".join(offenders),
        )

    def test_no_english_prose_inside_fstrings(self):
        """f-strings are the blind spot the plain-constant scan misses.

        `setText(f"{done}/{total} done")` renders English in every locale but
        contains no string constant, so the scan above walks straight past it.
        Interpolating translated parts is fine; literal prose is not.
        """
        offenders = []
        for rel, node, name in self._ui_calls():
            for arg in node.args:
                if not isinstance(arg, ast.JoinedStr):
                    continue
                literal = "".join(
                    v.value
                    for v in arg.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                )
                literal = re.sub(r"<[^>]*>", " ", literal)  # drop HTML markup
                words = [w for w in re.findall(r"[A-Za-z]{3,}", literal)]
                prose = [w for w in words if w.lower() not in self.FSTRING_ALLOWED_WORDS]
                if prose:
                    offenders.append(f"{rel}:{node.lineno} {name}(f-string with {prose})")

        self.assertEqual(
            offenders,
            [],
            "English prose inside f-strings; move it into a catalog key with "
            "format placeholders:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
