"""Test i18n translations."""

import unittest


class TestI18n(unittest.TestCase):
    def test_english_default(self):
        from utils.i18n import get_locale, t

        self.assertEqual(get_locale(), "en")
        self.assertEqual(t("btn.transcribe"), "Transcribe")

    def test_german(self):
        from utils.i18n import set_locale, t

        set_locale("de")
        self.assertEqual(t("btn.transcribe"), "Transkribieren")
        self.assertEqual(t("tab.history"), "Verlauf")
        set_locale("en")  # reset

    def test_fallback_to_english(self):
        from utils.i18n import set_locale, t

        set_locale("de")
        # Keys not in German should fall back to English
        result = t("nonexistent.key")
        self.assertEqual(result, "nonexistent.key")
        set_locale("en")

    def test_unknown_locale_falls_back(self):
        from utils.i18n import get_locale, set_locale

        set_locale("xx")
        self.assertEqual(get_locale(), "en")

    def test_available_locales(self):
        from utils.i18n import available_locales

        locales = available_locales()
        self.assertIn("en", locales)
        self.assertIn("de", locales)

    def test_all_en_keys_exist_in_de(self):
        from utils.i18n import TRANSLATIONS

        en_keys = set(TRANSLATIONS["en"].keys())
        de_keys = set(TRANSLATIONS["de"].keys())
        missing = en_keys - de_keys
        self.assertEqual(missing, set(), f"German missing keys: {missing}")

    def test_eu_ai_act_warning_in_both(self):
        from utils.i18n import TRANSLATIONS

        self.assertIn("Art. 50", TRANSLATIONS["en"]["warn.no_watermark"])
        self.assertIn("Art. 50", TRANSLATIONS["de"]["warn.no_watermark"])


if __name__ == "__main__":
    unittest.main()
