"""Test theme definitions and color utilities."""

import unittest


class TestSpeakerColors(unittest.TestCase):
    def test_8_distinct_colors(self):
        from gui.themes import SPEAKER_COLORS

        self.assertEqual(len(SPEAKER_COLORS), 8)
        self.assertEqual(len(set(SPEAKER_COLORS)), 8)

    def test_speaker_color_cycles(self):
        from gui.themes import speaker_color

        self.assertEqual(speaker_color(0), speaker_color(8))
        self.assertEqual(speaker_color(1), speaker_color(9))

    def test_speaker_color_valid_hex(self):
        from gui.themes import speaker_color

        for i in range(8):
            c = speaker_color(i)
            self.assertTrue(c.startswith("#"))
            self.assertEqual(len(c), 7)


class TestConfidenceColors(unittest.TestCase):
    def test_high_confidence(self):
        from gui.themes import confidence_color

        color, label = confidence_color(0.95)
        self.assertEqual(label, "high")
        self.assertIn("4CAF50", color)  # green

    def test_medium_confidence(self):
        from gui.themes import confidence_color

        color, label = confidence_color(0.7)
        self.assertEqual(label, "medium")
        self.assertIn("FF9800", color)  # orange

    def test_low_confidence(self):
        from gui.themes import confidence_color

        color, label = confidence_color(0.3)
        self.assertEqual(label, "low")
        self.assertIn("F44336", color)  # red

    def test_none_confidence(self):
        from gui.themes import confidence_color

        color, label = confidence_color(None)
        self.assertEqual(label, "unknown")

    def test_boundary_08(self):
        from gui.themes import confidence_color

        _, label = confidence_color(0.8)
        self.assertEqual(label, "high")

    def test_boundary_06(self):
        from gui.themes import confidence_color

        _, label = confidence_color(0.6)
        self.assertEqual(label, "medium")


class TestThemes(unittest.TestCase):
    def test_themes_registered(self):
        from gui.themes import THEMES

        self.assertIn("dark", THEMES)
        self.assertIn("light", THEMES)

    def test_theme_is_string(self):
        from gui.themes import THEMES

        for name, css in THEMES.items():
            self.assertIsInstance(css, str, f"{name} theme is not a string")
            self.assertGreater(len(css), 100, f"{name} theme too short")

    def test_dark_theme_has_dark_bg(self):
        from gui.themes import DARK_THEME

        self.assertIn("#1e1e1e", DARK_THEME)

    def test_light_theme_has_light_bg(self):
        from gui.themes import LIGHT_THEME

        self.assertIn("#fafafa", LIGHT_THEME)


if __name__ == "__main__":
    unittest.main()
