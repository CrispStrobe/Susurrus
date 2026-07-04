"""Test progress line parsing."""

import unittest


class TestProgressParser(unittest.TestCase):
    def test_percent_format(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("  45%|###       | 12/26")
        self.assertAlmostEqual(result["progress"], 0.45)

    def test_float_format(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("progress = 0.750")
        self.assertAlmostEqual(result["progress"], 0.75)

    def test_float_format_colon(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("progress: 0.500")
        self.assertAlmostEqual(result["progress"], 0.50)

    def test_fraction_format(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("Processing: 5/20 segments")
        self.assertAlmostEqual(result["progress"], 0.25)

    def test_rtf_extraction(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("progress: 0.5 RTF=3.2")
        self.assertAlmostEqual(result["rtf"], 3.2)

    def test_wps_extraction(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("50% WPS=120.5")
        self.assertAlmostEqual(result["wps"], 120.5)

    def test_no_progress(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("Loading model...")
        self.assertNotIn("progress", result)

    def test_100_percent(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("100%")
        self.assertAlmostEqual(result["progress"], 1.0)

    def test_zero_total_fraction(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("0/0 done")
        self.assertNotIn("progress", result)

    def test_empty_string(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("")
        self.assertEqual(result, {})

    def test_combined_metrics(self):
        from utils.progress_parser import parse_progress_line

        result = parse_progress_line("75% RTF=5.1 WPS=200.0")
        self.assertAlmostEqual(result["progress"], 0.75)
        self.assertAlmostEqual(result["rtf"], 5.1)
        self.assertAlmostEqual(result["wps"], 200.0)


if __name__ == "__main__":
    unittest.main()
