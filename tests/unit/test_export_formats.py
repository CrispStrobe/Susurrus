"""Test export format converters."""

import json
import unittest


class TestExportSRT(unittest.TestCase):
    def test_basic_srt(self):
        from utils.export_formats import export_srt

        segs = [(0.0, 2.5, "Hello world"), (2.5, 5.0, "Second line")]
        result = export_srt(segs)
        self.assertIn("1\n00:00:00,000 --> 00:00:02,500\nHello world", result)
        self.assertIn("2\n00:00:02,500 --> 00:00:05,000\nSecond line", result)

    def test_srt_long_timestamps(self):
        from utils.export_formats import export_srt

        segs = [(3661.5, 3723.0, "Over an hour in")]
        result = export_srt(segs)
        self.assertIn("01:01:01,500 --> 01:02:03,000", result)

    def test_srt_empty(self):
        from utils.export_formats import export_srt

        self.assertEqual(export_srt([]), "")

    def test_srt_dict_segments(self):
        from utils.export_formats import export_srt

        segs = [{"start": 1.0, "end": 2.0, "text": "Dict segment"}]
        result = export_srt(segs)
        self.assertIn("Dict segment", result)

    def test_srt_unicode(self):
        from utils.export_formats import export_srt

        segs = [(0.0, 1.0, "Hallo Welt — Ärger mit Ümlauten")]
        result = export_srt(segs)
        self.assertIn("Ärger mit Ümlauten", result)


class TestExportVTT(unittest.TestCase):
    def test_basic_vtt(self):
        from utils.export_formats import export_vtt

        segs = [(0.0, 2.5, "Hello world")]
        result = export_vtt(segs)
        self.assertTrue(result.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:02.500", result)
        self.assertIn("Hello world", result)

    def test_vtt_empty(self):
        from utils.export_formats import export_vtt

        result = export_vtt([])
        self.assertTrue(result.startswith("WEBVTT"))


class TestExportJSON(unittest.TestCase):
    def test_basic_json(self):
        from utils.export_formats import export_json

        segs = [(0.0, 2.5, "Hello")]
        result = export_json(segs)
        data = json.loads(result)
        self.assertEqual(len(data["segments"]), 1)
        self.assertEqual(data["segments"][0]["text"], "Hello")
        self.assertAlmostEqual(data["segments"][0]["start"], 0.0)

    def test_json_with_metadata(self):
        from utils.export_formats import export_json

        segs = [(0.0, 1.0, "Test")]
        result = export_json(segs, metadata={"backend": "whisper", "language": "en"})
        data = json.loads(result)
        self.assertEqual(data["metadata"]["backend"], "whisper")

    def test_json_unicode(self):
        from utils.export_formats import export_json

        segs = [(0.0, 1.0, "日本語テスト")]
        result = export_json(segs)
        self.assertIn("日本語テスト", result)

    def test_json_empty(self):
        from utils.export_formats import export_json

        data = json.loads(export_json([]))
        self.assertEqual(data["segments"], [])


class TestExportCSV(unittest.TestCase):
    def test_basic_csv(self):
        from utils.export_formats import export_csv

        segs = [(0.0, 2.5, "Hello world")]
        result = export_csv(segs)
        lines = result.strip().splitlines()
        self.assertEqual(lines[0].strip(), "start,end,text")
        self.assertIn("0.000", lines[1])
        self.assertIn("Hello world", lines[1])

    def test_csv_with_commas(self):
        from utils.export_formats import export_csv

        segs = [(0.0, 1.0, 'Text with "quotes" and, commas')]
        result = export_csv(segs)
        # CSV should properly quote fields with commas
        self.assertIn('"', result)

    def test_csv_empty(self):
        from utils.export_formats import export_csv

        result = export_csv([])
        self.assertIn("start,end,text", result)


class TestExportTXT(unittest.TestCase):
    def test_basic_txt(self):
        from utils.export_formats import export_txt

        segs = [(0.0, 2.5, "Hello"), (2.5, 5.0, "World")]
        result = export_txt(segs)
        self.assertEqual(result, "Hello\nWorld")

    def test_txt_empty(self):
        from utils.export_formats import export_txt

        self.assertEqual(export_txt([]), "")


class TestExportFormatsRegistry(unittest.TestCase):
    def test_all_formats_registered(self):
        from utils.export_formats import EXPORT_FORMATS

        for fmt in ("TXT", "SRT", "VTT", "JSON", "CSV"):
            self.assertIn(fmt, EXPORT_FORMATS)
            ext, fn = EXPORT_FORMATS[fmt]
            self.assertTrue(ext.startswith("."))
            self.assertTrue(callable(fn))

    def test_all_formats_work_on_same_input(self):
        from utils.export_formats import EXPORT_FORMATS

        segs = [(0.0, 1.5, "Test segment"), (1.5, 3.0, "Another one")]
        for fmt, (ext, fn) in EXPORT_FORMATS.items():
            result = fn(segs)
            self.assertIsInstance(result, str, f"{fmt} did not return str")
            self.assertGreater(len(result), 0, f"{fmt} returned empty")


if __name__ == "__main__":
    unittest.main()
