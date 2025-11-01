"""Test transcription backends"""

import unittest

from workers.backends import BackendFactory


class TestBackends(unittest.TestCase):
    def test_list_backends(self):
        backends = BackendFactory.list_backends()
        self.assertIsInstance(backends, list)
        self.assertIn("transformers", backends)

    def test_create_backend(self):
        backend = BackendFactory.create_backend("transformers", "openai/whisper-tiny", device="cpu")
        self.assertIsNotNone(backend)


if __name__ == "__main__":
    unittest.main()
