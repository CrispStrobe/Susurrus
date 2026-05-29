# workers/translation/backends/crispasr_translation_backend.py
"""CrispASR-based translation backend — wraps the binary with --text flag."""

import logging
import os
import subprocess

from .base import TranslationBackend


class CrispasrTranslationBackend(TranslationBackend):
    """Translation via the CrispASR binary.

    Supports m2m100 (100 languages), madlad (419 languages), and
    gemma4-e2b (dual ASR+MT, 140+ languages).

    Kwargs:
        crispasr_backend: str — force translation engine (default: m2m100)
        auto_download: bool — auto-download model (default: True)
        translate_max_tokens: int — max output tokens
    """

    def __init__(self, model_id=None, device="cpu", **kwargs):
        super().__init__(model_id, device, **kwargs)
        self.crispasr_backend = kwargs.get("crispasr_backend", "m2m100")
        self.auto_download = kwargs.get("auto_download", True)
        self.translate_max_tokens = kwargs.get("translate_max_tokens")

    def translate(self, text, source_lang="en", target_lang="de"):
        from utils.crispasr_utils import find_crispasr

        exe = find_crispasr()
        if not exe:
            raise FileNotFoundError(
                "crispasr binary not found. Set CRISPASR_EXECUTABLE or "
                "install from https://github.com/CrispStrobe/CrispASR"
            )

        model = self.model_id or "auto"
        cmd = [
            exe,
            "-m", model,
            "--backend", self.crispasr_backend,
            "--text", text,
            "--tr-sl", source_lang,
            "--tr-tl", target_lang,
            "-t", str(min(os.cpu_count() or 4, 8)),
        ]

        if self.auto_download:
            cmd.append("--auto-download")
        if self.translate_max_tokens is not None:
            cmd.extend(["--translate-max-tokens", str(self.translate_max_tokens)])

        logging.info(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()

        if stderr:
            for line in stderr.strip().splitlines():
                logging.info(f"crispasr: {line}")

        if process.returncode != 0:
            raise RuntimeError(
                f"CrispASR translation failed (code {process.returncode}): {stderr}"
            )

        return stdout.strip()

    def list_languages(self):
        if self.crispasr_backend == "madlad":
            return ["419+ languages — see MadLad documentation"]
        return [
            "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br",
            "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es",
            "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu",
            "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo",
            "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb",
            "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr",
            "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl",
            "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq",
            "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr",
            "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu",
        ]
