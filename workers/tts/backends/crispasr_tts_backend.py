# workers/tts/backends/crispasr_tts_backend.py
"""CrispASR-based TTS backend — wraps the crispasr binary with --tts."""

import logging
import os
import subprocess

from .base import TTSBackend


class CrispasrTTSBackend(TTSBackend):
    """TTS via the CrispASR binary.

    Supports kokoro, orpheus, qwen3-tts, chatterbox, vibevoice-tts,
    indextts, and voxcpm2-tts engines depending on the model loaded.

    Kwargs:
        crispasr_backend: str — force a TTS engine (e.g. "kokoro")
        voice: str — voice file or preset name
        ref_text: str — reference text for voice cloning
        instruct: str — natural-language voice description (qwen3-tts)
        codec_model: str — codec/companion GGUF
        tts_steps: int — diffusion/CFM steps
        auto_download: bool — auto-download model
    """

    def __init__(self, model_id=None, device="cpu", language=None, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.crispasr_backend = kwargs.get("crispasr_backend")
        self.voice = kwargs.get("voice")
        self.ref_text = kwargs.get("ref_text")
        self.instruct = kwargs.get("instruct")
        self.codec_model = kwargs.get("codec_model")
        self.tts_steps = kwargs.get("tts_steps")
        self.auto_download = kwargs.get("auto_download", True)

    def synthesize(self, text, output_path="tts_output.wav", voice=None):
        from utils.crispasr_utils import find_crispasr

        exe = find_crispasr()
        if not exe:
            raise FileNotFoundError(
                "crispasr binary not found. Set CRISPASR_EXECUTABLE or "
                "install from https://github.com/CrispStrobe/CrispASR"
            )

        model = self.model_id or "auto"
        cmd = [exe, "-m", model, "--tts", text, "--tts-output", output_path]
        cmd.extend(["-t", str(min(os.cpu_count() or 4, 8))])

        if self.crispasr_backend:
            cmd.extend(["--backend", self.crispasr_backend])

        voice_to_use = voice or self.voice
        if voice_to_use:
            cmd.extend(["--voice", voice_to_use])
        if self.ref_text:
            cmd.extend(["--ref-text", self.ref_text])
        if self.instruct:
            cmd.extend(["--instruct", self.instruct])
        if self.codec_model:
            cmd.extend(["--codec-model", self.codec_model])
        if self.tts_steps is not None:
            cmd.extend(["--tts-steps", str(self.tts_steps)])
        if self.auto_download:
            cmd.append("--auto-download")
        if self.language:
            cmd.extend(["-l", self.language])

        logging.info(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if stderr:
            for line in stderr.strip().splitlines():
                logging.info(f"crispasr: {line}")

        if process.returncode != 0:
            raise RuntimeError(f"CrispASR TTS failed (code {process.returncode}): {stderr}")

        if os.path.isfile(output_path):
            logging.info(f"TTS output: {output_path}")
            return output_path

        raise FileNotFoundError(f"TTS output not created: {output_path}")

    def list_voices(self):
        from config import TTS_BACKEND_MAP

        key = f"crispasr:{self.crispasr_backend}" if self.crispasr_backend else "crispasr:kokoro"
        entry = TTS_BACKEND_MAP.get(key, {})
        return entry.get("voices", [])
