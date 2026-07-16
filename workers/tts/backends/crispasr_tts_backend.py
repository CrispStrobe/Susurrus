# workers/tts/backends/crispasr_tts_backend.py
"""CrispASR-based TTS backend — wraps the crispasr binary with --tts."""

import logging
import os
import subprocess

from .base import TTSBackend


class CrispasrTTSBackend(TTSBackend):
    """TTS via the CrispASR binary.

    Supports kokoro, orpheus, qwen3-tts, chatterbox, vibevoice, indextts,
    voxcpm2-tts, melotts, piper, bark, dia, zonos, csm, mini-omni2,
    lfm2-audio, and more engines depending on the model loaded.

    Kwargs:
        crispasr_backend: str — force a TTS engine (e.g. "kokoro")
        voice: str — voice file or preset name
        ref_text: str — reference text for voice cloning
        ref_asr: str — ASR backend for auto-transcribing reference audio
        instruct: str — natural-language voice description (qwen3-tts)
        codec_model: str — codec/companion GGUF
        tts_steps: int — diffusion/CFM steps
        g2p_dict: str — G2P dictionary ('olaph', 'open-dict', or path)
        auto_download: bool — auto-download model
        i_have_rights: bool — attest voice-cloning consent (required for .wav clone)
        no_spoken_disclaimer: bool — skip the audible AI-disclosure prefix
        watermark_model: str — AudioSeal GGUF for neural watermarking
        tts_play: bool — play audio on local speaker after synthesis
        tts_play_device: int — audio device index for local playback
    """

    def __init__(self, model_id=None, device="cpu", language=None, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.crispasr_backend = kwargs.get("crispasr_backend")
        self.voice = kwargs.get("voice")
        self.ref_text = kwargs.get("ref_text")
        self.ref_asr = kwargs.get("ref_asr")
        self.instruct = kwargs.get("instruct")
        self.codec_model = kwargs.get("codec_model")
        self.tts_steps = kwargs.get("tts_steps")
        self.g2p_dict = kwargs.get("g2p_dict")
        self.auto_download = kwargs.get("auto_download", True)
        self.i_have_rights = kwargs.get("i_have_rights", False)
        self.no_spoken_disclaimer = kwargs.get("no_spoken_disclaimer", False)
        self.watermark_model = kwargs.get("watermark_model")
        self.cache_dir = kwargs.get("cache_dir")
        self.tts_play = kwargs.get("tts_play", False)
        self.tts_play_device = kwargs.get("tts_play_device")
        self.no_watermark = kwargs.get("no_watermark", False)
        self.c2pa_cert = kwargs.get("c2pa_cert")
        self.c2pa_key = kwargs.get("c2pa_key")

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
        if self.ref_asr:
            cmd.extend(["--ref-asr", self.ref_asr])
        if self.instruct:
            cmd.extend(["--instruct", self.instruct])
        if self.codec_model:
            cmd.extend(["--codec-model", self.codec_model])
        if self.tts_steps is not None:
            cmd.extend(["--tts-steps", str(self.tts_steps)])
        if self.g2p_dict:
            cmd.extend(["--g2p-dict", self.g2p_dict])
        if self.watermark_model:
            cmd.extend(["--watermark-model", self.watermark_model])
        # Provenance / EU AI Act controls for voice cloning
        # Watermark + C2PA signing are ON by default in CrispASR.
        # Only --no-watermark opts out (with operator responsibility shift).
        if self.i_have_rights:
            cmd.append("--i-have-rights")
        if self.no_spoken_disclaimer:
            cmd.append("--no-spoken-disclaimer")
        if self.no_watermark:
            cmd.append("--no-watermark")
            logging.warning(
                "Watermarking disabled. AI-content marking responsibility "
                "rests with the operator per EU AI Act Art. 50."
            )
        if self.c2pa_cert:
            cmd.extend(["--c2pa-cert", self.c2pa_cert])
        if self.c2pa_key:
            cmd.extend(["--c2pa-key", self.c2pa_key])
        if self.auto_download:
            cmd.append("--auto-download")
        if self.cache_dir:
            cmd.extend(["--cache-dir", self.cache_dir])
        if self.tts_play:
            cmd.append("--tts-play")
        if self.tts_play_device is not None:
            cmd.extend(["--tts-play-device", str(self.tts_play_device)])
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
