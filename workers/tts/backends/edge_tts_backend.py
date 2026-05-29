# workers/tts/backends/edge_tts_backend.py
"""Edge TTS backend — Microsoft Edge cloud TTS (no model download required).

License: edge-tts is MIT-licensed. The service is free but subject to
Microsoft's Terms of Service.
"""

import logging
import os
import tempfile

from .base import TTSBackend


class EdgeTTSBackend(TTSBackend):
    """Cloud-based TTS using Microsoft Edge's speech service.

    Requires: ``pip install edge-tts``
    """

    def synthesize(self, text, output_path="tts_output.wav", voice=None):
        try:
            import edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts is required for the Edge TTS backend. "
                "Install with: pip install edge-tts"
            )

        import asyncio

        voice_id = voice or self.kwargs.get("voice") or "de-DE-KatjaNeural"

        async def _synth():
            communicate = edge_tts.Communicate(text, voice_id)
            # Edge TTS outputs MP3; save to temp then convert
            mp3_path = output_path
            if output_path.endswith(".wav"):
                mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
                os.close(mp3_fd)

            await communicate.save(mp3_path)

            if output_path.endswith(".wav") and mp3_path != output_path:
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(mp3_path)
                    audio.export(output_path, format="wav")
                finally:
                    if os.path.isfile(mp3_path):
                        os.remove(mp3_path)
            return output_path

        # Run async synthesis
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, _synth()).result()
        else:
            result = asyncio.run(_synth())

        logging.info(f"Edge TTS output: {result}")
        return result

    def list_voices(self):
        return [
            "de-DE-KatjaNeural", "de-DE-ConradNeural", "de-DE-AmalaNeural",
            "de-DE-BerndNeural", "de-DE-ChristophNeural", "de-DE-ElkeNeural",
            "de-DE-GiselaNeural", "de-DE-KasperNeural",
            "en-US-JennyNeural", "en-US-GuyNeural",
            "en-GB-SoniaNeural", "en-GB-RyanNeural",
            "fr-FR-DeniseNeural", "fr-FR-HenriNeural",
            "es-ES-ElviraNeural", "es-ES-AlvaroNeural",
        ]
