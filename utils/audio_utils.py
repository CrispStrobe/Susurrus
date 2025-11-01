# utils/audio_utils.py:
"""Audio processing utilities"""
import logging
import os
import tempfile

from pydub import AudioSegment


def convert_audio_to_wav(audio_path):
    """Convert audio to 16-bit 16kHz WAV format required by most speech recognition models.
    Supports a wide range of formats including mp3, m4a, aac, ogg, flac, etc."""
    try:
        from pydub import AudioSegment
    except ImportError:
        logging.error("pydub is not installed. Please install it with 'pip install pydub'")
        return audio_path

    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        # Check if file is already WAV with correct parameters
        if audio_path.lower().endswith(".wav"):
            try:
                audio = AudioSegment.from_wav(audio_path)
                if audio.channels == 1 and audio.sample_width == 2 and audio.frame_rate == 16000:
                    logging.info("Audio is already in the correct format")
                    return audio_path
            except Exception as e:
                logging.warning(f"Unable to verify WAV format: {e}. Will convert anyway.")
                # If check fails, proceed with conversion
                pass

        # Detect audio format and convert
        audio_format = detect_audio_format(audio_path)
        if not audio_format:
            logging.warning(
                "Unable to detect audio format. Attempting to convert based on file extension."
            )
            # Use extension as fallback
            file_ext = os.path.splitext(audio_path)[1][1:].lower()

            # Map common extensions to pydub formats
            format_map = {
                "mp3": "mp3",
                "m4a": "mp4",
                "aac": "aac",
                "ogg": "ogg",
                "oga": "ogg",
                "flac": "flac",
                "wav": "wav",
                "webm": "webm",
                "mp4": "mp4",
                "wma": "asf",
                "opus": "opus",
            }

            audio_format = format_map.get(file_ext, file_ext)
            logging.info(f"Using format '{audio_format}' based on file extension '.{file_ext}'")

        # Load audio with appropriate format
        if audio_format == "mp4" or audio_format == "m4a" or audio_path.lower().endswith(".m4a"):
            # Special handling for m4a/mp4 files
            logging.info("Loading m4a/mp4 format audio")
            sound = AudioSegment.from_file(audio_path, format="mp4")
        else:
            sound = AudioSegment.from_file(audio_path, format=audio_format)

        # Convert to required format
        sound = sound.set_channels(1)  # Mono
        sound = sound.set_sample_width(2)  # 16-bit
        sound = sound.set_frame_rate(16000)  # 16kHz

        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav_file.name
        temp_wav_file.close()

        sound.export(temp_wav_path, format="wav")
        logging.info(f"Converted audio to 16-bit 16kHz WAV: {temp_wav_path}")
        return temp_wav_path
    except Exception as e:
        logging.error(f"Error converting audio to WAV: {str(e)}")
        return audio_path  # Return original file on error


def detect_audio_format(audio_path):
    """Detect the format of an audio file with improved format detection.
    Returns a string format name that can be used with pydub."""
    try:
        from pydub.utils import mediainfo
    except ImportError:
        logging.error("pydub is not installed. Please install it with 'pip install pydub'")
        return None

    file_ext = os.path.splitext(audio_path)[1][1:].lower()

    # Map of common file extensions to pydub format names
    format_map = {
        "mp3": "mp3",
        "m4a": "mp4",
        "aac": "aac",
        "ogg": "ogg",
        "oga": "ogg",
        "flac": "flac",
        "wav": "wav",
        "webm": "webm",
        "mp4": "mp4",
        "wma": "asf",
        "opus": "opus",
    }

    try:
        # First try using mediainfo
        info = mediainfo(audio_path)
        format_name = info.get("format_name", "")

        # For m4a files, mediainfo might return "mov,mp4,m4a,3gp,3g2,mj2" or similar
        if "m4a" in format_name or "mp4" in format_name:
            logging.info(f"Detected m4a/mp4 format: {format_name}")
            return "mp4"  # Use mp4 format for m4a files

        # Handle specific format names returned by mediainfo
        if "mp3" in format_name:
            return "mp3"
        elif "ogg" in format_name:
            return "ogg"
        elif "flac" in format_name:
            return "flac"
        elif "wav" in format_name:
            return "wav"

        # If we couldn't identify format from mediainfo output, try using the file extension
        if file_ext in format_map:
            logging.info(f"Using format from file extension: {format_map[file_ext]}")
            return format_map[file_ext]

        logging.warning(f"Couldn't determine format from mediainfo output: {format_name}")
        return format_name.split(",")[0]  # Use the first format if it's a list

    except Exception as e:
        logging.error(f"Error detecting audio format: {str(e)}")

        # Fallback to using the file extension
        if file_ext in format_map:
            logging.info(f"Falling back to format from file extension: {format_map[file_ext]}")
            return format_map[file_ext]

        return None


def trim_audio(audio_path, start_time, end_time):
    """Trim audio to specified start and end times."""
    try:
        from pydub import AudioSegment
    except ImportError:
        logging.error("pydub is not installed. Please install it with 'pip install pydub'")
        return audio_path

    if not start_time and not end_time:
        logging.info("No trimming required, using the original audio file.")
        return audio_path

    try:
        logging.info(f"Trimming audio from {start_time}s to {end_time}s")
        audio = AudioSegment.from_file(audio_path)
        audio_duration = len(audio) / 1000  # in seconds

        start_time = float(start_time) if start_time else 0
        end_time = float(end_time) if end_time else audio_duration

        start_time = max(0, start_time)
        end_time = min(audio_duration, end_time)

        if start_time >= end_time:
            logging.warning("Invalid trimming times, end time must be greater than start time.")
            return audio_path

        trimmed_audio = audio[int(start_time * 1000) : int(end_time * 1000)]
        temp_trimmed = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        trimmed_audio.export(temp_trimmed, format="wav")
        logging.info(f"Trimmed audio saved to: {temp_trimmed}")
        return temp_trimmed
    except Exception as e:
        logging.error(f"Error trimming audio: {str(e)}")
        return audio_path  # Return original file on error


def is_valid_time(time_value):
    """Check if a time value is valid for audio trimming."""
    if time_value is None or time_value == "":
        return False
    try:
        return float(time_value) >= 0
    except (ValueError, TypeError):
        return False
