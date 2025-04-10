#!/usr/bin/env python3
# transcribe_worker.py - Worker script for audio transcription

import argparse
import os
import logging
import subprocess
import time
import json
import platform
import threading
import tempfile
import shutil
from pathlib import Path
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Device fallbacks for incompatible backend+device combinations
device_fallbacks = {
    'faster-batched': [('mps', 'cpu')],
    'faster-whisper': [('mps', 'cpu')],
    'openai-whisper': [('mps', 'cpu')],
    'whisper-jax': [('mps', 'cpu')],
    'ctranslate2': [('mps', 'cpu'), ('cuda', 'cpu')],
}

def get_supported_device(backend, device):
    """Get a supported device for the backend, falling back if necessary."""
    if backend in device_fallbacks:
        for unsupported, fallback in device_fallbacks[backend]:
            if device.lower() == unsupported.lower():
                logging.warning(f"Device '{device}' is not supported by backend '{backend}'. Switching to '{fallback}'.")
                return fallback
    return device

class MyLogger:
    """Silent logger for yt-dlp and other libraries."""
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        logging.error(msg)

def my_hook(d):
    """Progress hook for yt-dlp to suppress progress messages."""
    pass

def get_original_model_id(model_id):
    """
    Get the original model ID from a variant model ID.
    Used to find the original OpenAI Whisper model corresponding to custom variants.
    """
    known_models = {
        'mlx-community/whisper-large-v3-turbo': 'openai/whisper-large-v3-turbo',
        'mlx-community/whisper-large-v3-turbo-q4': 'openai/whisper-large-v3-turbo',
        'mlx-community/whisper-tiny-mlx-4bit': 'openai/whisper-tiny',
        'mlx-community/whisper-base-mlx-4bit': 'openai/whisper-base',
        'mlx-community/whisper-small-mlx-q4': 'openai/whisper-small',
        'mlx-community/whisper-medium-mlx-4bit': 'openai/whisper-medium',
        'mlx-community/whisper-large-v3-mlx-4bit': 'openai/whisper-large-v3',
        'mlx-community/whisper-large-v3-mlx': 'openai/whisper-large-v3',
        'cstr/whisper-large-v3-turbo-int8_float32': 'openai/whisper-large-v3-turbo',
        'SYSTRAN/faster-whisper-large-v1': 'openai/whisper-large-v2',
        'GalaktischeGurke/primeline-whisper-large-v3-german-ct2': 'openai/whisper-large-v3'
    }

    if model_id in known_models:
        return known_models[model_id]

    model_id_lower = model_id.lower()

    if model_id.startswith("openai/whisper-"):
        return model_id
        
    if "endpoint" in model_id_lower:
        return "openai/whisper-large-v2"

    if "v3_turbo" in model_id_lower or "v3-turbo" in model_id_lower:
        base = "openai/whisper-large-v3-turbo"
    elif "v3" in model_id_lower:
        base = "openai/whisper-large-v3"
    elif "v2" in model_id_lower:
        base = "openai/whisper-large-v2"
    elif "v1" in model_id_lower:
        base = "openai/whisper-large-v1"
    else:
        base = "openai/whisper"

    if base == "openai/whisper":
        if "large" in model_id_lower:
            size = "large"
        elif "medium" in model_id_lower:
            size = "medium"
        elif "small" in model_id_lower:
            size = "small"
        elif "base" in model_id_lower:
            size = "base"
        elif "tiny" in model_id_lower:
            size = "tiny"
        else:
            size = "large"
        base = f"{base}-{size}"

    if "_en" in model_id_lower or ".en" in model_id_lower:
        lang = ".en"
    else:
        lang = ""

    return f"{base}{lang}"

def download_with_yt_dlp(url, proxies=None):
    """Download audio using yt-dlp with proper error handling."""
    logging.info("Downloading using yt-dlp...")
    import tempfile
    try:
        import yt_dlp
    except ImportError:
        logging.error("yt_dlp is not installed. Please install it to download from YouTube.")
        return None

    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, '%(id)s.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',  # Use WAV for better compatibility
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'logger': MyLogger(),
            'progress_hooks': [my_hook],
        }
        if proxies and proxies.get('http'):
            ydl_opts['proxy'] = proxies['http']

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if 'entries' in info:
                info = info['entries'][0]
            output_file = ydl.prepare_filename(info)
            output_file = os.path.splitext(output_file)[0] + '.wav'
            if os.path.exists(output_file):
                logging.info(f"Downloaded audio: {output_file}")
                return output_file
            else:
                raise Exception("yt-dlp did not produce an output file.")

    except Exception as e:
        logging.error(f"yt_dlp download failed: {str(e)}")
        # Clean up temp directory if something goes wrong
        if 'temp_dir' in locals():
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        return None

def download_with_pytube(url, proxies=None):
    """Download audio using pytube as a fallback option."""
    logging.info("Downloading using pytube...")
    import tempfile
    try:
        from pytube import YouTube
    except ImportError:
        logging.error("pytube package is not installed. Cannot use this method.")
        return None

    temp_dir = None
    try:
        yt = YouTube(url)
        if proxies and proxies.get('http'):
            yt.proxies = proxies
        
        audio_stream = yt.streams.filter(only_audio=True).first()
        if audio_stream is None:
            raise Exception("No audio streams available.")
        
        temp_dir = tempfile.mkdtemp()
        out_file = audio_stream.download(output_path=temp_dir)
        base, ext = os.path.splitext(out_file)
        new_file = base + '.wav'
        
        # Convert to WAV using ffmpeg
        try:
            subprocess.run([
                'ffmpeg', '-i', out_file, '-acodec', 'pcm_s16le', 
                '-ar', '44100', '-ac', '2', '-y', new_file
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.remove(out_file)  # Remove original file
        except (subprocess.SubprocessError, FileNotFoundError):
            # If ffmpeg fails or isn't available, just rename
            os.rename(out_file, new_file)
        
        if os.path.exists(new_file) and os.path.getsize(new_file) > 0:
            logging.info(f"Downloaded and converted audio to: {new_file}")
            return new_file
        else:
            raise Exception("Failed to produce a valid output file")

    except Exception as e:
        logging.error(f"pytube download failed: {str(e)}")
        # Clean up temp directory if something goes wrong
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        return None

def download_with_ffmpeg(url, proxies=None, ffmpeg_path='ffmpeg'):
    """Download audio using ffmpeg as a final fallback option."""
    logging.info("Downloading using ffmpeg...")
    
    if shutil.which(ffmpeg_path) is None:
        logging.error(f"{ffmpeg_path} not found in PATH. Please install it for direct downloads.")
        return None

    output_file = None
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        output_file = temp_file.name
        temp_file.close()

        command = [ffmpeg_path, '-i', url, 
                   '-acodec', 'pcm_s16le',  # Use standard PCM format
                   '-ar', '44100',          # 44.1kHz sample rate
                   '-ac', '2',              # Stereo
                   '-y',                    # Overwrite output file
                   output_file]
        env = os.environ.copy()
        
        # Handle proxy settings
        if proxies:
            if proxies.get('http') and 'socks5' in proxies['http']:
                env['ALL_PROXY'] = proxies['http']
            else:
                if proxies.get('http'):
                    env['http_proxy'] = proxies['http']
                if proxies.get('https'):
                    env['https_proxy'] = proxies['https']

        # Run ffmpeg
        process = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        if process.returncode != 0:
            error_msg = process.stderr.decode('utf-8', errors='replace')
            logging.error(f"FFmpeg error: {error_msg}")
            if os.path.exists(output_file):
                os.unlink(output_file)
            return None

        # Verify the output file
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logging.info(f"Downloaded audio using ffmpeg: {output_file}")
            return output_file
        else:
            if os.path.exists(output_file):
                os.unlink(output_file)
            logging.error("FFmpeg produced an empty file")
            return None

    except Exception as e:
        logging.error(f"FFmpeg download failed: {str(e)}")
        # Clean up the temporary file if something goes wrong
        if output_file and os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except Exception:
                pass
        return None

def download_audio(url, proxies=None, ffmpeg_path='ffmpeg'):
    """
    Download audio from URL using multiple fallback methods.
    Returns the path to the downloaded file or None on failure.
    """
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    # Clean up the URL and remove any playlist parameters
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'list' in query_params:
        query_params.pop('list', None)
        new_query = urlencode(query_params, doseq=True)
        parsed_url = parsed_url._replace(query=new_query)
        url = urlunparse(parsed_url)
        logging.info(f"Modified URL to remove playlist parameter: {url}")

    logging.info(f"Downloading audio from URL: {url}")

    # Determine if it's a YouTube URL
    is_youtube = any(domain in parsed_url.netloc for domain in ['youtube.com', 'youtu.be'])

    # Define download methods based on URL type
    if is_youtube:
        download_methods = [
            download_with_yt_dlp,
            download_with_pytube
        ]
    else:
        download_methods = [
            lambda u, p: download_with_ffmpeg(u, p, ffmpeg_path)
        ]

    # Try each download method in sequence
    for download_method in download_methods:
        try:
            audio_file = download_method(url, proxies)
            
            if audio_file and os.path.exists(audio_file):
                # Validate the downloaded file
                if os.path.getsize(audio_file) > 0:
                    try:
                        # Try to open the file to ensure it's valid
                        with open(audio_file, 'rb') as f:
                            # Read first few bytes to verify it's readable
                            f.read(1024)
                        
                        logging.info(f"Successfully downloaded audio using {download_method.__name__}")
                        return audio_file
                    except Exception as e:
                        logging.error(f"Downloaded file is corrupted: {str(e)}")
                        try:
                            os.remove(audio_file)
                        except OSError:
                            pass
                else:
                    logging.error(f"Downloaded file is empty: {audio_file}")
                    try:
                        os.remove(audio_file)
                    except OSError:
                        pass
            
        except Exception as e:
            logging.error(f"{download_method.__name__} failed: {str(e)}")
            continue

    logging.error("All download methods failed")
    return None

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
        'mp3': 'mp3',
        'm4a': 'mp4',
        'aac': 'aac',
        'ogg': 'ogg',
        'oga': 'ogg',
        'flac': 'flac',
        'wav': 'wav',
        'webm': 'webm',
        'mp4': 'mp4',
        'wma': 'asf',
        'opus': 'opus'
    }
    
    try:
        # First try using mediainfo
        info = mediainfo(audio_path)
        format_name = info.get('format_name', '')
        
        # For m4a files, mediainfo might return "mov,mp4,m4a,3gp,3g2,mj2" or similar
        if 'm4a' in format_name or 'mp4' in format_name:
            logging.info(f"Detected m4a/mp4 format: {format_name}")
            return 'mp4'  # Use mp4 format for m4a files
            
        # Handle specific format names returned by mediainfo
        if 'mp3' in format_name:
            return 'mp3'
        elif 'ogg' in format_name:
            return 'ogg'
        elif 'flac' in format_name:
            return 'flac'
        elif 'wav' in format_name:
            return 'wav'
        
        # If we couldn't identify format from mediainfo output, try using the file extension
        if file_ext in format_map:
            logging.info(f"Using format from file extension: {format_map[file_ext]}")
            return format_map[file_ext]
            
        logging.warning(f"Couldn't determine format from mediainfo output: {format_name}")
        return format_name.split(',')[0]  # Use the first format if it's a list
        
    except Exception as e:
        logging.error(f"Error detecting audio format: {str(e)}")
        
        # Fallback to using the file extension
        if file_ext in format_map:
            logging.info(f"Falling back to format from file extension: {format_map[file_ext]}")
            return format_map[file_ext]
        
        return None

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
        if audio_path.lower().endswith('.wav'):
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
            logging.warning("Unable to detect audio format. Attempting to convert based on file extension.")
            # Use extension as fallback
            file_ext = os.path.splitext(audio_path)[1][1:].lower()
            
            # Map common extensions to pydub formats
            format_map = {
                'mp3': 'mp3',
                'm4a': 'mp4',
                'aac': 'aac',
                'ogg': 'ogg',
                'oga': 'ogg',
                'flac': 'flac',
                'wav': 'wav',
                'webm': 'webm',
                'mp4': 'mp4',
                'wma': 'asf',
                'opus': 'opus'
            }
            
            audio_format = format_map.get(file_ext, file_ext)
            logging.info(f"Using format '{audio_format}' based on file extension '.{file_ext}'")
        
        # Load audio with appropriate format
        if audio_format == 'mp4' or audio_format == 'm4a' or audio_path.lower().endswith('.m4a'):
            # Special handling for m4a/mp4 files
            logging.info("Loading m4a/mp4 format audio")
            sound = AudioSegment.from_file(audio_path, format="mp4")
        else:
            sound = AudioSegment.from_file(audio_path, format=audio_format)
        
        # Convert to required format
        sound = sound.set_channels(1)  # Mono
        sound = sound.set_sample_width(2)  # 16-bit
        sound = sound.set_frame_rate(16000)  # 16kHz
        
        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_wav_path = temp_wav_file.name
        temp_wav_file.close()
        
        sound.export(temp_wav_path, format='wav')
        logging.info(f"Converted audio to 16-bit 16kHz WAV: {temp_wav_path}")
        return temp_wav_path
    except Exception as e:
        logging.error(f"Error converting audio to WAV: {str(e)}")
        return audio_path  # Return original file on error

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

        trimmed_audio = audio[int(start_time * 1000):int(end_time * 1000)]
        temp_trimmed = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        trimmed_audio.export(temp_trimmed, format="wav")
        logging.info(f"Trimmed audio saved to: {temp_trimmed}")
        return temp_trimmed
    except Exception as e:
        logging.error(f"Error trimming audio: {str(e)}")
        return audio_path  # Return original file on error

def is_valid_time(time_value):
    """Check if a time value is valid for audio trimming."""
    if time_value is None or time_value == '':
        return False
    try:
        return float(time_value) >= 0
    except (ValueError, TypeError):
        return False

def find_whisper_cpp_executable():
    """Find the whisper.cpp executable with fallbacks to common locations."""
    # Check common paths first
    system = platform.system()
    executable_names = ["whisper", "whisper.exe"] if system == "Windows" else ["whisper"]
    
    # Add additional names for different builds
    if system == "Windows":
        executable_names.extend(["main.exe", "whisper-cli.exe"])
    else:
        executable_names.extend(["main", "whisper-cli"])
    
    # Define search paths
    search_paths = [
        os.getcwd(),
        os.path.join(os.getcwd(), "whisper.cpp"),
        os.path.join(os.getcwd(), "bin"),
        os.path.join(os.getcwd(), "build"),
        os.path.expanduser("~"),
        os.path.join(os.path.expanduser("~"), "whisper.cpp"),
        os.path.join(os.path.expanduser("~"), "bin"),
    ]
    
    # Add platform-specific paths
    if system == "Windows":
        search_paths.extend([
            os.path.join(os.getcwd(), "build", "Release"),
            os.path.join(os.getcwd(), "build", "Debug"),
            "C:\\Program Files\\whisper.cpp",
            "C:\\Program Files (x86)\\whisper.cpp",
        ])
    elif system == "Darwin" or system == "Linux":
        search_paths.extend([
            "/usr/local/bin",
            "/usr/bin",
            "/opt/whisper.cpp/bin",
        ])
    
    # Search for executable
    for path in search_paths:
        if os.path.exists(path):
            for exe_name in executable_names:
                exe_path = os.path.join(path, exe_name)
                if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                    logging.info(f"Found whisper.cpp executable at: {exe_path}")
                    return exe_path
    
    logging.error("No whisper.cpp executable found in search paths")
    return None

def find_or_download_whisper_cpp_model(model_id):
    """Find or download a whisper.cpp model."""
    model_file = model_id
    if not model_file.startswith('ggml-'):
        model_file = f'ggml-{model_file}'
    if not model_file.endswith('.bin'):
        model_file = f'{model_file}.bin'

    # Search paths for existing model
    search_paths = [
        os.getcwd(),
        os.path.join(os.getcwd(), 'models'),
        os.path.expanduser('~'),
        os.path.join(os.path.expanduser('~'), 'models'),
        os.path.join(os.path.expanduser('~'), 'whisper.cpp', 'models'),
    ]

    # Look for existing model
    for path in search_paths:
        if os.path.exists(path):
            model_path = os.path.join(path, model_file)
            if os.path.exists(model_path):
                logging.info(f"Model file '{model_file}' found at: {model_path}")
                return model_path

    # Create models directory if needed
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_file)
    
    # Download model if not found
    try:
        # Try direct download using requests
        import requests
        base_url = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/'
        url = base_url + model_file
        logging.info(f'Downloading {model_file} from {url}')
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f'Model downloaded to {model_path}')
        return model_path
    except Exception as e:
        logging.error(f"Failed to download model: {str(e)}")
        if os.path.exists(model_path) and os.path.getsize(model_path) == 0:
            os.remove(model_path)
        raise

def main():
    # Track temporary files for cleanup
    temp_files = []
    
    try:
        parser = argparse.ArgumentParser(description='Susurrus Transcription worker script')
        parser.add_argument('--audio-input', help='Path to the audio input file')
        parser.add_argument('--audio-url', help='URL to the audio file')
        parser.add_argument('--model-id', type=str, required=True, help='Model ID to use')
        parser.add_argument('--word-timestamps', action='store_true', help='Enable word timestamps')
        parser.add_argument('--language', default=None, help='Language code')
        parser.add_argument('--backend', type=str, default='mlx-whisper', help='Backend to use')
        parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, gpu, mps)')
        parser.add_argument('--pipeline-type', default='default', help='Pipeline type')
        parser.add_argument('--max-chunk-length', type=float, default=0.0, help='Max chunk length in seconds')
        parser.add_argument('--output-format', default='txt', help='Output format for whisper.cpp')
        parser.add_argument('--quantization', default=None, help='Quantization type for ctranslate2')
        parser.add_argument('--preprocessor-path', type=str, default='', help='Path to preprocessor files')
        parser.add_argument('--original-model-id', type=str, default='', help='Original model ID')
        parser.add_argument('--start-time', type=str, default=None, help='Start time for audio trimming in seconds')
        parser.add_argument('--end-time', type=str, default=None, help='End time for audio trimming in seconds')
        parser.add_argument('--proxy-url', type=str, default=None, help='Proxy URL')
        parser.add_argument('--proxy-username', type=str, default=None, help='Proxy username')
        parser.add_argument('--proxy-password', type=str, default=None, help='Proxy password')
        parser.add_argument('--ffmpeg-path', type=str, default='ffmpeg', help='Path to ffmpeg executable')

        args = parser.parse_args()

        if not args.audio_input and not args.audio_url:
            parser.error("Either --audio-input or --audio-url must be provided")

        # Parse common arguments
        audio_input = args.audio_input
        audio_url = args.audio_url
        model_id = args.model_id
        word_timestamps = args.word_timestamps
        language = args.language
        backend = args.backend.lower()
        device_arg = args.device.lower()
        pipeline_type = args.pipeline_type
        max_chunk_length = args.max_chunk_length
        output_format = args.output_format
        quantization = args.quantization
        start_time = args.start_time
        end_time = args.end_time
        original_model_id = args.original_model_id

        # Set up proxy configuration
        proxies = None
        if args.proxy_url:
            proxies = {
                'http': args.proxy_url,
                'https': args.proxy_url
            }
            if args.proxy_username and args.proxy_password:
                # Include authentication in the proxy URL
                from urllib.parse import urlparse, urlunparse
                parsed_url = urlparse(args.proxy_url)
                netloc = f"{args.proxy_username}:{args.proxy_password}@{parsed_url.hostname}"
                if parsed_url.port:
                    netloc += f":{parsed_url.port}"
                proxy_url_with_auth = urlunparse((
                    parsed_url.scheme,
                    netloc,
                    parsed_url.path,
                    parsed_url.params,
                    parsed_url.query,
                    parsed_url.fragment
                ))
                proxies['http'] = proxy_url_with_auth
                proxies['https'] = proxy_url_with_auth

        # Clean up language parameter
        if language is not None and language.strip() == '':
            language = None

        # Determine device to use
        try:
            import torch
            if device_arg == 'auto':
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            elif device_arg == 'cpu':
                device = "cpu"
            elif device_arg == 'gpu':
                device = "cuda"
            elif device_arg == 'mps':
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            logging.warning("PyTorch not installed, defaulting to CPU")
            device = "cpu"

        # Apply device fallbacks for incompatible backends
        device = get_supported_device(backend, device)

        logging.info(f"Starting transcription on device: {device} using backend: {backend}")
        logging.info(f"Model: {model_id}")
        if language:
            logging.info(f"Language: {language}")
        if word_timestamps:
            logging.info("Word timestamps enabled")

        # Handle audio input
        working_audio_path = None
        
        if audio_input and os.path.exists(audio_input):
            # Using local file
            original_audio_path = audio_input
        elif audio_url:
            # Download from URL
            logging.info(f"Downloading audio from URL: {audio_url}")
            original_audio_path = download_audio(audio_url, proxies=proxies, ffmpeg_path=args.ffmpeg_path)
            if not original_audio_path:
                raise Exception(f"Failed to download audio from {audio_url}")
            temp_files.append(original_audio_path)
        else:
            raise Exception("No valid audio source provided.")

        # Trim audio if requested
        if is_valid_time(start_time) or is_valid_time(end_time):
            logging.info(f"Trimming audio from {start_time}s to {end_time}s")
            working_audio_path = trim_audio(original_audio_path, start_time, end_time)
            if working_audio_path != original_audio_path:
                temp_files.append(working_audio_path)
        else:
            working_audio_path = original_audio_path

        audio_input = working_audio_path
        start_time_perf = time.time()

        #
        # Backend-specific transcription code
        #
        if backend == 'mlx-whisper':
            try:
                import mlx_whisper
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError("mlx_whisper and huggingface_hub packages are required for mlx-whisper backend")

            logging.info(f"Using mlx-whisper with model {model_id}")
            
            # Make sure we're working with WAV format
            wav_path = convert_audio_to_wav(audio_input)
            if wav_path != audio_input:
                temp_files.append(wav_path)
                audio_input = wav_path

            # Download the model files from Hugging Face if needed
            try:
                model_path = snapshot_download(repo_id=model_id)
                logging.info(f"Downloaded model files to: {model_path}")
            except Exception as e:
                logging.error(f"Failed to download model files: {str(e)}")
                raise

            # Transcribe with MLX
            transcribe_options = {
                "path_or_hf_repo": model_id,
                "verbose": True,
                "word_timestamps": word_timestamps,
                "language": language,
            }
            
            try:
                result = mlx_whisper.transcribe(audio_input, **transcribe_options)
                
                # Process segments
                if 'segments' in result:
                    for segment in result['segments']:
                        text = segment['text'].strip()
                        start = segment['start']
                        end = segment['end']
                        print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
                else:
                    # Handle case with no segments
                    text = result.get('text', '').strip()
                    print(text, flush=True)
                
            except Exception as e:
                logging.error(f"Transcription failed: {str(e)}")
                raise
        
        elif backend == 'faster-batched':
            try:
                from faster_whisper import WhisperModel, BatchedInferencePipeline
            except ImportError:
                raise ImportError("faster_whisper package is required for faster-batched backend")

            compute_type = quantization if quantization else 'float16' if device == 'cuda' else 'int8'

            logging.info(f"Loading model {model_id} with compute_type={compute_type}")
            model = WhisperModel(model_id, device=device, compute_type=compute_type)
            pipeline = BatchedInferencePipeline(model=model)

            logging.info("Starting batched transcription")
            segments, info = pipeline.transcribe(
                audio_input, 
                batch_size=16,  # Reasonable default batch size
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=True,  # Filter out non-speech
            )

            if hasattr(info, 'language') and info.language:
                logging.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")

            for segment in segments:
                text = segment.text.strip()
                start = segment.start
                end = segment.end
                print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
                
                # Print word timestamps if requested
                if word_timestamps and segment.words:
                    for word in segment.words:
                        word_text = word.word.strip()
                        word_start = word.start
                        word_end = word.end
                        print(f'  [{word_start:.3f} --> {word_end:.3f}] {word_text}', flush=True)
        
        elif backend == 'faster-sequenced':
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError("faster_whisper package is required for faster-sequenced backend")

            # Determine compute type based on device
            if device == 'cuda':
                compute_type = "float16"
            else:
                compute_type = "int8"

            # Override with user-specified quantization if provided
            if quantization:
                compute_type = quantization

            logging.info(f"Loading model {model_id} with compute_type={compute_type}")
            model = WhisperModel(model_id, device=device, compute_type=compute_type)

            options = {
                "language": language,
                "beam_size": 5,
                "best_of": 5,
                "word_timestamps": word_timestamps,
                "vad_filter": True,  # Filter out non-speech
            }

            segments, info = model.transcribe(audio_input, **options)
            
            if hasattr(info, 'language') and info.language:
                logging.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")

            for segment in segments:
                text = segment.text.strip()
                start = segment.start
                end = segment.end
                print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
                
                # Print word timestamps if requested
                if word_timestamps and segment.words:
                    for word in segment.words:
                        word_text = word.word.strip()
                        word_start = word.start
                        word_end = word.end
                        print(f'  [{word_start:.3f} --> {word_end:.3f}] {word_text}', flush=True)
        
        elif backend == 'insanely-fast-whisper':
            try:
                # This backend uses a CLI tool, so we need to create a temporary file for output
                output_json = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
                temp_files.append(output_json)
                
                # Make sure we have a WAV file for best compatibility
                wav_path = convert_audio_to_wav(audio_input)
                if wav_path != audio_input:
                    temp_files.append(wav_path)
                    audio_input = wav_path
                
                cmd = [
                    "insanely-fast-whisper",
                    "--file-name", audio_input,
                    "--device-id", "0" if device == "cuda" else device,
                    "--model-name", model_id,
                    "--task", "transcribe",
                    "--batch-size", "24",
                    "--timestamp", "chunk" if not word_timestamps else "word",
                    "--transcript-path", output_json
                ]
                
                if language:
                    cmd.extend(["--language", language])
                
                logging.info(f"Running insanely-fast-whisper: {' '.join(cmd)}")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                
                # Read stderr in real-time and log it
                def read_stderr():
                    for line in iter(process.stderr.readline, ''):
                        logging.info(f"insanely-fast-whisper: {line.strip()}")
                
                stderr_thread = threading.Thread(target=read_stderr, daemon=True)
                stderr_thread.start()
                
                # Wait for process to complete
                process.wait()
                stderr_thread.join(timeout=1)
                
                if process.returncode != 0:
                    error_output = process.stderr.read() if process.stderr else "Unknown error"
                    raise Exception(f"Insanely Fast Whisper failed with error: {error_output}")

                # Parse the output JSON file
                import json
                with open(output_json, 'r') as json_file:
                    transcription_data = json.load(json_file)
                
                if word_timestamps and 'words' in transcription_data:
                    # Word-level timestamps
                    for word in transcription_data['words']:
                        text = word['text'].strip()
                        start = word['timestamp'][0]
                        end = word['timestamp'][1]
                        print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
                elif 'chunks' in transcription_data:
                    # Chunk-level timestamps
                    for chunk in transcription_data['chunks']:
                        text = chunk['text'].strip()
                        start = chunk['timestamp'][0]
                        end = chunk['timestamp'][1]
                        print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
                else:
                    # Fall back to full text
                    print(transcription_data.get('text', ''), flush=True)

            except FileNotFoundError:
                raise ImportError("insanely-fast-whisper not found. Install with: pip install insanely-fast-whisper")
            except Exception as e:
                logging.error(f"Error with insanely-fast-whisper: {str(e)}")
                raise
        
        elif backend == 'transformers':
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            except ImportError:
                raise ImportError("transformers and torch are required for transformers backend")

            # Determine torch data type based on device
            if device == 'cpu':
                torch_dtype = torch.float32
            elif device == 'cuda':
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            logging.info(f"Loading model {model_id} with {torch_dtype}")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, device_map=device
            )
            processor = AutoProcessor.from_pretrained(model_id)
            
            # Create pipeline with chunking for long audio
            chunk_length_s = 30
            if max_chunk_length > 0:
                chunk_length_s = max_chunk_length
                
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=chunk_length_s,
                return_timestamps="word" if word_timestamps else "chunk",
                device=device,
            )
            
            # Set language if specified
            if language:
                logging.info(f"Setting language to: {language}")
                asr_pipeline.model.config.forced_decoder_ids = (
                    asr_pipeline.tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
                )
            
            logging.info("Starting transcription...")
            result = asr_pipeline(audio_input)
            
            if word_timestamps and 'chunks' in result:
                for chunk in result['chunks']:
                    text = chunk['text'].strip()
                    start = chunk['timestamp'][0]
                    end = chunk['timestamp'][1]
                    print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
            else:
                text = result.get('text', '').strip()
                print(f'[0.000 --> 0.000] {text}', flush=True)
        
        elif backend == 'whisper.cpp':
            logging.info("=== Starting whisper.cpp pipeline ===")
            
            # 1. Ensure we have a compatible WAV file for best results
            wav_path = convert_audio_to_wav(audio_input)
            if wav_path != audio_input:
                temp_files.append(wav_path)
                audio_input = wav_path
            
            # 2. Find or download the model
            try:
                model_path = find_or_download_whisper_cpp_model(model_id)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model not found at {model_path}")
                model_size = os.path.getsize(model_path)/1024/1024
                logging.info(f"Using model file: {os.path.basename(model_path)} ({model_size:.2f} MB)")
            except Exception as e:
                logging.error(f"Model preparation failed: {str(e)}")
                raise
                
            # 3. Locate the whisper.cpp executable
            whisper_cpp_executable = find_whisper_cpp_executable()
            if not whisper_cpp_executable:
                raise FileNotFoundError("Could not find whisper.cpp executable")
            logging.info(f"Using executable: {whisper_cpp_executable}")
            
            # 4. Prepare output file path if needed
            output_file = None
            if output_format in ('srt', 'vtt'):
                output_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}').name
                temp_files.append(output_file)
            
            # 5. Build the command
            cmd = [
                whisper_cpp_executable,
                '-m', model_path,
                '-f', audio_input,
                '-t', str(min(os.cpu_count() or 4, 8)),  # Reasonable thread count
            ]
            
            # Add options
            if language:
                cmd.extend(['-l', language])
            else:
                cmd.extend(['-l', 'auto'])  # Auto-detect language
                
            if word_timestamps:
                cmd.append('--word-timestamps')
            
            # Add output format options
            if output_format == 'srt':
                cmd.append('--output-srt')
            elif output_format == 'vtt':
                cmd.append('--output-vtt')
            elif output_format == 'txt':
                cmd.append('--output-txt')
                
            # Add file paths if specified
            if output_file:
                cmd.append(output_file)
                
            # 6. Run the command
            logging.info(f"Running whisper.cpp: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # 7. Process output in real-time
            def read_output(stream, is_stderr=False):
                for line in iter(stream.readline, ''):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if is_stderr:
                        logging.info(f"whisper.cpp: {line}")
                    else:
                        # Try to extract timestamps and text
                        timestamp_match = re.match(r'\[(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+)\]\s+(.+)', line)
                        if timestamp_match:
                            print(line, flush=True)
                        else:
                            # For lines without timestamps
                            print(f"[0.000 --> 0.000] {line}", flush=True)
            
            # Start separate threads for stdout and stderr
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout,), daemon=True)
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, True), daemon=True)
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            return_code = process.wait()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            # 8. Handle the process result
            if return_code != 0:
                logging.error(f"whisper.cpp failed with return code {return_code}")
                raise Exception("Transcription process failed")
                
            # 9. If we have a separate output file (srt/vtt), print its contents
            if output_file and os.path.exists(output_file):
                print(f"OUTPUT FILE: {output_file}", flush=True)
                with open(output_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    if not file_content.strip():
                        logging.warning("Output file is empty")
        
        elif backend == 'ctranslate2':
            try:
                import ctranslate2
                from transformers import WhisperProcessor, WhisperTokenizer
                import librosa
            except ImportError:
                raise ImportError("ctranslate2, transformers, and librosa are required for ctranslate2 backend")
            
            # Check if model directory exists
            model_dir = args.model_id
            preprocessor_path = args.preprocessor_path
            
            if not os.path.exists(os.path.join(model_dir, 'model.bin')):
                logging.error(f"model.bin not found in {model_dir}")
                raise FileNotFoundError(f"model.bin not found in {model_dir}")
            
            # Determine if we need to download preprocessor files
            preprocessor_files = ["tokenizer.json", "vocabulary.json", "tokenizer_config.json"]
            preprocessor_missing = not all(os.path.exists(os.path.join(preprocessor_path, f)) for f in preprocessor_files)
            
            if preprocessor_missing:
                logging.info("Preprocessor files not found locally. Attempting to download.")
                
                if original_model_id is None:
                    logging.error("Original model ID is not specified. Cannot load tokenizer and processor.")
                    raise Exception("Original model ID is not specified.")
                
                logging.info(f"Using original model ID: {original_model_id}")
                
                try:
                    tokenizer = WhisperTokenizer.from_pretrained(original_model_id)
                    processor = WhisperProcessor.from_pretrained(original_model_id)
                    logging.info("Loaded tokenizer and processor from original model.")
                except Exception as e:
                    logging.error(f"Failed to load tokenizer and processor: {str(e)}")
                    raise
            else:
                try:
                    tokenizer = WhisperTokenizer.from_pretrained(preprocessor_path)
                    processor = WhisperProcessor.from_pretrained(preprocessor_path)
                    logging.info("Loaded tokenizer and processor from local files.")
                except Exception as e:
                    logging.error(f"Failed to load tokenizer and processor: {str(e)}")
                    raise
            
            # Load the model
            try:
                logging.info(f"Loading CTranslate2 model from {model_dir} on {device}")
                model = ctranslate2.models.Whisper(model_dir, device=device)
                
                # Load and process audio
                logging.info(f"Loading audio from {audio_input}")
                audio_array, sr = librosa.load(audio_input, sr=16000, mono=True)
                
                # Process audio
                inputs = processor(audio_array, return_tensors="np", sampling_rate=16000)
                features = ctranslate2.StorageView.from_array(inputs.input_features)
                
                # Detect language if not specified
                if not language:
                    results = model.detect_language(features)
                    detected_language, probability = results[0][0]
                    logging.info(f"Detected language: {detected_language} with probability {probability:.4f}")
                    language = detected_language
                
                # Prepare prompt
                prompt = tokenizer.convert_tokens_to_ids([
                    "<|startoftranscript|>",
                    language,
                    "<|transcribe|>",
                    "<|notimestamps|>" if not word_timestamps else "",
                ])
                
                # Generate transcription
                logging.info("Running transcription...")
                results = model.generate(features, [prompt], beam_size=5)
                
                # Decode results
                transcription = tokenizer.decode(results[0].sequences_ids[0])
                
                # Clean up and print
                transcription = transcription.replace("<|startoftranscript|>", "")
                transcription = transcription.replace(f"<|{language}|>", "")
                transcription = transcription.replace("<|transcribe|>", "")
                transcription = transcription.replace("<|notimestamps|>", "")
                transcription = transcription.replace("<|endoftext|>", "").strip()
                
                print(f"[0.000 --> 0.000] {transcription}", flush=True)
                
                logging.info("Transcription completed successfully")
                
            except Exception as e:
                logging.error(f"CTranslate2 transcription failed: {str(e)}")
                raise
        
        elif backend == 'whisper-jax':
            try:
                import jax
                import jax.numpy as jnp
                from whisper_jax import FlaxWhisperPipeline
            except ImportError:
                raise ImportError("whisper-jax, jax, and jaxlib are required for whisper-jax backend")
            
            # Set JAX device
            jax_device = None
            if device == 'cuda' or device == 'gpu':
                jax_device = 'gpu' if jax.devices('gpu') else 'cpu'
            else:
                jax_device = 'cpu'
            
            # Set data type based on device
            dtype = jnp.bfloat16 if jax_device == 'gpu' else jnp.float32
            
            logging.info(f"Loading whisper-jax model '{model_id}' with dtype={dtype}")
            pipeline = FlaxWhisperPipeline(model_id, dtype=dtype)
            
            logging.info("Starting transcription with whisper-jax")
            
            # Convert WAV for best compatibility
            wav_path = convert_audio_to_wav(audio_input)
            if wav_path != audio_input:
                temp_files.append(wav_path)
                audio_input = wav_path
            
            # Run transcription
            outputs = pipeline(audio_input, language=language, return_timestamps=word_timestamps)
            
            if word_timestamps and 'chunks' in outputs:
                for chunk in outputs['chunks']:
                    text = chunk['text'].strip()
                    start = chunk['timestamp'][0]
                    end = chunk['timestamp'][1]
                    print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
            else:
                print(outputs['text'], flush=True)
            
            # Print performance metrics
            transcription_time = time.time() - start_time_perf
            audio_file_size = os.path.getsize(audio_input) / (1024 * 1024)
            print(f"Transcription time: {transcription_time:.2f} seconds", flush=True)
            print(f"Audio file size: {audio_file_size:.2f} MB", flush=True)
            print(f"Total transcription time: {transcription_time:.2f} seconds", flush=True)
        
        elif backend == 'openai whisper':
            try:
                import whisper
                from pydub import AudioSegment
                from pydub.silence import split_on_silence
            except ImportError:
                raise ImportError("openai-whisper and pydub are required for OpenAI Whisper backend")
            
            # Load the model
            logging.info(f"Loading OpenAI Whisper model: {model_id}")
            model = whisper.load_model(model_id, device=device)
            
            # Process audio - ensure it's WAV format for best results
            wav_path = convert_audio_to_wav(audio_input)
            if wav_path != audio_input:
                temp_files.append(wav_path)
                audio_input = wav_path
            
            logging.info("Starting transcription")
            
            # Handle chunking for long files if specified
            if max_chunk_length > 0:
                logging.info(f"Processing audio in chunks of {max_chunk_length} seconds")
                audio_segment = AudioSegment.from_file(audio_input)
                
                # Split audio on silence for natural chunks
                chunks = split_on_silence(
                    audio_segment,
                    min_silence_len=500,
                    silence_thresh=audio_segment.dBFS - 14,
                    keep_silence=250
                )
                
                # Merge chunks to respect max_chunk_length
                merged_chunks = []
                current_chunk = AudioSegment.empty()
                for chunk in chunks:
                    if len(current_chunk) + len(chunk) <= max_chunk_length * 1000:
                        current_chunk += chunk
                    else:
                        merged_chunks.append(current_chunk)
                        current_chunk = chunk
                merged_chunks.append(current_chunk)
                
                # Process each chunk
                total_offset = 0.0
                for chunk_index, chunk in enumerate(merged_chunks):
                    # Create temporary file for this chunk
                    chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_files.append(chunk_file.name)
                    chunk_file.close()
                    
                    # Export chunk to file
                    chunk.export(chunk_file.name, format="wav")
                    
                    # Transcribe chunk
                    result = model.transcribe(
                        chunk_file.name, 
                        language=language,
                        word_timestamps=word_timestamps
                    )
                    
                    # Output segments with adjusted timestamps
                    for segment in result['segments']:
                        text = segment['text'].strip()
                        start = segment['start'] + total_offset
                        end = segment['end'] + total_offset
                        print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
                        
                        # Print word timestamps if requested
                        if word_timestamps and 'words' in segment:
                            for word in segment['words']:
                                word_text = word['word'].strip()
                                word_start = word['start'] + total_offset
                                word_end = word['end'] + total_offset
                                print(f'  [{word_start:.3f} --> {word_end:.3f}] {word_text}', flush=True)
                    
                    # Update offset for next chunk
                    total_offset += len(chunk) / 1000.0
            else:
                # Process the entire file at once
                result = model.transcribe(
                    audio_input,
                    language=language,
                    word_timestamps=word_timestamps
                )
                
                # Output segments
                for segment in result['segments']:
                    text = segment['text'].strip()
                    start = segment['start']
                    end = segment['end']
                    print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
                    
                    # Print word timestamps if requested
                    if word_timestamps and 'words' in segment:
                        for word in segment['words']:
                            word_text = word['word'].strip()
                            word_start = word['start']
                            word_end = word['end']
                            print(f'  [{word_start:.3f} --> {word_end:.3f}] {word_text}', flush=True)
            
            # Print performance data
            end_time_perf = time.time()
            transcription_time = end_time_perf - start_time_perf
            print(f"Total transcription time: {transcription_time:.2f} seconds", flush=True)
        
        else:
            logging.error(f"Unsupported backend: {backend}")
            raise ValueError(f"Unsupported backend: {backend}")

        # Final performance report
        end_tr_time = time.time()
        transcription_time = end_tr_time - start_time_perf
        logging.info(f"Transcription completed in {transcription_time:.2f} seconds")
    
    except Exception as e:
        logging.error(f"An error occurred during transcription: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logging.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")

if __name__ == "__main__":
    main()
