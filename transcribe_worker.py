# transcribe_worker.py
# handled from Susurrus main script or run manually

import argparse
import os
import logging
import subprocess
import time
import json
import requests
import sys
from datetime import datetime

# Debug imports
import sys
print("Python path at startup:", sys.path)
try:
    import pydub
    print("Found pydub at:", pydub.__file__)
except ImportError as e:
    print("Failed to import pydub, sys.path is:", sys.path)
    print("Import error:", str(e))

# Set up logging immediately at script start
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'susurrus_debug_{timestamp}.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Log system information
logging.info("=== Starting Susurrus Debug Log ===")
logging.info(f"Log file: {log_file}")
logging.info(f"Platform: {sys.platform}")
logging.info(f"Python version: {sys.version}")

device_fallbacks = {
    'faster-batched': [('mps', 'cpu')],
    'faster-whisper': [('mps', 'cpu')],
    'openai-whisper': [('mps', 'cpu')],
    # Add other backends and unsupported devices as needed
    # Format: 'backend': [(unsupported_device, fallback_device), ...]
}

def get_supported_device(backend, device):
    if backend in device_fallbacks:
        for unsupported, fallback in device_fallbacks[backend]:
            if device == unsupported:
                logging.warning(f"Device '{device}' is not supported by backend '{backend}'. Switching to '{fallback}'.")
                return fallback
    return device

class MyLogger:
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        logging.error(msg)

def my_hook(d):
    pass  # Suppress progress messages

def get_original_model_id(model_id):
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

def download_audio(url, proxies=None):
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'list' in query_params:
        query_params.pop('list', None)
        new_query = urlencode(query_params, doseq=True)
        parsed_url = parsed_url._replace(query=new_query)
        url = urlunparse(parsed_url)
        logging.info(f"Modified URL to remove playlist parameter: {url}")
    logging.info(f"Downloading audio from URL: {url}")
    audio_file = None

    download_methods = [
        lambda url: download_with_yt_dlp(url, proxies),
        lambda url: download_with_pytube(url, proxies),
        lambda url: download_with_ffmpeg(url, proxies)
    ]

    for method in download_methods:
        try:
            audio_file = method(url)
            if audio_file and os.path.exists(audio_file):
                logging.info(f"Audio downloaded successfully using {method.__name__}")
                return audio_file, True
        except Exception as e:
            logging.error(f"{method.__name__} failed: {str(e)}")
            continue

    logging.error(f"All download methods failed for URL: {url}")
    return None, False

def download_with_yt_dlp(url, proxies=None):
    logging.info("Trying to download using yt-dlp...")
    import tempfile
    try:
        import yt_dlp
    except ImportError:
        logging.error("yt_dlp is not installed. Please install it to download from youtube.")
        return None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = os.path.join(temp_dir, '%(id)s.%(ext)s')
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'noplaylist': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                'logger': MyLogger(),
                'progress_hooks': [my_hook],
            }
            if proxies and proxies.get('http'):
                ydl_opts['proxy'] = proxies['http'] # Can be 'socks5://hostname:port'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if 'entries' in info:
                    info = info['entries'][0]
                output_file = ydl.prepare_filename(info)
                output_file = os.path.splitext(output_file)[0] + '.mp3'
                if os.path.exists(output_file):
                    logging.info(f"Downloaded YouTube audio: {output_file}")
                    return output_file
                else:
                    raise Exception("yt-dlp did not produce an output file.")
    except Exception as e:
        logging.error(f"yt_dlp failed: {str(e)}")
        raise

def download_with_pytube(url, proxies=None):
    logging.info("Trying to download using pytube...")
    import tempfile
    try:
        from pytube import YouTube
    except ImportError:
        logging.error("pytube and tempfile packages are needed for youtube downloads. Please install it to use this backend.")
        return None
    
    yt = YouTube(url)
    if proxies and proxies.get('http'):
        YouTube.proxy = proxies['http']
    audio_stream = yt.streams.filter(only_audio=True).first()
    if audio_stream is None:
        raise Exception("No audio streams available.")
    temp_dir = tempfile.mkdtemp()
    out_file = audio_stream.download(output_path=temp_dir)
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    logging.info(f"Downloaded and converted audio to: {new_file}")
    return new_file

def download_with_ffmpeg(url, proxies=None, ffmpeg_path='ffmpeg'):
    logging.info("Trying to download using ffmpeg...")
    import tempfile
    import shutil
    if shutil.which("ffmpeg") is None:
        logging.error("ffmpeg is not installed or not found in PATH. Please install it to download with ffmpeg.")
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        output_file = temp_file.name
        command = [ffmpeg_path, '-i', url, '-q:a', '0', '-map', 'a', output_file]
        env = os.environ.copy()
        if proxies and 'socks5' in proxies['http']:
            env['ALL_PROXY'] = proxies['http']
        else:
            env['http_proxy'] = proxies['http']
            env['https_proxy'] = proxies['https']
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            if os.path.exists(output_file):
                logging.info(f"Downloaded audio using ffmpeg: {output_file}")
                return output_file
            else:
                raise Exception("ffmpeg did not produce an output file.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running ffmpeg: {e}")
            return None

def detect_audio_format(audio_path):
    """Detect audio format with detailed logging."""
    logging.info(f"Detecting format for audio file: {audio_path}")
    try:
        from pydub.utils import mediainfo
        logging.info("Successfully imported mediainfo from pydub")
        
        if not os.path.exists(audio_path):
            logging.error(f"Audio file does not exist at path: {audio_path}")
            return None
            
        info = mediainfo(audio_path)
        format_name = info.get('format_name', 'wav')
        logging.info(f"Detected audio format: {format_name}")
        logging.debug(f"Full mediainfo: {info}")
        return format_name
    except ImportError as e:
        logging.error(f"ImportError with pydub: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error detecting audio format: {str(e)}")
        logging.error(f"File path: {audio_path}")
        logging.error(f"Exception type: {type(e).__name__}")
        raise

def convert_to_ctranslate2_model(model_id, model_path, quantization):
    try:
        import ctranslate2
    except ImportError:
        logging.error("ctranslate2 package not available. Please install it to use this backend.")
        raise

    logging.info(f"Converting model '{model_id}' to CTranslate2 format at '{model_path}' with quantization '{quantization}'")
    try:
        converter = ctranslate2.converters.TransformersConverter.from_model(
            model_name_or_path=model_id,
            load_as_float16=quantization in ['int8_float16', 'int8_float32']
        )
        converter.convert(model_path, quantization=quantization)
        logging.info(f"Model converted and saved to '{model_path}'")
    except Exception as e:
        logging.error(f"Failed to convert model: {str(e)}")
        raise

def convert_audio_to_wav(audio_path):
    """Convert audio to WAV with detailed logging."""
    logging.info(f"Starting audio conversion for: {audio_path}")
    try:
        from pydub import AudioSegment
        logging.info("Successfully imported AudioSegment from pydub")
        
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Log file details
        file_size = os.path.getsize(audio_path)
        logging.info(f"Input file size: {file_size} bytes")
        
        # Detect format
        audio_format = detect_audio_format(audio_path)
        if not audio_format:
            logging.error("Unable to detect audio format.")
            return None
            
        logging.info(f"Loading audio file with format: {audio_format}")
        sound = AudioSegment.from_file(audio_path, format=audio_format)
        logging.info(f"Original audio: {len(sound)/1000}s, {sound.channels} channels, {sound.frame_rate}Hz, {sound.sample_width} bytes/sample")
        
        # Convert to standard format
        sound = sound.set_channels(1)
        sound = sound.set_sample_width(2)
        sound = sound.set_frame_rate(16000)
        logging.info(f"Converted audio: {len(sound)/1000}s, {sound.channels} channels, {sound.frame_rate}Hz, {sound.sample_width} bytes/sample")
        
        # Export
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
            output_path = temp_wav_file.name
            logging.info(f"Exporting to temporary WAV file: {output_path}")
            sound.export(output_path, format='wav')
            
            # Verify the output file
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                logging.info(f"Successfully created WAV file: {output_path} ({output_size} bytes)")
                return output_path
            else:
                logging.error(f"Failed to create output file at: {output_path}")
                return None
                
    except ImportError as e:
        logging.error(f"ImportError with pydub: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error converting audio to WAV: {str(e)}")
        logging.error(f"Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


def find_whisper_cpp_executable():
    """Find whisper.cpp executable with improved Windows support."""
    logging.info("\nLooking for whisper.cpp executable...")
    
    # On Windows, we need to look in build directories
    if os.name == 'nt':  # Windows
        possible_names = ['whisper.exe', 'main.exe']
        # Add Visual Studio build paths
        extra_paths = [
            'build/bin/Release',
            'build/Release',
            'build/x64/Release',
            'bin/Release',
            'x64/Release',
            'Release'
        ]
    else:
        possible_names = ['whisper', 'main']
        extra_paths = ['build']
    
    # Search in common locations
    search_paths = [
        os.getcwd(),
        os.path.join(os.getcwd(), 'whisper.cpp'),
        os.path.expanduser('~'),
        os.path.join(os.path.expanduser('~'), 'whisper.cpp'),
    ]
    
    # Add the build paths to each search location
    full_search_paths = []
    for base_path in search_paths:
        full_search_paths.append(base_path)
        for extra in extra_paths:
            full_search_paths.append(os.path.join(base_path, extra))
            full_search_paths.append(os.path.join(base_path, 'whisper.cpp', extra))
    
    logging.info("Searching for executable in paths:")
    for path in full_search_paths:
        logging.info(f"- {path}")
        if os.path.exists(path):
            for name in possible_names:
                exe_path = os.path.join(path, name)
                logging.info(f"Checking: {exe_path}")
                if os.path.isfile(exe_path):
                    # On Windows, we don't rely on access() for executable check
                    if os.name == 'nt' or os.access(exe_path, os.X_OK):
                        logging.info(f"Found valid executable at: {exe_path}")
                        return exe_path
                    else:
                        logging.info(f"Found {exe_path} but it's not executable")
                else:
                    logging.info(f"Not a file: {exe_path}")
    
    logging.error("\nwhisper.cpp executable not found!")
    logging.error("Please either:")
    logging.error("1. Install whisper.cpp from https://github.com/ggerganov/whisper.cpp")
    logging.error("2. Specify the path using --whisper-cpp-path argument")
    return None

def find_file(filename, search_paths):
    for base_path in search_paths:
        if os.path.isdir(base_path):
            for root, _, files in os.walk(base_path):
                if filename in files:
                    return os.path.join(root, filename)
    return None

def find_whisper_cpp_download_script():
    search_paths = [
        os.path.expanduser('~'),
        os.getcwd(),
    ] + os.environ.get('PATH', '').split(os.pathsep)

    return find_file('download-ggml-model.sh', search_paths)

def find_or_create_whisper_cpp_models_dir():
    common_locations = [
        os.getcwd(),
        os.path.expanduser('~'),
        os.path.join(os.path.expanduser('~'), 'whisper.cpp'),
    ]
    
    for location in common_locations:
        models_dir = os.path.join(location, 'models')
        if os.path.isdir(models_dir):
            logging.info(f"Existing 'models' directory found at: {models_dir}")
            return models_dir

    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    logging.info(f"Created 'models' directory at: {models_dir}")
    return models_dir

def download_whisper_cpp_model_directly(model_file, model_path, proxies=None):
    """Download a Whisper CPP model with optimized performance."""
    print("\nInitiating direct download...")
    base_url = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/'
    url = base_url + model_file
    print(f"Download URL: {url}")
    
    try:
        import requests
        from tqdm import tqdm
        import certifi
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "tqdm", "certifi"])
        import requests
        from tqdm import tqdm
        import certifi

    # Create session with retry strategy
    session = requests.Session()
    session.mount('https://', requests.adapters.HTTPAdapter(max_retries=1))

    try:
        print(f"Using certificates from: {certifi.where()}")
        print("Checking file availability...")
        
        # First try with SSL verification
        try:
            response = session.get(
                url,
                stream=True,
                timeout=5,
                verify=True,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            print("SSL verification failed, attempting without verification (not recommended)...")
            response = session.get(
                url,
                stream=True,
                timeout=5,
                verify=False,
                headers={'User-Agent': 'Mozilla/5.0'}
            )

        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        print(f"File size: {total_size / (1024*1024):.1f} MB")

        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Download with optimized chunk size and less frequent updates
        chunk_size = 1024 * 1024  # 1MB chunks
        downloaded = 0
        last_update_time = time.time()
        update_interval = 1.0  # Update progress every 1 second

        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress less frequently
                    current_time = time.time()
                    if current_time - last_update_time >= update_interval:
                        percent = (downloaded * 100) / total_size
                        speed = downloaded / (current_time - last_update_time) / (1024*1024)
                        print(f"Downloaded: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB) - {speed:.1f} MB/s")
                        last_update_time = current_time

        actual_size = os.path.getsize(model_path)
        print(f"\nDownload completed. File size: {actual_size / (1024*1024):.1f} MB")

        if total_size > 0 and actual_size != total_size:
            raise Exception(
                f"Download may be incomplete. Expected {total_size} bytes but got {actual_size} bytes"
            )

        return True

    except requests.Timeout:
        print("\nError: Download timed out. Network may be slow or unstable.")
        if os.path.exists(model_path):
            os.remove(model_path)
        raise
    except requests.ConnectionError as e:
        print(f"\nError: Connection failed - {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)
        raise
    except Exception as e:
        print(f"\nError during download: {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)
        raise
    finally:
        session.close()

def find_or_download_whisper_cpp_model(model_id):
    """Find or download a Whisper CPP model."""
    logging.info(f"Looking for Whisper model: {model_id}")
    
    # Format model filename
    model_file = model_id
    if not model_file.startswith('ggml-'):
        model_file = f'ggml-{model_file}'
    if not model_file.endswith('.bin'):
        model_file = f'{model_file}.bin'
    logging.info(f"Formatted model filename: {model_file}")

    # Search for existing model
    search_paths = [
        os.getcwd(),
        os.path.join(os.getcwd(), 'models'),
        os.path.expanduser('~'),
        os.path.join(os.path.expanduser('~'), 'models')
    ]
    
    logging.info("\nSearching for existing model in common locations:")
    for path in search_paths:
        model_path = os.path.join(path, model_file)
        if os.path.exists(model_path):
            logging.info(f"Found existing model at: {model_path}")
            return model_path
        logging.error(f"Not found in: {path}")

    # Prepare for download
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    new_model_path = os.path.join(models_dir, model_file)
    
    if os.path.exists(new_model_path):
        logging.info(f"Found existing model in models directory: {new_model_path}")
        return new_model_path

    # Download model
    logging.info(f"\nDownloading model to: {new_model_path}")
    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if download_whisper_cpp_model_directly(model_file, new_model_path):
                    if os.path.exists(new_model_path) and os.path.getsize(new_model_path) > 0:
                        logging.info(f"\n[OK] Model downloaded successfully: {new_model_path}")
                        return new_model_path
                    else:
                        raise Exception("Download completed but file is missing or empty")
            except (requests.Timeout, requests.RequestException) as e:
                if attempt < max_retries - 1:
                    logging.info(f"\nRetrying download (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(2)  # Wait before retrying
                else:
                    raise
    except Exception as e:
        logging.error(f"\n[ERROR] Download failed after {max_retries} attempts: {str(e)}")
        if os.path.exists(new_model_path):
            os.remove(new_model_path)
        raise

    return new_model_path

def download_whisper_cpp_model_with_script(model_name, model_path):
    print("\nStarting script-based download process...")
    script_path = find_whisper_cpp_download_script()

    if not script_path:
        print("✗ ERROR: download-ggml-model.sh script not found")
        print("Searched in:")
        print("- Home directory")
        print("- Current working directory")
        print("- PATH locations")
        raise FileNotFoundError("download-ggml-model.sh script not found.")

    print(f"Found download script at: {script_path}")
    os.chmod(script_path, 0o755)

    print(f"\nDownloading model '{model_name}' using download-ggml-model.sh")
    try:
        import shutil

        print("Executing download script...")
        process = subprocess.Popen(
            [script_path, model_name],
            cwd=os.path.dirname(script_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        returncode = process.wait()

        if returncode != 0:
            error = process.stderr.read()
            raise subprocess.CalledProcessError(returncode, script_path, error)

        print(f"\nDownload script completed successfully")

        expected_model_file = os.path.join(os.path.dirname(script_path), f"ggml-{model_name}.bin")
        if not os.path.exists(expected_model_file):
            raise FileNotFoundError(f"Expected model file {expected_model_file} not found after download.")

        print(f"Moving model file to final location: {model_path}")
        shutil.move(expected_model_file, model_path)
        print(f"✓ Model successfully moved to: {model_path}")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Script execution failed: {str(e)}")
        if e.stderr:
            print(f"Error output:\n{e.stderr}")
        raise
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {str(e)}")
        raise

def trim_audio(audio_path, start_time, end_time):
    try:
        try:
            from pydub import AudioSegment
        except ImportError:
            logging.error("pydub is not installed. Please install it to use this backend.")
            return audio_path
            
        if not start_time and not end_time:
            logging.info("No trimming required, using the original audio file.")
            return audio_path

        logging.info(f"Trimming audio from {start_time} to {end_time}")
        audio = AudioSegment.from_file(audio_path)
        audio_duration = len(audio) / 1000

        start_time = float(start_time) if start_time else 0
        end_time = float(end_time) if end_time else audio_duration

        start_time = max(0, start_time)
        end_time = min(audio_duration, end_time)

        if start_time >= end_time:
            raise Exception("End time must be greater than start time.")

        trimmed_audio = audio[int(start_time * 1000):int(end_time * 1000)]
        import tempfile
        temp_trimmed = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        trimmed_audio.export(temp_trimmed, format="wav")
        logging.info(f"Trimmed audio saved to: {temp_trimmed}")
        return temp_trimmed
    except Exception as e:
        logging.error(f"Error trimming audio: {str(e)}")
        raise

def is_valid_time(time_value):
    if time_value is None:
        return False
    try:
        return float(time_value) >= 0
    except (ValueError, TypeError):
        return False
                                          
def main():
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
        parser.add_argument('--quantization', default=None, help='Quantization type for ctranslate2 or faster-whisper (e.g., int8, float16)')
        parser.add_argument('--batch-size', type=int, default=16, help='Batch size for batched inference')
        parser.add_argument('--preprocessor-path', type=str, required=False, help='Path to preprocessor files (tokenizer and processor)')
        parser.add_argument('--original-model-id', type=str, required=False, help='Original model ID to use for loading tokenizer and processor if necessary')
       
        parser.add_argument('--start-time', type=float, default=None, help='Start time for audio trimming in seconds')
        parser.add_argument('--end-time', type=float, default=None, help='End time for audio trimming in seconds')
        
        parser.add_argument('--proxy-url', type=str, default=None, help='Proxy URL (supports http://, https://, socks5://)')
        parser.add_argument('--proxy-username', type=str, default=None, help='Proxy username')
        parser.add_argument('--proxy-password', type=str, default=None, help='Proxy password')

        parser.add_argument('--ffmpeg-path', type=str, default='ffmpeg', help='Path to ffmpeg executable')
        parser.add_argument('--whisper-cpp-path', type=str, default=None, help='Path to whisper.cpp executable')

        args = parser.parse_args()

        if not args.audio_input and not args.audio_url:
            parser.error("Either --audio-input or --audio-url must be provided")

        is_temp_file = False
        original_audio_path = None
        working_audio_path = None

        audio_input = args.audio_input
        audio_url = args.audio_url
        model_id = args.model_id
        word_timestamps = args.word_timestamps
        language = args.language
        backend = args.backend
        device_arg = args.device
        pipeline_type = args.pipeline_type
        max_chunk_length = args.max_chunk_length
        output_format = args.output_format
        quantization = args.quantization
        start_time = args.start_time
        end_time = args.end_time
        original_model_id = args.original_model_id
        proxy_url = args.proxy_url
        proxy_username = args.proxy_username
        proxy_password = args.proxy_password

        proxies = None
        if proxy_url:
            proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            if proxy_username and proxy_password:
                # Include authentication in the proxy URL
                from urllib.parse import urlparse, urlunparse
                parsed_url = urlparse(proxy_url)
                netloc = f"{proxy_username}:{proxy_password}@{parsed_url.hostname}"
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

        is_working_file_temp = False

        if language is not None and language.strip() == '':
            language = None    

        try:
            import torch
        except ImportError:
            logging.warning("PyTorch is not installed. Some backends may not work without it.")
            torch = None

        if torch is not None and device_arg.lower() == 'auto':
            if torch is not None and torch.backends.mps.is_available():
                device = "mps"
            elif torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        elif device_arg.lower() == 'cpu':
            device = "cpu"
        elif device_arg.lower() == 'gpu':
            device = "cuda"
        elif device_arg.lower() == 'mps':
            device = "mps"
        else:
            device = "cpu"

        device = get_supported_device(backend, device)
        
        logging.info(f"Starting transcription on device: {device} using backend: {backend}")

        if audio_input and len(audio_input) > 0:
            original_audio_path = audio_input
            is_temp_file = False
        elif audio_url and len(audio_url.strip()) > 0:
            original_audio_path, is_temp_file = download_audio(audio_url, proxies=proxies, ffmpeg_path=args.ffmpeg_path)
            if not original_audio_path:
                error_msg = f"Error downloading audio from {audio_url}. Check logs for details."
                logging.error(error_msg)
                raise Exception(error_msg)
        else:
            error_msg = "No audio source provided. Please upload an audio file or enter a URL."
            logging.error(error_msg)
            raise Exception(error_msg)

        if is_valid_time(start_time) or is_valid_time(end_time):
            logging.info(f"Trimming audio from {start_time} to {end_time}")
            working_audio_path = trim_audio(original_audio_path, start_time, end_time)
            is_working_file_temp = working_audio_path != original_audio_path
        else:
            logging.info("No valid trim times provided, using the original audio file.")
            working_audio_path = original_audio_path
            is_working_file_temp = False

        audio_input = working_audio_path
        
        start_tr_time = time.time()

        if backend == 'mlx-whisper':
            try:
                import mlx_whisper
            except ImportError:
                logging.error("mlx_whisper package not available. Please install these packages to use this backend.")
                raise

            transcribe_options = {
                "path_or_hf_repo": model_id,
                "verbose": True,
                "word_timestamps": word_timestamps,
                "language": language,
            }
            result = mlx_whisper.transcribe(audio_input, **transcribe_options)
        
        elif backend == 'faster-batched':
            try:
                from faster_whisper import WhisperModel, BatchedInferencePipeline
            except ImportError:
                logging.error("faster_whisper package not available. Please install it to use this backend.")
                raise

            # we should not need this anymore with: device = get_supported_device(backend, device)
            if device == 'mps':
                logging.warning("Faster-Whisper does not support MPS device. Switching to CPU.")
                device = 'cpu'

            compute_type = args.quantization if args.quantization else 'int8'

            logging.info("Loading model...")
            model = WhisperModel(model_id, device=device, compute_type=compute_type)
            pipeline = BatchedInferencePipeline(model=model)

            logging.info("Starting batched transcription")
            segments, info = pipeline.transcribe(audio_input, batch_size=args.batch_size)

            for segment in segments:
                text = segment.text.strip()
                start = segment.start
                end = segment.end
                print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
        
        elif backend == 'faster-sequenced':
            
            # we should not need this anymore with: device = get_supported_device(backend, device)
            if device == 'mps':
                logging.warning("Faster-Whisper does not support MPS device. Switching to CPU.")
                device = 'cpu'

            try:
                from faster_whisper import WhisperModel
            except ImportError:
                logging.error("faster_whisper package not available. Please install it to use this backend.")
                raise

            if device == 'cuda':
                compute_type = "float16"
            elif device == 'cpu':
                compute_type = "int8"
            elif device == 'mps':
                compute_type = "int8"
            else:
                compute_type = "int8"

            logging.info("Loading model...")
            model = WhisperModel(model_id, device=device, compute_type=compute_type)

            options = {
                "language": language,
                "beam_size": 5,
                "best_of": 5,
                "word_timestamps": word_timestamps,
            }

            segments, _ = model.transcribe(audio_input, **options)
            for segment in segments:
                text = segment.text.strip()
                start = segment.start
                end = segment.end
                print(f'[{start:0>5.3f} --> {end:0>5.3f}] {text}', flush=True)
        
        elif backend == 'insanely-fast-whisper':
            output_json = "output.json"
            cmd = [
                "insanely-fast-whisper",
                "--file-name", audio_input,
                "--device-id", "0" if device == "cuda" else device,
                "--model-name", model_id,
                "--task", "transcribe",
                "--batch-size", "24",
                "--timestamp", "chunk",
                "--transcript-path", output_json
            ]
            
            if language:
                cmd.extend(["--language", language])
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )

            process.wait()

            if process.returncode != 0:
                error_output = process.stderr.read()
                raise Exception(f"Insanely Fast Whisper failed with error: {error_output}")

            try:
                with open(output_json, 'r') as json_file:
                    transcription_data = json.load(json_file)
                
                for chunk in transcription_data['chunks']:
                    text = chunk['text'].strip()
                    start = chunk['timestamp'][0]
                    end = chunk['timestamp'][1]
                    print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)

                os.remove(output_json)

            except Exception as e:
                raise Exception(f"Error parsing output JSON: {str(e)}")
        
        
        elif backend == 'transformers':
            try:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            except ImportError:
                logging.error("transformers package not available. Please install it to use this backend.")
                raise

            if device == 'cpu':
                torch_dtype = torch.float32
            elif device == 'cuda':
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            logging.info("Loading model...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype
            ).to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                return_timestamps="word" if word_timestamps else None,
                device=device,
            )
            logging.info("Starting transcription...")
            result = asr_pipeline(audio_input)
            if 'chunks' in result:
                for chunk in result['chunks']:
                    text = chunk['text'].strip()
                    start = chunk['timestamp'][0]
                    end = chunk['timestamp'][1]
                    print(f'[{start:0>5.3f} --> {end:0>5.3f}] {text}', flush=True)
            else:
                text = result['text'].strip()
                print(f'[00:00.000 --> XX:XX.XXX] {text}', flush=True)
        
        elif backend == 'whisper.cpp':

            model_path = find_or_download_whisper_cpp_model(model_id)
            logging.info(f"Using model at: {model_path}")

            logging.info(f"Audio input path before conversion: {audio_input}")
            logging.info(f"Audio input path exists: {os.path.exists(audio_input)}")
            if os.path.exists(audio_input):
                logging.info(f"Audio input file size: {os.path.getsize(audio_input)} bytes")
            
            logging.info("Converting audio to 16-bit 16kHz WAV format")
            converted_audio = convert_audio_to_wav(audio_input)
            if converted_audio is None:
                logging.error("Failed to convert audio to WAV format.")
                return
                
            audio_input = converted_audio  # Use the converted audio path
            logging.info(f"Using converted audio at: {audio_input}")
            if audio_input is None:
                logging.error("Failed to convert audio to WAV format.")
                return

            whisper_cpp_executable = None
            if args.whisper_cpp_path:
                if os.path.isfile(args.whisper_cpp_path) and os.access(args.whisper_cpp_path, os.X_OK):
                    whisper_cpp_executable = args.whisper_cpp_path
                    logging.info(f"Using provided whisper.cpp path: {whisper_cpp_executable}")
                else:
                    logging.warning(f"Provided whisper.cpp path '{args.whisper_cpp_path}' is not valid.")

            if not whisper_cpp_executable:
                whisper_cpp_executable = find_whisper_cpp_executable()

            if not whisper_cpp_executable:
                error_msg = "whisper.cpp executable not found. You need to either:\n" + \
                        "1. Install whisper.cpp from https://github.com/ggerganov/whisper.cpp\n" + \
                        "2. Provide the path using --whisper-cpp-path argument"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)

            logging.info(f"Using whisper.cpp executable: {whisper_cpp_executable}")

            cmd = [
                whisper_cpp_executable,
                '-m', model_path,
                '-f', audio_input,
                '--language', language if language else 'en',
                f'--output-{output_format}',
                '--threads', str(os.cpu_count()),
            ]
            
            # For German text, let's also add print-colors and UTF-8 handling
            #cmd.extend(['--print-colors', '--pc'])
            
            logging.info(f"Running command: {' '.join(cmd)}")
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1,
                    encoding='utf-8'  # Explicitly specify UTF-8 encoding
                )

                # Capture and log stdout
                output_lines = []
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    
                    text = line.strip()
                    if text:
                        logging.info(f"STDOUT: {text}")
                        output_lines.append(text)
                        print(text, flush=True)  # For GUI display

                # Get any remaining stderr
                stderr_output = process.stderr.read()
                if stderr_output:
                    logging.error(f"STDERR: {stderr_output}")

                # Check the return code
                if process.returncode != 0:
                    error_msg = f"whisper.cpp transcription failed with code {process.returncode}"
                    if stderr_output:
                        error_msg += f": {stderr_output}"
                    logging.error(error_msg)
                    raise Exception(error_msg)

                # Handle SRT/VTT output files
                if output_format in ('srt', 'vtt'):
                    output_file = f"{audio_input}.{output_format}"
                    if os.path.exists(output_file):
                        logging.info(f"Found output file: {output_file}")
                        print(f"OUTPUT FILE: {output_file}", flush=True)
                    else:
                        error_msg = f"Output file not found: {output_file}"
                        logging.error(error_msg)
                        raise FileNotFoundError(error_msg)

            except Exception as e:
                logging.error(f"Error during whisper.cpp execution: {str(e)}")
                raise
    
        elif backend == 'ctranslate2':
            import tempfile
            try:
                import ctranslate2
                from transformers import WhisperProcessor, WhisperTokenizer
                import librosa
                from pydub import AudioSegment
                from pydub.silence import split_on_silence
            except ImportError:
                    logging.error("packages for ctranslate2 (ctranslate2, librosa, transformers, pydub, torch) are not installed. Please install them to use this backend.")
                    raise

            try:
                import torch
            except ImportError:
                logging.warning("PyTorch is not installed. Some backends may not work without it.")
                torch = None
            
            if torch is not None and torch.backends.mps.is_available():
                device = "cpu"
                logging.warning("Defaulting to CPU on Apple Metal (MPS) architecture for ctranslate2")

            quantization = args.quantization if args.quantization else 'int8_float16'

            logging.info(f"Loading model from {args.model_id}...")

            model_dir = args.model_id
            preprocessor_path = args.preprocessor_path

            if not os.path.exists(os.path.join(model_dir, 'model.bin')):
                logging.error(f"model.bin not found in {model_dir}. Model conversion may have failed.")
                raise FileNotFoundError(f"model.bin not found in {model_dir}")
            else:
                logging.info(f"Using existing model in {model_dir}")

            preprocessor_files = ["tokenizer.json", "vocabulary.json", "tokenizer_config.json"]
            preprocessor_missing = not all(os.path.exists(os.path.join(preprocessor_path, f)) for f in preprocessor_files)

            if preprocessor_missing:
                logging.info("Preprocessor files not found locally. Attempting to download from original model.")

                if original_model_id is None:
                    logging.error("Original model ID is not specified. Cannot load tokenizer and processor.")
                    raise Exception("Original model ID is not specified.")

                logging.info(f"Original model ID determined as: {original_model_id}")

                try:
                    tokenizer = WhisperTokenizer.from_pretrained(original_model_id)
                    processor = WhisperProcessor.from_pretrained(original_model_id)
                    logging.info("WhisperTokenizer and WhisperProcessor loaded successfully from original model.")
                except Exception as e:
                    logging.error(f"Failed to load tokenizer and processor from original model: {str(e)}")
                    raise
            else:
                try:
                    tokenizer = WhisperTokenizer.from_pretrained(preprocessor_path)
                    processor = WhisperProcessor.from_pretrained(preprocessor_path)
                    logging.info("WhisperTokenizer and WhisperProcessor loaded successfully.")
                except Exception as e:
                    logging.error(f"Failed to load tokenizer and processor from preprocessor path: {str(e)}")
                    raise

            try:
                model = ctranslate2.models.Whisper(model_dir, device=device)
                logging.info("CTranslate2 model loaded successfully.")

                logging.info(f"Loading audio from {audio_input}")
                if audio_input is None:
                    raise ValueError("audio_input is None. Please provide a valid audio file path.")
                
                if not os.path.exists(audio_input):
                    raise FileNotFoundError(f"Audio file not found: {audio_input}")
                
                audio_segment = AudioSegment.from_file(audio_input)
                logging.info(f"Audio loaded. Duration: {len(audio_segment)/1000:.2f} seconds")

                chunks = split_on_silence(
                    audio_segment,
                    min_silence_len=500,
                    silence_thresh=audio_segment.dBFS - 14,
                    keep_silence=250
                )

                max_chunk_length = 30 * 1000
                merged_chunks = []
                current_chunk = AudioSegment.empty()
                for chunk in chunks:
                    if len(current_chunk) + len(chunk) <= max_chunk_length:
                        current_chunk += chunk
                    else:
                        merged_chunks.append(current_chunk)
                        current_chunk = chunk
                merged_chunks.append(current_chunk)

                total_offset = 0.0
                for chunk_index, chunk in enumerate(merged_chunks):
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
                        chunk.export(temp_audio_file.name, format="wav")
                        
                        audio_array, sr = librosa.load(temp_audio_file.name, sr=16000, mono=True)
                        
                        inputs = processor(audio_array, return_tensors="np", sampling_rate=16000)
                        features = ctranslate2.StorageView.from_array(inputs.input_features)

                        if chunk_index == 0 and not language:
                            results = model.detect_language(features)
                            detected_language, probability = results[0][0]
                            logging.info(f"Detected language {detected_language} with probability {probability:.4f}")
                            language = detected_language

                        prompt = tokenizer.convert_tokens_to_ids([
                            "<|startoftranscript|>",
                            language,
                            "<|transcribe|>",
                            "<|notimestamps|>" if not word_timestamps else "",
                        ])

                        results = model.generate(features, [prompt], beam_size=5)

                        transcription = tokenizer.decode(results[0].sequences_ids[0])

                        chunk_start_time = total_offset
                        print(f"[{chunk_start_time:.2f}s] {transcription}", flush=True)

                    total_offset += len(chunk) / 1000.0

                logging.info("Transcription completed successfully.")

            except Exception as e:
                logging.error(f"Transcription failed: {str(e)}")
                logging.error(f"Error details: {type(e).__name__}")
                import traceback
                logging.error(traceback.format_exc())
                raise

        elif backend == 'whisper-jax':
            try:
                import jax
                import jax.numpy as jnp
                from whisper_jax import FlaxWhisperPipline
                
                if device.lower() == 'auto':
                    device = 'gpu' if jax.devices('gpu') else 'cpu'
                elif device.lower() in ['cpu', 'gpu']:
                    device = device.lower()
                else:
                    logging.warning("whisper-jax supports 'cpu' or 'gpu' devices. Using 'cpu'.")
                    device = 'cpu'

                dtype = jnp.bfloat16 if device == 'gpu' else jnp.float32

                logging.info(f"Loading whisper-jax model '{model_id}' with dtype={dtype}")
                pipeline = FlaxWhisperPipline(model_id, dtype=dtype)
                
                logging.info("Starting transcription with whisper-jax")
                start_time_perf = time.time()

                text = pipeline(audio_input)

                end_time_perf = time.time()

                print(text['text'], flush=True)

                transcription_time = end_time_perf - start_time_perf
                audio_file_size = os.path.getsize(audio_input) / (1024 * 1024)
                metrics_output = (
                    f"Transcription time: {transcription_time:.2f} seconds\n"
                    f"Audio file size: {audio_file_size:.2f} MB\n"
                )
                print(metrics_output, flush=True)

            except ImportError as e:
                logging.error(f"Failed to import required modules for whisper-jax: {str(e)}")
                logging.error("Please ensure you have installed whisper-jax and its dependencies correctly.")
                logging.error("You may need to update JAX and whisper-jax to compatible versions.")
                logging.error("Try running: pip install --upgrade jax jaxlib whisper-jax")
                raise
            except AttributeError as e:
                if 'NamedShape' in str(e):
                    logging.error("Encountered a NamedShape AttributeError.")
                else:
                    logging.error(f"An unexpected AttributeError occurred: {str(e)}")
                raise
            except Exception as e:
                logging.error(f"An error occurred while using whisper-jax: {str(e)}")
                logging.error("Please ensure you have the latest versions of jax and whisper-jax installed.")
                raise
        
        elif backend == 'openai whisper':
            try:
                import whisper
                from pydub import AudioSegment
                from pydub.silence import split_on_silence
            except ImportError:
                logging.error("The 'whisper' and 'pydub' packages are needed for OpenAI Whisper. Please install these packages to use this backend.")
                raise
                
            # we should not need this anymore with: device = get_supported_device(backend, device)
            if device == 'mps':
                logging.warning("OpenAI Whisper does not currently support MPS device. Switching to CPU.")
                device = 'cpu'

            model = whisper.load_model(model_id, device=device)

            max_chunk_length = float(args.max_chunk_length) if args.max_chunk_length else 0.0

            logging.info("Starting transcription")

            audio_input = convert_audio_to_wav(audio_input)

            start_oaw_time = time.time()

            if max_chunk_length > 0:
                audio_segment = AudioSegment.from_wav(audio_input)

                chunks = split_on_silence(
                    audio_segment,
                    min_silence_len=500,
                    silence_thresh=audio_segment.dBFS - 14,
                    keep_silence=250
                )

                merged_chunks = []
                current_chunk = AudioSegment.empty()
                for chunk in chunks:
                    if len(current_chunk) + len(chunk) <= max_chunk_length * 1000:
                        current_chunk += chunk
                    else:
                        merged_chunks.append(current_chunk)
                        current_chunk = chunk
                merged_chunks.append(current_chunk)

                total_offset = 0.0
                for chunk in merged_chunks:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
                        chunk.export(temp_audio_file.name, format="wav")
                        result = model.transcribe(temp_audio_file.name, language=language)
                        for segment in result['segments']:
                            text = segment['text'].strip()
                            start = segment['start'] + total_offset
                            end = segment['end'] + total_offset
                            print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)
                    total_offset += len(chunk) / 1000.0
            else:
                result = model.transcribe(audio_input, language=language)
                for segment in result['segments']:
                    text = segment['text'].strip()
                    start = segment['start']
                    end = segment['end']
                    print(f'[{start:.3f} --> {end:.3f}] {text}', flush=True)

            end_oaw_time = time.time()
            transcription_oaw_time = end_oaw_time - start_oaw_time
            print(f"Transcription time with OpenAI Whisper: {transcription_oaw_time:.2f} seconds", flush=True)

        else:
            logging.error(f"Unsupported backend: {backend}")
            raise Exception(f"Unsupported backend: {backend}")

        
        end_tr_time = time.time()
        transcription_time = end_tr_time - start_tr_time
        print(f"Total transcription time: {transcription_time:.2f} seconds", flush=True)
        
        logging.info(f"Transcription completed in {transcription_time:.2f} seconds")
    
    except Exception as e:
        logging.error(f"An error occurred during transcription: {str(e)}", exc_info=True)
        is_working_file_temp = False
        raise

    finally:
        if is_temp_file and os.path.exists(original_audio_path):
            os.remove(original_audio_path)
            logging.info(f"Removed temporary downloaded file: {original_audio_path}")
        if is_working_file_temp and os.path.exists(working_audio_path):
            os.remove(working_audio_path)
            logging.info(f"Removed temporary trimmed file: {working_audio_path}")
        # Clean up temporary files created during downloads
        import tempfile
        tempfile.tempdir = None  # Reset the temporary directory if changed

if __name__ == "__main__":
    main()
