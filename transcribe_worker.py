# transcribe_worker.py
# handled from susurrus.py main script or run manually

import argparse
import os
import logging
import subprocess
import time
import json
import platform
from pathlib import Path

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

def download_with_yt_dlp(url, proxies=None):
    """Download audio using yt-dlp with fallback options."""
    logging.info("Trying to download using yt-dlp...")
    import tempfile
    try:
        import yt_dlp
    except ImportError:
        logging.error("yt_dlp is not installed. Please install it to download from youtube.")
        return None

    try:
        # Create a temporary directory without using context manager
        temp_dir = tempfile.mkdtemp()
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
            ydl_opts['proxy'] = proxies['http']  # Can be 'socks5://hostname:port'

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
        # Clean up temp directory if something goes wrong
        if 'temp_dir' in locals():
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        raise

def download_with_pytube(url, proxies=None):
    """Download audio using pytube as a fallback option."""
    logging.info("Trying to download using pytube...")
    import tempfile
    try:
        from pytube import YouTube
    except ImportError:
        logging.error("pytube package is not installed. Please install it to use this backend.")
        return None

    temp_dir = None
    try:
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
        
        if os.path.exists(new_file) and os.path.getsize(new_file) > 0:
            logging.info(f"Downloaded and converted audio to: {new_file}")
            return new_file
        else:
            raise Exception("Pytube did not produce a valid output file")

    except Exception as e:
        logging.error(f"pytube failed: {str(e)}")
        # Clean up temp directory if something goes wrong
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        raise

def download_with_ffmpeg(url, proxies=None, ffmpeg_path='ffmpeg'):
    """Download audio using ffmpeg as a final fallback option."""
    logging.info("Trying to download using ffmpeg...")
    import tempfile
    import shutil

    if shutil.which(ffmpeg_path) is None:
        logging.error(f"{ffmpeg_path} is not installed or not found in PATH. Please install it to download with ffmpeg.")
        return None

    output_file = None
    try:
        # Create a temporary file that won't be immediately deleted
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        output_file = temp_file.name
        temp_file.close()

        command = [ffmpeg_path, '-i', url, '-q:a', '0', '-map', 'a', output_file]
        env = os.environ.copy()
        
        if proxies:
            if proxies.get('http') and 'socks5' in proxies['http']:
                env['ALL_PROXY'] = proxies['http']
            else:
                if proxies.get('http'):
                    env['http_proxy'] = proxies['http']
                if proxies.get('https'):
                    env['https_proxy'] = proxies['https']

        # Run ffmpeg with proper error handling
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg command failed with exit code {e.returncode}")

        # Verify the output file
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logging.info(f"Downloaded audio using ffmpeg: {output_file}")
            return output_file
        else:
            raise Exception("FFmpeg did not produce a valid output file")

    except Exception as e:
        logging.error(f"ffmpeg download failed: {str(e)}")
        # Clean up the temporary file if something goes wrong
        if output_file and os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except Exception:
                pass
        raise

def download_audio(url, proxies=None, ffmpeg_path='ffmpeg'):
    """
    Download audio from URL using multiple fallback methods.
    Returns tuple of (file_path, is_temporary_file).
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

    last_exception = None
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
                        return audio_file, True
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
            else:
                logging.error(f"No file produced by {download_method.__name__}")
                
        except Exception as e:
            last_exception = e
            logging.error(f"{download_method.__name__} failed: {str(e)}")
            continue

    if last_exception:
        logging.error(f"All download methods failed. Last error: {str(last_exception)}")
    else:
        logging.error("All download methods failed without specific errors")
    
    return None, False

def detect_audio_format(audio_path):
    try:
        from pydub.utils import mediainfo
    except ImportError:
        logging.error("pydub is not installed. Please install it to use this backend.")
        return None
    try:
        info = mediainfo(audio_path)
        return info.get('format_name', 'wav')
    except Exception as e:
        logging.error(f"Could not detect audio format: {str(e)}")
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
    try:
        from pydub import AudioSegment
    except ImportError:
        logging.error("pydub is not installed. Please install it to use this backend.")
        return None
    
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        audio_format = detect_audio_format(audio_path)
        if not audio_format:
            logging.error("Unable to detect audio format.")
            return None
        sound = AudioSegment.from_file(audio_path, format=audio_format)
        sound = sound.set_channels(1)
        sound = sound.set_sample_width(2)
        sound = sound.set_frame_rate(16000)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
            sound.export(temp_wav_file.name, format='wav')
            return temp_wav_file.name
    except Exception as e:
        logging.error(f"Error converting audio to 16-bit 16kHz WAV: {str(e)}")
        raise

def download_whisper_cpp_model_directly(model_file, model_path, proxies=None):
    base_url = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/'
    url = base_url + model_file
    logging.info(f'Downloading {model_file} from {url}')
    try:
        try:
            import requests
        except ImportError:
            logging.error("requests package not available. Cannot download the model.")
            return False
        response = requests.get(url, stream=True, proxies=proxies)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f'Model downloaded to {model_path}')
    except Exception as e:
        logging.error(f"Failed to download model file: {str(e)}")
        return False

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

def setup_whisper_cpp():
    """Download, build and setup whisper.cpp if not present"""
    import os
    import subprocess
    import platform
    import shutil
    from pathlib import Path
    
    logging.info("Setting up whisper.cpp...")
    
    # Define paths
    home_dir = Path.home()
    code_dir = home_dir / "Downloads" / "code"
    susurrus_dir = code_dir / "susurrus"
    whisper_cpp_dir = susurrus_dir / "whisper.cpp"
    build_dir = whisper_cpp_dir / "build"
    
    # Create directories if they don't exist
    code_dir.mkdir(parents=True, exist_ok=True)
    susurrus_dir.mkdir(exist_ok=True)
    
    # Determine executable name based on platform
    is_windows = platform.system() == "Windows"
    exe_name = "whisper-cli.exe" if is_windows else "whisper-cli"
    
    # Check if already built
    if is_windows:
        executable = build_dir / "bin" / "Release" / exe_name
    else:
        executable = build_dir / exe_name
        
    if executable.exists() and os.access(executable, os.X_OK):
        logging.info(f"Found existing whisper.cpp executable at: {executable}")
        return str(executable)
    
    # Clone or update repository
    if not whisper_cpp_dir.exists():
        logging.info("Cloning whisper.cpp repository...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/whisper.cpp.git", str(whisper_cpp_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to clone whisper.cpp: {e.stderr}")
            raise
    else:
        logging.info("Updating existing whisper.cpp repository...")
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=str(whisper_cpp_dir),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to update whisper.cpp: {e.stderr}")
    
    # Clean build directory
    if build_dir.exists():
        logging.info("Cleaning build directory...")
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    
    if is_windows:
        logging.info("Setting up Visual Studio environment...")
        try:
            # Find Visual Studio installation
            vswhere_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
            if not os.path.exists(vswhere_path):
                raise FileNotFoundError("Visual Studio installation not found. Please install Visual Studio with C++ development tools.")
            
            vs_path = subprocess.check_output([
                vswhere_path,
                "-latest",
                "-property", "installationPath"
            ], text=True).strip()
            
            if not vs_path:
                raise RuntimeError("Visual Studio installation path not found")
            
            # Get VS developer command prompt environment
            vcvars_path = os.path.join(vs_path, "VC\\Auxiliary\\Build\\vcvars64.bat")
            if not os.path.exists(vcvars_path):
                raise FileNotFoundError(f"vcvars64.bat not found at {vcvars_path}")
            
            # Get environment variables from VS developer command prompt
            env_cmd = f'"{vcvars_path}" && set'
            env_str = subprocess.check_output(env_cmd, shell=True, text=True)
            env = dict(
                line.split('=', 1)
                for line in env_str.splitlines()
                if '=' in line
            )
            
            # Update current environment
            os.environ.update(env)
            logging.info("Visual Studio environment configured successfully")
            
        except Exception as e:
            logging.error(f"Failed to set up Visual Studio environment: {str(e)}")
            raise
    
    # Build using CMake
    logging.info("Building whisper.cpp...")
    try:
        # Configure
        logging.info("Running CMake configure...")
        cmake_cmd = ["cmake", ".."]
        if is_windows:
            cmake_cmd.extend(["-G", "Visual Studio 17 2022", "-A", "x64"])
        
        subprocess.run(
            cmake_cmd,
            cwd=str(build_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ
        )
        
        # Build
        logging.info("Running CMake build...")
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release"],
            cwd=str(build_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Build failed: {e.stderr}")
        raise
    
    # Verify the build
    if not executable.exists() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Failed to find executable after build at {executable}")
    
    logging.info(f"Successfully built whisper.cpp executable at: {executable}")
    return str(executable)

def find_whisper_cpp_executable():
    """Find or set up whisper.cpp executable"""
    # Define paths
    home_dir = Path.home()
    code_dir = home_dir / "Downloads" / "code"
    susurrus_dir = code_dir / "susurrus"
    whisper_cpp_dir = susurrus_dir / "whisper.cpp"
    build_dir = whisper_cpp_dir / "build"
    
    # Check for existing executable
    is_windows = platform.system() == "Windows"
    exe_name = "whisper-cli.exe" if is_windows else "whisper-cli"  # Updated executable name
    
    if is_windows:
        executable = build_dir / "bin" / "Release" / exe_name
    else:
        executable = build_dir / exe_name
        
    if executable.exists() and os.access(executable, os.X_OK):
        logging.info(f"Found existing whisper.cpp executable at: {executable}")
        return str(executable)
    
    logging.info("No existing executable found, setting up whisper.cpp...")
    return setup_whisper_cpp()

def find_or_download_whisper_cpp_model(model_id):
    """Find or download whisper.cpp model with caching"""
    model_file = f"ggml-{model_id}.bin"
    
    # Check common locations in order of preference
    common_paths = [
        os.path.expanduser(os.path.join("~", "Downloads", model_file)),
        os.path.join(os.getcwd(), model_file),
        os.path.join(os.getcwd(), "models", model_file)
    ]
    
    # Use cached path if exists
    for path in common_paths:
        if os.path.exists(path):
            logging.info(f"Model file '{model_file}' found at: {path}")
            return path
            
    # If not found, download to Downloads directory
    target_path = common_paths[0]
    logging.info(f"Downloading model '{model_file}' to {target_path}")
    download_whisper_cpp_model_directly(model_file, target_path)
    return target_path

def download_whisper_cpp_model(model_name, model_path):
    script_path = find_whisper_cpp_download_script()
    
    if not script_path:
        logging.error("download-ggml-model.sh script not found in home directory, current working directory, PATH locations, or their subfolders.")
        raise FileNotFoundError("download-ggml-model.sh script not found.")

    os.chmod(script_path, 0o755)
    
    logging.info(f"Downloading model '{model_name}' using download-ggml-model.sh")
    try:
        import shutil
        
        subprocess.check_call([script_path, model_name], cwd=os.path.dirname(script_path))
        logging.info(f"Model '{model_name}' downloaded successfully.")
        
        expected_model_file = os.path.join(os.path.dirname(script_path), f"ggml-{model_name}.bin")
        if not os.path.exists(expected_model_file):
            raise FileNotFoundError(f"Expected model file {expected_model_file} not found after download.")
        
        shutil.move(expected_model_file, model_path)
        logging.info(f"Model moved to {model_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download model using download-ggml-model.sh: {str(e)}")
        raise
    except FileNotFoundError as e:
        logging.error(str(e))
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while downloading the model: {str(e)}")
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

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        
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

            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                logging.error("huggingface_hub package not available. Please install these packages to use this backend.")
                raise

            # Download the model files from Hugging Face
            try:
                model_path = snapshot_download(repo_id=model_id)
                logging.info(f"Downloaded model files to: {model_path}")
            except Exception as e:
                logging.error(f"Failed to download model files: {str(e)}")
                raise

            transcribe_options = {
                "path_or_hf_repo": model_id,
                "verbose": True,
                "word_timestamps": word_timestamps,
                "language": language,
            }
            
            try:
                result = mlx_whisper.transcribe(audio_input, **transcribe_options)
            except Exception as e:
                logging.error(f"Transcription failed: {str(e)}")
                raise
        
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
            logging.info("=== Starting whisper.cpp pipeline ===")
            
            # 1. Model preparation
            logging.info(f"Looking for model: {model_id}")
            try:
                model_path = find_or_download_whisper_cpp_model(model_id)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model not found at {model_path}")
                model_size = os.path.getsize(model_path)/1024/1024
                logging.info(f"Using model at: {model_path} (Size: {model_size:.2f} MB)")
            except Exception as e:
                logging.error(f"Model preparation failed: {str(e)}")
                raise

            # 2. Audio conversion
            logging.info("Converting audio to 16-bit 16kHz WAV format")
            try:
                wav_path = convert_audio_to_wav(audio_input)
                if wav_path is None:
                    raise RuntimeError("Audio conversion failed")
                wav_size = os.path.getsize(wav_path)/1024/1024
                logging.info(f"Converted audio saved to: {wav_path} (Size: {wav_size:.2f} MB)")
            except Exception as e:
                logging.error(f"Audio conversion failed: {str(e)}")
                raise

            # 3. Executable location with automatic setup
            try:
                whisper_cpp_executable = None
                if args.whisper_cpp_path:
                    if os.path.isfile(args.whisper_cpp_path) and os.access(args.whisper_cpp_path, os.X_OK):
                        whisper_cpp_executable = args.whisper_cpp_path
                        logging.info(f"Using provided whisper.cpp path: {whisper_cpp_executable}")
                    else:
                        logging.warning(f"Provided whisper.cpp path '{args.whisper_cpp_path}' is invalid")

                if not whisper_cpp_executable:
                    logging.info("Searching for whisper.cpp executable...")
                    try:
                        whisper_cpp_executable = find_whisper_cpp_executable()
                    except FileNotFoundError:
                        logging.info("Executable not found, attempting to set up whisper.cpp...")
                        whisper_cpp_executable = setup_whisper_cpp()

                if not os.path.isfile(whisper_cpp_executable) or not os.access(whisper_cpp_executable, os.X_OK):
                    raise FileNotFoundError(f"Invalid executable: {whisper_cpp_executable}")
                    
                logging.info(f"Using whisper.cpp executable: {whisper_cpp_executable}")
            except Exception as e:
                logging.error(f"Executable setup failed: {str(e)}")
                raise

            # 4. Command preparation
            cmd = [
                str(Path(whisper_cpp_executable).parent / "whisper-cli.exe"),  # Use whisper-cli.exe
                '-m', model_path,
                '-f', wav_path,
                '-l', language if language else 'auto',
                '-t', str(min(os.cpu_count() or 4, 8)),  # Limit threads to reasonable number
                '--output-txt'
            ]

            logging.info(f"Executing command: {' '.join(cmd)}")

            # 5. Process execution with proper handling
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                logging.info("Process started, monitoring output...")
                
                def read_output(pipe, log_func):
                    """Helper function to read and log output"""
                    for line in iter(pipe.readline, ''):
                        line = line.strip()
                        if line:
                            log_func(line)
                            if log_func == logging.info:
                                print(line, flush=True)
                
                # Create threads for non-blocking I/O
                import threading
                
                stdout_thread = threading.Thread(
                    target=read_output,
                    args=(process.stdout, logging.info),
                    daemon=True,
                    name="stdout-monitor"
                )
                stderr_thread = threading.Thread(
                    target=read_output,
                    args=(process.stderr, logging.warning),
                    daemon=True,
                    name="stderr-monitor"
                )
                
                stdout_thread.start()
                stderr_thread.start()
                
                # Wait for process to complete
                return_code = process.wait()
                
                # Wait for output threads to finish
                stdout_thread.join(timeout=1.0)
                stderr_thread.join(timeout=1.0)
                
                # Check return code
                if return_code != 0:
                    stderr_output = process.stderr.read()
                    raise subprocess.CalledProcessError(
                        return_code,
                        cmd,
                        stderr=stderr_output
                    )
                
                # Look for output file
                expected_output = f"{wav_path}.txt"
                if os.path.exists(expected_output):
                    with open(expected_output, 'r', encoding='utf-8') as f:
                        transcription = f.read()
                    logging.info("Successfully read transcription from output file")
                    print(transcription, flush=True)
                else:
                    raise FileNotFoundError(f"Expected output file not found: {expected_output}")
                    
            except subprocess.CalledProcessError as e:
                logging.error(f"whisper.cpp process failed with return code {e.returncode}")
                if e.stderr:
                    logging.error(f"Error output: {e.stderr}")
                raise
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
