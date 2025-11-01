# utils/download_utils.py
"""Download utilities"""
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


def download_audio(url, proxies=None, ffmpeg_path="ffmpeg"):
    """
    Download audio from a generic URL using yt-dlp or ffmpeg.
    YouTube URLs are not supported and should be blocked by the caller (GUI).
    """
    import logging
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    # Clean up the URL and remove any playlist parameters (still good practice)
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "list" in query_params:
        query_params.pop("list", None)
        new_query = urlencode(query_params, doseq=True)
        parsed_url = parsed_url._replace(query=new_query)
        url = urlunparse(parsed_url)
        logging.info(f"Modified URL to remove playlist parameter: {url}")

    logging.info(f"Downloading audio from URL: {url}")

    # Define download methods for generic URLs.
    # We try yt-dlp first as it handles many sites (podcasts, soundcloud, etc.),
    # then fall back to ffmpeg for direct links.
    download_methods = [
        download_with_yt_dlp,
        lambda u, p: download_with_ffmpeg(u, p, ffmpeg_path),
    ]

    # Try each download method in sequence with improved error handling
    for download_method in download_methods:
        method_name = (
            download_method.__name__ if hasattr(download_method, "__name__") else "download_method"
        )
        logging.info(f"Trying {method_name}...")

        try:
            audio_file = download_method(url, proxies)

            if audio_file and os.path.exists(audio_file):
                # Validate the downloaded file
                if os.path.getsize(audio_file) > 0:
                    try:
                        # Try to open the file to ensure it's valid
                        with open(audio_file, "rb") as f:
                            # Read first few bytes to verify it's readable
                            f.read(1024)

                        logging.info(f"Successfully downloaded audio using {method_name}")
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
            logging.error(f"{method_name} failed: {str(e)}")
            continue

    logging.error("All download methods failed")
    return None


def download_with_yt_dlp(url, proxies=None):
    """Enhanced version of download_with_yt_dlp for generic URLs."""
    logging.info("Downloading using yt-dlp...")
    import os
    import sys
    import tempfile

    try:
        import yt_dlp
    except ImportError:
        logging.error("yt_dlp is not installed. Please install it to download from URLs.")
        return None

    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="susurrus_ytdlp_")
        output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "noplaylist": True,
            "geo_bypass": True,
            "geo_bypass_country": "US",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",  # Use MP3 for better compatibility
                    "preferredquality": "192",
                }
            ],
            "quiet": False,
            "no_warnings": False,
            "logger": MyLogger(),
            "progress_hooks": [my_hook],
            "ffmpeg_location": shutil.which("ffmpeg"),  # Find ffmpeg in PATH
        }

        if proxies and proxies.get("http"):
            ydl_opts["proxy"] = proxies["http"]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info(f"Downloading {url}...")
            info = ydl.extract_info(url, download=True)
            if not info:
                logging.error("No metadata extracted for the URL")
                return None

            if "entries" in info:
                info = info["entries"][0]

            # Determine the output filename
            output_base = ydl.prepare_filename(info)
            output_file = os.path.splitext(output_base)[0] + ".mp3"

            if os.path.exists(output_file):
                logging.info(f"Downloaded audio: {output_file}")
                return output_file

            # If the MP3 file is not found, search the directory
            mp3_files = [
                os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".mp3")
            ]
            if mp3_files:
                logging.info(f"Found MP3 file: {mp3_files[0]}")
                return mp3_files[0]

            # If still not found, look for any audio file
            audio_extensions = [".mp3", ".m4a", ".wav", ".flac", ".opus", ".ogg"]
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in audio_extensions):
                        audio_file = os.path.join(root, file)
                        logging.info(f"Found audio file: {audio_file}")
                        return audio_file

            raise Exception("yt-dlp did not produce an output file.")

    except Exception as e:
        logging.error(f"yt_dlp download failed: {str(e)}")
        # Clean up temp directory if something goes wrong
        if "temp_dir" in locals():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        return None


# --- REMOVED: download_with_pytube ---
# (This function was YouTube-specific and is no longer needed)


def download_with_ffmpeg(url, proxies=None, ffmpeg_path="ffmpeg"):
    """Download audio using ffmpeg as a final fallback option."""
    logging.info("Downloading using ffmpeg...")

    if shutil.which(ffmpeg_path) is None:
        logging.error(f"{ffmpeg_path} not found in PATH. Please install it for direct downloads.")
        return None

    output_file = None
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_file = temp_file.name
        temp_file.close()

        command = [
            ffmpeg_path,
            "-i",
            url,
            "-acodec",
            "pcm_s16le",  # Use standard PCM format
            "-ar",
            "44100",  # 44.1kHz sample rate
            "-ac",
            "2",  # Stereo
            "-y",  # Overwrite output file
            output_file,
        ]
        env = os.environ.copy()

        # Handle proxy settings
        if proxies:
            if proxies.get("http") and "socks5" in proxies["http"]:
                env["ALL_PROXY"] = proxies["http"]
            else:
                if proxies.get("http"):
                    env["http_proxy"] = proxies["http"]
                if proxies.get("https"):
                    env["https_proxy"] = proxies["https"]

        # Run ffmpeg
        process = subprocess.run(
            command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        if process.returncode != 0:
            error_msg = process.stderr.decode("utf-8", errors="replace")
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

# Helper classes for yt-dlp logging
class MyLogger:
    def debug(self, msg):
        if 'yt-dlp' in msg:
            logging.debug(msg)
        else:
            logging.info(msg)
    def warning(self, msg):
        logging.warning(msg)
    def error(self, msg):
        logging.error(msg)

def my_hook(d):
    if d['status'] == 'downloading':
        logging.info(f"Downloading: {d['_percent_str']} of {d['_total_bytes_str']} at {d['_speed_str']}")
    elif d['status'] == 'finished':
        logging.info(f"Done downloading, now post-processing... {d.get('filename', '')}")