# utils/download_utils.py
"""Download utilities for YouTube and other sources"""
import logging
import os
import subprocess
import tempfile


def download_audio(url, proxies=None, ffmpeg_path="ffmpeg"):
    """
    Enhanced download_audio function with improved YouTube handling.
    This is a direct replacement for the existing download_audio function.
    """
    import logging
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    # Clean up the URL and remove any playlist parameters
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "list" in query_params:
        query_params.pop("list", None)
        new_query = urlencode(query_params, doseq=True)
        parsed_url = parsed_url._replace(query=new_query)
        url = urlunparse(parsed_url)
        logging.info(f"Modified URL to remove playlist parameter: {url}")

    logging.info(f"Downloading audio from URL: {url}")

    # Determine if it's a YouTube URL
    is_youtube = any(domain in parsed_url.netloc for domain in ["youtube.com", "youtu.be"])

    # For YouTube URLs, first try using the dl_yt_mp3.py script if available
    if is_youtube:
        try:
            logging.info("Checking for dl_yt_mp3.py...")
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_yt_mp3.py")

            if os.path.exists(script_path):
                logging.info(f"Found dl_yt_mp3.py at {script_path}, using it for YouTube download")

                # Create a temporary directory for the download
                temp_dir = tempfile.mkdtemp(prefix="susurrus_yt_")
                current_dir = os.getcwd()

                try:
                    # Change to temp directory so the script saves files there
                    os.chdir(temp_dir)

                    # Run the script
                    cmd = [sys.executable, script_path, url, "--type", "audio"]
                    logging.info(f"Running command: {' '.join(cmd)}")

                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=300,  # 5 minute timeout
                    )

                    logging.info(f"Process output: {proc.stdout}")
                    if proc.returncode != 0:
                        logging.warning(f"dl_yt_mp3.py returned non-zero: {proc.stderr}")

                    # Find the downloaded mp3 file
                    mp3_files = [f for f in os.listdir(temp_dir) if f.endswith(".mp3")]
                    if mp3_files:
                        mp3_path = os.path.join(temp_dir, mp3_files[0])
                        logging.info(f"Downloaded MP3: {mp3_path}")
                        return mp3_path
                    else:
                        logging.warning("No MP3 file found after running dl_yt_mp3.py")
                finally:
                    # Change back to original directory
                    os.chdir(current_dir)
        except Exception as e:
            logging.warning(f"Error using dl_yt_mp3.py: {e}. Falling back to standard methods.")

    # Define download methods based on URL type
    if is_youtube:
        download_methods = [
            download_with_yt_dlp,
            download_with_pytube,
            lambda u, p: download_with_ffmpeg(u, p, ffmpeg_path),
        ]
    else:
        download_methods = [lambda u, p: download_with_ffmpeg(u, p, ffmpeg_path)]

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

    # Specially improved yt-dlp handling with direct command line call
    if is_youtube:
        try:
            logging.info("Trying direct yt-dlp command line...")

            # Create temporary directory and file for output
            temp_dir = tempfile.mkdtemp(prefix="susurrus_ytdl_")
            output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")

            # Build yt-dlp command
            cmd = [
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format",
                "mp3",
                "-o",
                output_template,
                url,
            ]

            # Add proxy if needed
            if proxies and proxies.get("http"):
                cmd.extend(["--proxy", proxies["http"]])

            logging.info(f"Running command: {' '.join(cmd)}")

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if proc.returncode == 0:
                # Find the output file
                mp3_files = [
                    os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".mp3")
                ]
                if mp3_files:
                    logging.info(f"Downloaded MP3: {mp3_files[0]}")
                    return mp3_files[0]
            else:
                logging.error(f"yt-dlp command failed: {proc.stderr}")
        except Exception as e:
            logging.error(f"Direct yt-dlp command failed: {str(e)}")

    logging.error("All download methods failed")
    return None


def download_with_yt_dlp(url, proxies=None):
    """Enhanced version of download_with_yt_dlp with better error handling and options."""
    logging.info("Downloading using yt-dlp...")
    import os
    import sys
    import tempfile

    try:
        import yt_dlp
    except ImportError:
        logging.error("yt_dlp is not installed. Please install it to download from YouTube.")
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
        # Check if yt-dlp needs updating
        if "This video is unavailable" in str(e) or "HTTP Error 410" in str(e):
            logging.error(
                "YouTube may have changed their API. Try updating yt-dlp: pip install -U yt-dlp"
            )
        # Clean up temp directory if something goes wrong
        if "temp_dir" in locals():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        return None


def download_with_pytube(url, proxies=None):
    """Enhanced version of download_with_pytube with better error handling and options."""
    logging.info("Downloading using pytube...")
    import os
    import sys
    import tempfile

    try:
        from pytube import YouTube
    except ImportError:
        logging.error("pytube package is not installed. Cannot use this method.")
        return None

    temp_dir = None
    try:
        # Use cipher bypass to handle common pytube errors
        yt = YouTube(url, use_oauth=False, allow_oauth_cache=False)

        # Set proxy if needed
        if proxies and proxies.get("http"):
            yt.proxies = proxies

        logging.info(f"Video title: {yt.title}")
        logging.info(f"Available audio streams: {len(yt.streams.filter(only_audio=True))}")

        # Try to get highest bitrate audio stream
        audio_stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()

        if audio_stream is None:
            # Try progressive streams as fallback
            audio_stream = yt.streams.filter(progressive=True).order_by("resolution").desc().first()
            logging.info("No audio streams found, using progressive video stream")

        if audio_stream is None:
            raise Exception("No suitable streams available.")

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="susurrus_pytube_")

        logging.info(f"Downloading stream: {audio_stream}")
        out_file = audio_stream.download(output_path=temp_dir)

        # Get file extension to handle conversion appropriately
        base, ext = os.path.splitext(out_file)
        new_file = base + ".mp3"

        # Convert to MP3 using ffmpeg if available
        try:
            # Check if ffmpeg is available
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                logging.info(f"Converting to MP3 using ffmpeg at {ffmpeg_path}")
                subprocess.run(
                    [
                        ffmpeg_path,
                        "-i",
                        out_file,
                        "-vn",  # No video
                        "-acodec",
                        "libmp3lame",
                        "-ab",
                        "192k",
                        "-ar",
                        "44100",
                        "-y",  # Overwrite output
                        new_file,
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Remove original file
                if os.path.exists(new_file) and os.path.getsize(new_file) > 0:
                    os.remove(out_file)
                else:
                    # If conversion failed, use original file
                    new_file = out_file
            else:
                # No ffmpeg, just rename the file
                logging.info("ffmpeg not found, using original download")
                os.rename(out_file, new_file)

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logging.warning(f"Error converting with ffmpeg: {e}. Using original file.")
            # If ffmpeg fails, just rename
            if os.path.exists(out_file):
                if not os.path.exists(new_file):
                    os.rename(out_file, new_file)
                else:
                    new_file = out_file

        if os.path.exists(new_file) and os.path.getsize(new_file) > 0:
            logging.info(f"Downloaded and processed audio: {new_file}")
            return new_file
        else:
            raise Exception("Failed to produce a valid output file")

    except Exception as e:
        error_msg = str(e)
        logging.error(f"pytube download failed: {error_msg}")

        # Special handling for common pytube errors
        if "decryption" in error_msg.lower() or "cipher" in error_msg.lower():
            logging.error("YouTube cipher issue. Try updating pytube: pip install -U pytube")
        elif "video is unavailable" in error_msg.lower():
            logging.error("Video is unavailable or restricted. Try with a VPN or different URL.")

        # Clean up temp directory if something goes wrong
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        return None


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
