# dl_yt_mp3.py
import argparse
import os
import re
import subprocess
import sys
from typing import Optional


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\u0000-\u001F\u007F-\u009F]', "_", filename)
    sanitized = sanitized.strip(". ")
    return sanitized


# ----- AUDIO DOWNLOAD FUNCTIONS -----
def download_audio_with_yt_dlp(url: str) -> Optional[str]:
    try:
        import yt_dlp

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": "%(title)s.%(ext)s",
            "restrictfilenames": True,
            "quiet": False,
            "no_warnings": False,
            "http_headers": {"User-Agent": "Mozilla/5.0"},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info["title"])
            output_file = f"{title}.mp3"
            return output_file
    except Exception as e:
        print(f"yt-dlp audio attempt failed: {str(e)}")
        return None


def download_audio_with_pytube(url: str) -> Optional[str]:
    try:
        from pydub import AudioSegment
        from pytube import YouTube

        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        output_filename = sanitize_filename(yt.title)
        temp_file = audio_stream.download(filename=output_filename)
        output_file = os.path.splitext(temp_file)[0] + ".mp3"
        audio = AudioSegment.from_file(temp_file)
        audio.export(output_file, format="mp3")
        os.remove(temp_file)
        return output_file
    except Exception as e:
        print(f"PyTube audio attempt failed: {str(e)}")
        return None


def download_audio_with_youtube_dl(url: str) -> Optional[str]:
    try:
        import youtube_dl

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "restrictfilenames": True,
            "quiet": False,
            "no_warnings": False,
            "http_headers": {"User-Agent": "Mozilla/5.0"},
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info["title"])
            output_file = f"{title}.mp3"
            return output_file
    except Exception as e:
        print(f"youtube-dl audio attempt failed: {str(e)}")
        return None


# ----- VIDEO DOWNLOAD FUNCTIONS -----
def download_video_with_yt_dlp(url: str) -> Optional[str]:
    try:
        import yt_dlp

        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": "%(title)s.%(ext)s",
            "restrictfilenames": True,
            "quiet": False,
            "no_warnings": False,
            "http_headers": {"User-Agent": "Mozilla/5.0"},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info["title"])
            ext = info.get("ext", "mp4")
            output_file = f"{title}.{ext}"
            return output_file
    except Exception as e:
        print(f"yt-dlp video attempt failed: {str(e)}")
        return None


def download_video_with_pytube(url: str) -> Optional[str]:
    try:
        from pytube import YouTube

        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        output_filename = sanitize_filename(yt.title)
        output_file = stream.download(filename=output_filename)
        return output_file
    except Exception as e:
        print(f"PyTube video attempt failed: {str(e)}")
        return None


def download_video_with_youtube_dl(url: str) -> Optional[str]:
    try:
        import youtube_dl

        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": "%(title)s.%(ext)s",
            "restrictfilenames": True,
            "quiet": False,
            "no_warnings": False,
            "http_headers": {"User-Agent": "Mozilla/5.0"},
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info["title"])
            ext = info.get("ext", "mp4")
            output_file = f"{title}.{ext}"
            return output_file
    except Exception as e:
        print(f"youtube-dl video attempt failed: {str(e)}")
        return None


# ----- NO-AUDIO DOWNLOAD (Video with audio removed) -----
def download_noaudio(method, url: str) -> Optional[str]:
    video_file = method(url)
    if video_file:
        try:
            base, ext = os.path.splitext(video_file)
            output_file = f"{base}_noaudio{ext}"
            subprocess.run(
                ["ffmpeg", "-i", video_file, "-c", "copy", "-an", output_file],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(video_file)
            return output_file
        except Exception as e:
            print(f"Error removing audio: {str(e)}")
            return video_file
    return None


# ----- TRANSCRIPTION DOWNLOAD FUNCTIONS -----
def convert_subtitle_format_builtin(input_file: str, output_format: str = "srt") -> Optional[str]:
    """Convert VTT to other formats using only built-in Python libraries (no dependencies)."""
    try:
        base_name = os.path.splitext(input_file)[0]

        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        if output_format == "txt":
            output_file = f"{base_name}.txt"

            # Extract text content from VTT
            lines = content.split("\n")
            text_lines = []
            prev_line = None

            # Skip VTT header and metadata
            in_cue = False
            for line in lines:
                line = line.strip()

                # Skip empty lines, WEBVTT header, and NOTE lines
                if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
                    continue

                # Skip timestamp lines (contain -->)
                if "-->" in line:
                    in_cue = True
                    continue

                # Skip cue identifiers (lines that are just numbers or IDs before timestamps)
                if not in_cue and (line.isdigit() or ":" in line):
                    continue

                # This should be subtitle text
                if in_cue and line:
                    # Remove VTT tags like <c.colorCCCCCC> or <i>
                    import re

                    clean_line = re.sub(r"<[^>]+>", "", line)
                    clean_line = clean_line.strip()

                    # Only add if not empty and not duplicate
                    if clean_line and clean_line != prev_line:
                        text_lines.append(clean_line)
                        prev_line = clean_line

                # Reset cue flag on empty line
                if not line:
                    in_cue = False

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(text_lines))

            return output_file

        elif output_format == "srt":
            output_file = f"{base_name}.srt"

            # Parse VTT and convert to SRT
            lines = content.split("\n")
            srt_content = []
            cue_number = 1
            current_cue = {}

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Skip WEBVTT header and empty lines
                if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
                    i += 1
                    continue

                # Look for timestamp line
                if "-->" in line:
                    # Parse timestamp
                    timestamp = line.replace(".", ",")  # SRT uses comma instead of dot

                    # Get subtitle text (next non-empty lines)
                    subtitle_lines = []
                    i += 1
                    while i < len(lines) and lines[i].strip():
                        text_line = lines[i].strip()
                        # Remove VTT tags
                        import re

                        clean_text = re.sub(r"<[^>]+>", "", text_line)
                        if clean_text:
                            subtitle_lines.append(clean_text)
                        i += 1

                    # Add to SRT format
                    if subtitle_lines:
                        srt_content.append(f"{cue_number}")
                        srt_content.append(timestamp)
                        srt_content.extend(subtitle_lines)
                        srt_content.append("")  # Empty line between cues
                        cue_number += 1
                else:
                    i += 1

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_content))

            return output_file

    except Exception as e:
        print(f"Built-in conversion failed: {str(e)}")
        return None


def convert_subtitle_format(input_file: str, output_format: str = "srt") -> Optional[str]:
    """Convert VTT to other formats, trying webvtt-py first, then built-in fallback."""

    # First try with webvtt-py (more accurate)
    try:
        import webvtt

        # Get the base filename without extension
        base_name = os.path.splitext(input_file)[0]

        if output_format == "srt":
            vtt = webvtt.read(input_file)
            output_file = f"{base_name}.srt"

            with open(output_file, "w", encoding="utf-8") as f:
                for i, caption in enumerate(vtt, 1):
                    # Convert WebVTT timestamp format to SRT format
                    start = caption.start.replace(".", ",")
                    end = caption.end.replace(".", ",")

                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{caption.text}\n\n")

            return output_file

        elif output_format == "txt":
            vtt = webvtt.read(input_file)
            output_file = f"{base_name}.txt"

            # Keep track of previous line to avoid duplicates
            prev_line = None

            with open(output_file, "w", encoding="utf-8") as f:
                for caption in vtt:
                    # Split into lines and process each one
                    lines = caption.text.split("\n")
                    for line in lines:
                        line = line.strip()
                        # Only write if line is not empty and not a duplicate of previous line
                        if line and line != prev_line:
                            f.write(f"{line}\n")
                            prev_line = line

            return output_file

    except ImportError:
        print("webvtt-py not found, using built-in conversion (may be less accurate)")
        print("For better results, install webvtt-py: pip install webvtt-py")
        return convert_subtitle_format_builtin(input_file, output_format)
    except Exception as e:
        print(f"webvtt-py conversion failed: {str(e)}")
        print("Trying built-in conversion as fallback...")
        return convert_subtitle_format_builtin(input_file, output_format)


def download_transcript_with_yt_dlp(
    url: str, sub_lang: str = "en", output_format: str = "vtt"
) -> Optional[str]:
    """Download transcript using yt-dlp with enhanced options."""
    try:
        import glob

        import yt_dlp

        # Clean up any existing subtitle files first
        existing_files = glob.glob(f"*.{sub_lang}.vtt") + glob.glob(f"*.{sub_lang}.*.vtt")
        for f in existing_files:
            try:
                os.remove(f)
            except:
                pass

        # Try multiple subtitle download strategies
        strategies = [
            # Strategy 1: Standard subtitle download
            {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "vtt",
                "subtitleslangs": [sub_lang],
            },
            # Strategy 2: Try with all available subtitles if specific language fails
            {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "vtt",
                "allsubtitles": True,
            },
            # Strategy 3: Force automatic captions only
            {
                "writeautomaticsub": True,
                "writesubtitles": False,
                "subtitlesformat": "vtt",
                "subtitleslangs": [sub_lang],
            },
        ]

        for i, strategy in enumerate(strategies, 1):
            print(f"Trying download strategy {i}/3...")

            ydl_opts = {
                "skip_download": True,
                "outtmpl": "%(title)s.%(ext)s",
                "restrictfilenames": True,
                "quiet": False,
                "no_warnings": False,
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                **strategy,
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)

                    # Look for subtitle files with various patterns
                    search_patterns = [
                        f"*.{sub_lang}.vtt",
                        f"*.{sub_lang}.*.vtt",
                        "*.en.vtt",  # Fallback to English
                        "*.vtt",  # Any VTT file
                    ]

                    vtt_file = None
                    for pattern in search_patterns:
                        vtt_files = glob.glob(pattern)
                        if vtt_files:
                            vtt_file = vtt_files[0]
                            print(f"Found subtitle file: {vtt_file}")
                            break

                    if vtt_file and os.path.exists(vtt_file):
                        if output_format != "vtt":
                            print(f"Converting to {output_format}...")
                            converted_file = convert_subtitle_format(vtt_file, output_format)
                            if converted_file and os.path.exists(converted_file):
                                try:
                                    os.remove(vtt_file)
                                except OSError:
                                    pass
                                return converted_file
                            else:
                                print(f"Conversion to {output_format} failed, keeping VTT file")
                                return vtt_file
                        return vtt_file

            except Exception as e:
                print(f"Strategy {i} failed: {str(e)}")
                continue

        return None

    except Exception as e:
        print(f"yt-dlp transcript download failed: {str(e)}")
        return None


def download_transcript_with_youtube_dl(
    url: str, sub_lang: str = "en", output_format: str = "vtt"
) -> Optional[str]:
    """Download transcript using youtube-dl as fallback."""
    try:
        import glob

        import youtube_dl

        print(f"Trying youtube-dl fallback...")

        # Clean up any existing subtitle files first
        existing_files = glob.glob(f"*.{sub_lang}.vtt") + glob.glob(f"*.{sub_lang}.*.vtt")
        for f in existing_files:
            try:
                os.remove(f)
            except:
                pass

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "vtt",
            "subtitleslangs": [sub_lang],
            "outtmpl": "%(title)s.%(ext)s",
            "restrictfilenames": True,
            "quiet": False,
            "no_warnings": False,
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # Look for subtitle files
            search_patterns = [f"*.{sub_lang}.vtt", f"*.{sub_lang}.*.vtt", "*.en.vtt", "*.vtt"]

            vtt_file = None
            for pattern in search_patterns:
                vtt_files = glob.glob(pattern)
                if vtt_files:
                    vtt_file = vtt_files[0]
                    print(f"Found subtitle file: {vtt_file}")
                    break

            if vtt_file and os.path.exists(vtt_file):
                if output_format != "vtt":
                    print(f"Converting to {output_format}...")
                    converted_file = convert_subtitle_format(vtt_file, output_format)
                    if converted_file and os.path.exists(converted_file):
                        try:
                            os.remove(vtt_file)
                        except OSError:
                            pass
                        return converted_file
                    else:
                        print(f"Conversion to {output_format} failed, keeping VTT file")
                        return vtt_file
                return vtt_file

        return None

    except Exception as e:
        print(f"youtube-dl transcript download failed: {str(e)}")
        return None


def download_transcript_with_pytube(
    url: str, sub_lang: str = "en", output_format: str = "txt"
) -> Optional[str]:
    """Download transcript using pytube as another fallback."""
    try:

        from pytube import YouTube

        print(f"Trying pytube fallback...")

        yt = YouTube(url)

        # Get captions
        captions = yt.captions

        # Try to get the requested language, fallback to English, then any available
        caption = None
        for lang_code in [sub_lang, "en", "a.en"]:  # 'a.en' is auto-generated English
            if lang_code in captions:
                caption = captions[lang_code]
                print(f"Found captions in language: {lang_code}")
                break

        if not caption:
            # Try any available caption
            if captions:
                caption = list(captions.values())[0]
                print(f"Using available caption: {caption.code}")
            else:
                print("No captions found")
                return None

        # Get caption content
        caption_content = caption.generate_srt_captions()

        # Generate filename
        title = sanitize_filename(yt.title)

        if output_format == "srt":
            output_file = f"{title}.srt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(caption_content)
        elif output_format == "txt":
            output_file = f"{title}.txt"
            # Convert SRT to plain text
            import re

            # Remove SRT formatting (numbers, timestamps, empty lines)
            text_lines = []
            lines = caption_content.split("\n")
            for line in lines:
                line = line.strip()
                # Skip empty lines, numbers, and timestamp lines
                if line and not line.isdigit() and not re.match(r"\d+:\d+:\d+", line):
                    text_lines.append(line)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(text_lines))
        else:  # vtt format
            output_file = f"{title}.vtt"
            # Convert to VTT format (basic conversion)
            vtt_content = "WEBVTT\n\n" + caption_content.replace(",", ".")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(vtt_content)

        return output_file

    except Exception as e:
        print(f"pytube transcript download failed: {str(e)}")
        return None


def download_transcription_with_yt_dlp(
    url: str, sub_lang: str = "en", output_format: str = "vtt"
) -> Optional[str]:
    """Legacy function name - redirects to download_transcript_with_yt_dlp"""
    return download_transcript_with_yt_dlp(url, sub_lang, output_format)


def list_available_subtitles(url: str):
    """List all available subtitle languages for a video."""
    try:
        import yt_dlp

        print("Checking available subtitles...")

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "outtmpl": "%(title)s.%(ext)s",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subtitles = info.get("subtitles", {})
            automatic_captions = info.get("automatic_captions", {})
            available = {}
            for lang, subs in {**subtitles, **automatic_captions}.items():
                available[lang] = [sub["ext"] for sub in subs]

            if available:
                print("\nAvailable subtitles:")
                print("-" * 40)
                for lang, exts in available.items():
                    print(f"  Language: {lang}, Formats: {', '.join(exts)}")
                print("-" * 40)
                print(f"Total languages available: {len(available)}")
            else:
                print("No subtitles found for this video.")

    except Exception as e:
        print(f"Failed to retrieve subtitles: {str(e)}")


# ----- MAIN -----
def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube content (audio, video, or transcript only)",
        epilog="""
Examples:
  python dl_yt.py <url>                           # Download audio (default)
  python dl_yt.py <url> --type video              # Download video
  python dl_yt.py <url> --type transcript         # Download transcript only
  python dl_yt.py <url> --type transcript --sub-format txt  # Download as plain text
  python dl_yt.py <url> --list-subs               # List available subtitle languages
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", "-o", help="Output filename (without extension)")
    parser.add_argument(
        "--type",
        "-t",
        choices=["audio", "video", "noaudio", "transcript", "transcription"],
        default="audio",
        help="Type of download: audio (default), video, noaudio, or transcript",
    )
    parser.add_argument(
        "--first-method",
        "-f",
        choices=["yt-dlp", "PyTube", "youtube-dl"],
        help="Method to try first (for audio/video downloads)",
    )
    parser.add_argument(
        "--sub-lang",
        default="en",
        help="Subtitle language code (default: en). Use --list-subs to see available languages",
    )
    parser.add_argument(
        "--sub-format",
        choices=["vtt", "srt", "txt"],
        default="txt",
        help="Transcript format: vtt (with timestamps), srt (SubRip), or txt (plain text, default)",
    )
    parser.add_argument(
        "--list-subs", action="store_true", help="List available subtitle languages and exit"
    )

    args = parser.parse_args()

    # Handle --list-subs flag
    if args.list_subs:
        list_available_subtitles(args.url)
        return

    # Handle transcript-only downloads (support both 'transcript' and 'transcription')
    if args.type in ["transcript", "transcription"]:
        print(f"\n{'='*50}")
        print(f"TRANSCRIPT-ONLY DOWNLOAD")
        print(f"{'='*50}")
        print(f"URL: {args.url}")
        print(f"Language: {args.sub_lang}")
        print(f"Format: {args.sub_format}")
        print(f"{'='*50}")

        # Define transcript download methods with fallbacks
        transcript_methods = [
            ("yt-dlp", download_transcript_with_yt_dlp),
            ("youtube-dl", download_transcript_with_youtube_dl),
            ("pytube", download_transcript_with_pytube),
        ]

        result = None
        for method_name, method in transcript_methods:
            print(f"\n🔄 Attempting transcript download with {method_name}...")
            try:
                result = method(args.url, args.sub_lang, args.sub_format)
                if result and os.path.exists(result):
                    print(f"✅ Success with {method_name}!")
                    break
                else:
                    print(f"❌ {method_name} did not produce a valid transcript file")
            except Exception as e:
                print(f"❌ {method_name} failed: {str(e)}")
                continue

        if result and os.path.exists(result):
            # Handle custom output filename
            if args.output:
                base_name = sanitize_filename(args.output)
                new_filename = f"{base_name}.{args.sub_format}"
                try:
                    if os.path.exists(result):
                        os.rename(result, new_filename)
                        result = new_filename
                        print(f"Renamed to: {new_filename}")
                except OSError as e:
                    print(f"Warning: Could not rename file: {e}")

            print(f"\n🎉 SUCCESS! Transcript saved as: {result}")

            # Show file info
            try:
                file_size = os.path.getsize(result)
                print(f"📄 File size: {file_size:,} bytes")
                if args.sub_format == "txt":
                    with open(result, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        non_empty_lines = [line for line in lines if line.strip()]
                        print(f"📝 Lines of text: {len(non_empty_lines)}")
                        if non_empty_lines:
                            # Show a preview of the first few lines
                            preview_lines = non_empty_lines[:3]
                            print("📖 Preview:")
                            for line in preview_lines:
                                print(
                                    f"   {line.strip()[:80]}{'...' if len(line.strip()) > 80 else ''}"
                                )
            except Exception as e:
                print(f"Could not read file info: {e}")

            return
        else:
            print("\n❌ ALL TRANSCRIPT METHODS FAILED")
            print("\n🔍 Troubleshooting steps:")
            print("1. Check if the video has captions/subtitles available")
            print("2. Try listing available languages: --list-subs")
            print("3. Try a different language (e.g., --sub-lang en)")
            print("4. Some videos may not have any captions available")
            print("5. Check if the video URL is correct and publicly accessible")

            # Try to get video info to help with debugging
            try:
                print(f"\n🔍 Checking video accessibility...")
                import yt_dlp

                with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                    info = ydl.extract_info(args.url, download=False)
                    print(f"✅ Video found: {info.get('title', 'Unknown title')}")
                    subs = info.get("subtitles", {})
                    auto_subs = info.get("automatic_captions", {})
                    if subs or auto_subs:
                        print(
                            f"📝 Available subtitle languages: {list((subs or {}).keys()) + list((auto_subs or {}).keys())}"
                        )
                    else:
                        print("❌ No subtitles found for this video")
            except Exception as e:
                print(f"❌ Could not access video: {e}")

            return

    # Original handling for audio/video downloads
    methods = {
        "audio": [
            ("yt-dlp", download_audio_with_yt_dlp),
            ("PyTube", download_audio_with_pytube),
            ("youtube-dl", download_audio_with_youtube_dl),
        ],
        "video": [
            ("yt-dlp", download_video_with_yt_dlp),
            ("PyTube", download_video_with_pytube),
            ("youtube-dl", download_video_with_youtube_dl),
        ],
        "noaudio": [
            ("yt-dlp", lambda url: download_noaudio(download_video_with_yt_dlp, url)),
            ("PyTube", lambda url: download_noaudio(download_video_with_pytube, url)),
            ("youtube-dl", lambda url: download_noaudio(download_video_with_youtube_dl, url)),
        ],
    }.get(args.type, [])

    if args.first_method:
        methods.sort(key=lambda m: 0 if m[0] == args.first_method else 1)

    print(f"\n{'='*50}")
    print(f"{args.type.upper()} DOWNLOAD")
    print(f"{'='*50}")

    for method_name, method in methods:
        print(f"\nAttempting {args.type} download with {method_name}...")
        result = method(args.url)
        if result:
            if args.output:
                new_filename = f"{sanitize_filename(args.output)}{os.path.splitext(result)[1]}"
                try:
                    os.rename(result, new_filename)
                    result = new_filename
                except OSError as e:
                    print(f"Warning: Could not rename file: {e}")
            print(f"\n✅ SUCCESS! File saved as: {result}")
            return

    print(f"\n❌ All {args.type} download methods failed.")
    print("Please check the URL and your internet connection.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage examples:")
        print("  python dl_yt.py <youtube_url>                    # Download audio")
        print("  python dl_yt.py <youtube_url> --type transcript  # Download transcript only")
        print("  python dl_yt.py <youtube_url> --list-subs        # List available languages")
        print("  python dl_yt.py <youtube_url> --help             # Show all options")
        sys.exit(1)
    main()
