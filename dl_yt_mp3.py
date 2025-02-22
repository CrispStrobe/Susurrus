import sys
import os
import re
import argparse
import subprocess
from typing import Optional

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\u0000-\u001F\u007F-\u009F]', '_', filename)
    sanitized = sanitized.strip('. ')
    return sanitized

# ----- AUDIO DOWNLOAD FUNCTIONS -----
def download_audio_with_yt_dlp(url: str) -> Optional[str]:
    try:
        import yt_dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': '%(title)s.%(ext)s',
            'restrictfilenames': True,
            'quiet': False,
            'no_warnings': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0'
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info['title'])
            output_file = f"{title}.mp3"
            return output_file
    except Exception as e:
        print(f"yt-dlp audio attempt failed: {str(e)}")
        return None

def download_audio_with_pytube(url: str) -> Optional[str]:
    try:
        from pytube import YouTube
        from pydub import AudioSegment
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        output_filename = sanitize_filename(yt.title)
        temp_file = audio_stream.download(filename=output_filename)
        output_file = os.path.splitext(temp_file)[0] + '.mp3'
        audio = AudioSegment.from_file(temp_file)
        audio.export(output_file, format='mp3')
        os.remove(temp_file)
        return output_file
    except Exception as e:
        print(f"PyTube audio attempt failed: {str(e)}")
        return None

def download_audio_with_youtube_dl(url: str) -> Optional[str]:
    try:
        import youtube_dl
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'restrictfilenames': True,
            'quiet': False,
            'no_warnings': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0'
            }
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info['title'])
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
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': '%(title)s.%(ext)s',
            'restrictfilenames': True,
            'quiet': False,
            'no_warnings': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0'
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info['title'])
            ext = info.get('ext', 'mp4')
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
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': '%(title)s.%(ext)s',
            'restrictfilenames': True,
            'quiet': False,
            'no_warnings': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0'
            }
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info['title'])
            ext = info.get('ext', 'mp4')
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
            subprocess.run(['ffmpeg', '-i', video_file, '-c', 'copy', '-an', output_file],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(video_file)
            return output_file
        except Exception as e:
            print(f"Error removing audio: {str(e)}")
            return video_file
    return None

# ----- TRANSCRIPTION DOWNLOAD FUNCTION -----
def convert_subtitle_format(input_file: str, output_format: str = 'srt') -> Optional[str]:
    try:
        import webvtt
        
        # Get the base filename without extension
        base_name = os.path.splitext(input_file)[0]
        
        if output_format == 'srt':
            vtt = webvtt.read(input_file)
            output_file = f"{base_name}.srt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, caption in enumerate(vtt, 1):
                    # Convert WebVTT timestamp format to SRT format
                    start = caption.start.replace('.', ',')
                    end = caption.end.replace('.', ',')
                    
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{caption.text}\n\n")
            
            return output_file
            
        elif output_format == 'txt':
            vtt = webvtt.read(input_file)
            output_file = f"{base_name}.txt"
            
            # Keep track of previous line to avoid duplicates
            prev_line = None
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for caption in vtt:
                    # Split into lines and process each one
                    lines = caption.text.split('\n')
                    for line in lines:
                        line = line.strip()
                        # Only write if line is not empty and not a duplicate of previous line
                        if line and line != prev_line:
                            f.write(f"{line}\n")
                            prev_line = line
            
            return output_file
            
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        print(f"If using txt/srt format, make sure webvtt-py is installed: pip install webvtt-py")
        return None

def download_transcription_with_yt_dlp(url: str, sub_lang: str = 'en', output_format: str = 'vtt') -> Optional[str]:
    try:
        import yt_dlp
        import glob
        
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'vtt',
            'subtitleslangs': [sub_lang],
            'outtmpl': '%(title)s.%(ext)s',
            'restrictfilenames': True,
            'quiet': False,
            'no_warnings': False
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Use glob to find the actual file
            vtt_files = glob.glob(f"*.{sub_lang}.vtt")
            if vtt_files:
                vtt_file = vtt_files[0]  # Take the first match
                print(f"Found subtitle file: {vtt_file}")
                
                if output_format != 'vtt':
                    print(f"Converting to {output_format}...")
                    converted_file = convert_subtitle_format(vtt_file, output_format)
                    if converted_file and os.path.exists(converted_file):
                        # os.remove(vtt_file)  # Commented out VTT deletion
                        return converted_file
                    else:
                        print(f"Conversion to {output_format} failed, keeping VTT file")
                        return vtt_file
                return vtt_file
            else:
                print(f"No .{sub_lang}.vtt files found after download")
                return None

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def list_available_subtitles(url: str):
    try:
        import yt_dlp
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'outtmpl': '%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})
            available = {}
            for lang, subs in {**subtitles, **automatic_captions}.items():
                available[lang] = [sub['ext'] for sub in subs]
            if available:
                print("Available subtitles:")
                for lang, exts in available.items():
                    print(f"  Language: {lang}, Formats: {', '.join(exts)}")
            else:
                print("No subtitles found for this video.")
    except Exception as e:
        print(f"Failed to retrieve subtitles: {str(e)}")

# ----- MAIN -----
def main():
    parser = argparse.ArgumentParser(description='Download YouTube content')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--output', '-o', help='Output filename (without extension)')
    parser.add_argument('--type', '-t', choices=['audio', 'video', 'noaudio', 'transcription'],
                        default='audio', help='Type of download (default: audio)')
    parser.add_argument('--first-method', '-f', choices=['yt-dlp', 'PyTube', 'youtube-dl'],
                        help='Method to try first')
    parser.add_argument('--sub-lang', default='en', help='Subtitle language (default: en)')
    parser.add_argument('--sub-format', choices=['vtt', 'srt', 'txt'], default='vtt',
                        help='Subtitle format (default: vtt)')
    parser.add_argument('--list-subs', action='store_true', help='List available subtitles and exit')
    
    args = parser.parse_args()

    if args.list_subs:
        list_available_subtitles(args.url)
        return

    # Different handling for transcription vs other types
    if args.type == 'transcription':
        print(f"\nAttempting transcription download with yt-dlp...")
        result = download_transcription_with_yt_dlp(args.url, args.sub_lang, args.sub_format)
        
        if result:
            if args.output:
                new_filename = f"{sanitize_filename(args.output)}.{args.sub_format}"
                try:
                    os.rename(result, new_filename)
                    result = new_filename
                except OSError as e:
                    print(f"Warning: Could not rename file: {e}")
            print(f"\nSuccess! File saved as: {result}")
            return
        else:
            print("\nFailed to download subtitles. They might not be available for this video.")
            return
    
    # Original handling for other types
    methods = {
        'audio': [
            ('yt-dlp', download_audio_with_yt_dlp),
            ('PyTube', download_audio_with_pytube),
            ('youtube-dl', download_audio_with_youtube_dl)
        ],
        'video': [
            ('yt-dlp', download_video_with_yt_dlp),
            ('PyTube', download_video_with_pytube),
            ('youtube-dl', download_video_with_youtube_dl)
        ],
        'noaudio': [
            ('yt-dlp', lambda url: download_noaudio(download_video_with_yt_dlp, url)),
            ('PyTube', lambda url: download_noaudio(download_video_with_pytube, url)),
            ('youtube-dl', lambda url: download_noaudio(download_video_with_youtube_dl, url))
        ]
    }.get(args.type, [])

    if args.first_method:
        methods.sort(key=lambda m: 0 if m[0] == args.first_method else 1)

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
            print(f"\nSuccess! File saved as: {result}")
            return

    print("\nAll download methods failed. Please check the URL and your internet connection.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dl_yt.py <youtube_url> [--output filename] [--type audio|video|noaudio|transcription] [--first-method yt-dlp|PyTube|youtube-dl]")
        sys.exit(1)
    main()
