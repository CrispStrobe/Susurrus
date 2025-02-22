import sys
import os
from urllib.error import HTTPError
import argparse
from typing import Optional
import re

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\u0000-\u001F\u007F-\u009F]', '_', filename)
    sanitized = sanitized.strip('. ')
    return sanitized

def download_with_yt_dlp(url: str) -> Optional[str]:
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
            # Add custom headers to avoid 403 errors
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info['title'])
            output_mp3 = f"{title}.mp3"
            return output_mp3
    except Exception as e:
        print(f"yt-dlp attempt failed: {str(e)}")
        return None

def download_with_pytube(url: str) -> Optional[str]:
    try:
        from pytube import YouTube
        from pydub import AudioSegment
        
        # Add custom headers to avoid 403 errors
        YouTube.bypass_age_gate = lambda s: True
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        output_filename = sanitize_filename(yt.title)
        output_file = audio_stream.download(filename=output_filename)
        output_mp3 = os.path.splitext(output_file)[0] + '.mp3'
        
        audio = AudioSegment.from_file(output_file)
        audio.export(output_mp3, format='mp3')
        
        os.remove(output_file)
        return output_mp3
    except Exception as e:
        print(f"PyTube attempt failed: {str(e)}")
        return None

def download_with_youtube_dl(url: str) -> Optional[str]:
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
            # Add custom headers to avoid 403 errors
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = sanitize_filename(info['title'])
            output_mp3 = f"{title}.mp3"
            return output_mp3
    except Exception as e:
        print(f"youtube-dl attempt failed: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download YouTube video as MP3')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--output', '-o', help='Output filename (without extension)')
    args = parser.parse_args()

    # Changed order to try yt-dlp first
    methods = [
        ('yt-dlp', download_with_yt_dlp),
        ('PyTube', download_with_pytube),
        ('youtube-dl', download_with_youtube_dl)
    ]
    
    for method_name, method in methods:
        print(f"\nAttempting download with {method_name}...")
        result = method(args.url)
        if result:
            if args.output:
                new_filename = f"{sanitize_filename(args.output)}.mp3"
                try:
                    os.rename(result, new_filename)
                    result = new_filename
                except OSError as e:
                    print(f"Warning: Could not rename file: {e}")
            print(f"\nSuccess! Audio saved to: {result}")
            return
    
    print("\nAll download methods failed. Please check the URL and your internet connection.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dl_yt_mp3.py <youtube_url> [--output filename]")
        sys.exit(1)
    main()
