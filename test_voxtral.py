#!/usr/bin/env python3
# test_voxtral.py - Test Voxtral integration

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")


def test_voxtral_local(audio_file, language=None):
    """Test local Voxtral inference"""
    print("\n" + "=" * 60)
    print("Testing Voxtral Local Inference")
    print("=" * 60)

    try:
        from voxtral_local import VoxtralLocal

        voxtral = VoxtralLocal(device="auto")

        print(f"\nTranscribing: {audio_file}")
        if language:
            print(f"Language: {language}")

        segments = voxtral.transcribe(audio_file, language=language)

        print(f"\n✅ Transcription complete! Found {len(segments)} segments:\n")

        for i, segment in enumerate(segments, 1):
            print(f"Segment {i}:")
            print(f"  [{segment['start']:.2f}s -> {segment['end']:.2f}s]")
            print(f"  {segment['text']}\n")

        voxtral.unload_model()
        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nPlease run: ./install_voxtral.sh")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_voxtral_api(audio_file, api_key=None, language=None):
    """Test Voxtral API inference"""
    print("\n" + "=" * 60)
    print("Testing Voxtral API Inference")
    print("=" * 60)

    try:
        from voxtral_api import VoxtralAPI

        if not api_key:
            api_key = os.environ.get("MISTRAL_API_KEY")

        if not api_key:
            print("\n❌ No API key provided")
            print("Set MISTRAL_API_KEY environment variable or pass --api-key")
            print("Get your key from: https://console.mistral.ai/")
            return False

        voxtral = VoxtralAPI(api_key=api_key)

        print(f"\nTranscribing: {audio_file}")
        if language:
            print(f"Language: {language}")

        segments = voxtral.transcribe(audio_file, language=language)

        print(f"\n✅ Transcription complete! Found {len(segments)} segments:\n")

        for i, segment in enumerate(segments, 1):
            print(f"Segment {i}:")
            print(f"  [{segment['start']:.2f}s -> {segment['end']:.2f}s]")
            print(f"  {segment['text']}\n")

        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Voxtral integration")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--backend", choices=["local", "api", "both"], default="both", help="Backend to test"
    )
    parser.add_argument("--api-key", help="Mistral API key (for API test)")
    parser.add_argument("--language", help="Language code (en, fr, es, de, it, pt, pl, nl)")

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        sys.exit(1)

    success = True

    if args.backend in ["local", "both"]:
        if not test_voxtral_local(args.audio_file, args.language):
            success = False

    if args.backend in ["api", "both"]:
        if not test_voxtral_api(args.audio_file, args.api_key, args.language):
            success = False

    if success:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
