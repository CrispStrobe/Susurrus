#!/usr/bin/env python3
"""
Thin diarization worker script - delegates to DiarizationCoordinator
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.diarization_coordinator import main

if __name__ == "__main__":
    main()
