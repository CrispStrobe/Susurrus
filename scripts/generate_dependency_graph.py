#!/usr/bin/env python3
"""Generate visual dependency graph"""

import os
import sys
from pathlib import Path

try:
    import pydeps  # noqa: F401
    from pydeps.cli import main as pydeps_main
except ImportError:
    print("❌ Install pydeps: pip install pydeps")
    sys.exit(1)

project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Check if main package directory exists
main_dirs = ["gui", "utils", "backends", "workers"]
existing_dirs = [d for d in main_dirs if (project_root / d).is_dir()]

if not existing_dirs:
    print("❌ No main package directories found")
    sys.exit(1)

# Generate graph for each main directory
for directory in existing_dirs:
    output_file = f"dependency_graph_{directory}.svg"

    print(f"📊 Generating dependency graph for {directory}...")

    try:
        sys.argv = [
            "pydeps",
            directory,
            "--max-bacon",
            "2",
            "--cluster",
            "-o",
            output_file,
            "--noshow",
        ]

        pydeps_main()
        print(f"✅ Graph saved to {output_file}")

    except Exception as e:
        print(f"⚠️  Could not generate graph for {directory}: {e}")

print("\n✅ Dependency graphs generated!")
