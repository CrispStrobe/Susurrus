#!/usr/bin/env python3
"""
Fix common syntax errors from automated cleanup
"""

import re
from pathlib import Path


def fix_file(filepath: Path):
    """Fix common syntax errors in a file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original = content
        modified = False

        # Fix 1: Unclosed parentheses in imports
        # Find patterns like "from X import (\n    Y,\n    Z\n    ← missing )"
        pattern = r"from\s+[\w.]+\s+import\s+\([^)]+$"
        if re.search(pattern, content, re.MULTILINE):
            # Add missing closing paren before first non-import line
            lines = content.split("\n")
            new_lines = []
            in_import = False
            import_buffer = []

            for i, line in enumerate(lines):
                if line.strip().startswith("from ") and "import (" in line:
                    in_import = True
                    import_buffer = [line]
                elif in_import:
                    if line.strip() and not line.strip().startswith("#"):
                        if ")" not in line and not any(
                            x in line for x in ["import", ",", "from"]
                        ):
                            # End of import block, add closing paren
                            import_buffer[-1] = import_buffer[-1].rstrip() + ")"
                            new_lines.extend(import_buffer)
                            new_lines.append(line)
                            in_import = False
                            import_buffer = []
                        else:
                            import_buffer.append(line)
                    else:
                        import_buffer.append(line)
                else:
                    new_lines.append(line)

            if import_buffer:
                import_buffer[-1] = import_buffer[-1].rstrip() + ")"
                new_lines.extend(import_buffer)

            content = "\n".join(new_lines)
            modified = True

        # Fix 2: Incorrect indentation in except blocks
        # Pattern: "except SomeError as e:" at wrong indentation
        lines = content.split("\n")
        new_lines = []
        prev_indent = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)

            if stripped.startswith("except ") and i > 0:
                # Check if previous line is indented (inside try block)
                prev_line = lines[i - 1].lstrip()
                if prev_line and not prev_line.startswith(("try:", "except ", "else:", "finally:")):
                    # We're inside a try block, need to dedent
                    if current_indent > prev_indent:
                        line = " " * prev_indent + stripped
                        modified = True

            new_lines.append(line)
            if stripped and not stripped.startswith("#"):
                prev_indent = current_indent

        content = "\n".join(new_lines)

        # Write back if modified
        if modified or content != original:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✓ Fixed {filepath}")
            return True

    except Exception as e:
        print(f"⚠️  Error fixing {filepath}: {e}")

    return False


def main():
    project_root = Path(__file__).parent.parent

    print("🔧 Fixing syntax errors...")

    files_to_check = [
        "gui/dialogs/cuda_diagnostics_dialog.py",
        "gui/main_window.py",
        "workers/diarize_worker.py",
    ]

    fixed = 0
    for filepath in files_to_check:
        full_path = project_root / filepath
        if full_path.exists():
            if fix_file(full_path):
                fixed += 1

    print(f"\n✅ Fixed {fixed} files")


if __name__ == "__main__":
    main()