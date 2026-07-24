"""Release packaging regression checks."""

from pathlib import Path


def test_pyinstaller_collects_local_gui_packages():
    """The frozen release must include top-level local packages."""
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "--paths=." in workflow
    for package in ("gui", "utils", "workers", "backends"):
        assert f"--collect-submodules={package}" in workflow
