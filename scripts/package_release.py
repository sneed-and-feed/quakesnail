#!/usr/bin/env python3
"""Cross-platform release bundler for QuakeSnail (Linux, macOS, Windows).

Creates platform-specific release archives in `dist/`:
  - quakesnail-<version>-linux-<arch>.tar.gz
  - quakesnail-<version>-macos-<arch>.tar.gz
  - quakesnail-<version>-windows-<arch>.zip
along with OS-tagged native Rust SIMD shared libraries.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import stat
import sys
import tarfile
import zipfile
from pathlib import Path


def detect_platform_tag() -> tuple[str, str, str]:
    sys_name = sys.platform.lower()
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine or "x86_64"

    if sys_name.startswith("win"):
        return "windows", arch, "quakesnail_rs.dll"
    elif sys_name == "darwin":
        return "macos", arch, "libquakesnail_rs.dylib"
    else:
        return "linux", arch, "libquakesnail_rs.so"


def package_release(version: str, out_dir: Path) -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    os_name, arch, lib_name = detect_platform_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    ver_tag = version if version.startswith("v") else f"v{version}"
    bundle_name = f"quakesnail-{ver_tag}-{os_name}-{arch}"
    staging_root = out_dir / "_staging" / bundle_name
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    # 1. Copy core files and directories
    for item in ("README.md", "LICENSE", "CITATION.cff", "pyproject.toml"):
        src = repo_root / item
        if src.exists():
            shutil.copy2(src, staging_root / item)

    for folder in ("quakesnail", "studio", "formalization", "docs", "results"):
        src_dir = repo_root / folder
        if src_dir.exists():
            shutil.copytree(
                src_dir,
                staging_root / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "_ci_*"),
            )

    # 2. Bundle compiled #[no_std] Rust SIMD C-ABI library
    created_artifacts: list[Path] = []
    lib_src = repo_root / "rust" / "quakesnail_rs" / "target" / "release" / lib_name
    if lib_src.exists():
        shutil.copy2(lib_src, staging_root / "quakesnail" / lib_name)
        ext = Path(lib_name).suffix
        stem = Path(lib_name).stem
        tagged_lib = out_dir / f"{stem}-{ver_tag}-{os_name}-{arch}{ext}"
        shutil.copy2(lib_src, tagged_lib)
        created_artifacts.append(tagged_lib)

    # 3. Create platform launcher scripts
    sh_launcher = staging_root / "quakesnail-studio.sh"
    sh_launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'export PYTHONPATH="$DIR:${PYTHONPATH}"\n'
        'exec python3 -m quakesnail studio --live --open "$@"\n',
        encoding="utf-8",
    )
    try:
        sh_launcher.chmod(sh_launcher.stat().st_mode | stat.S_IEXEC)
    except OSError:
        pass

    bat_launcher = staging_root / "quakesnail-studio.bat"
    bat_launcher.write_text(
        "@echo off\r\n"
        "setlocal\r\n"
        'set "PYTHONPATH=%~dp0;%PYTHONPATH%"\r\n'
        "python -m quakesnail studio --live --open %*\r\n",
        encoding="utf-8",
    )

    # 4. Create archive (.zip on Windows, .tar.gz on Linux/macOS)
    if os_name == "windows":
        archive_path = out_dir / f"{bundle_name}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(staging_root.rglob("*")):
                if file_path.is_file():
                    arcname = f"{bundle_name}/{file_path.relative_to(staging_root).as_posix()}"
                    zf.write(file_path, arcname)
    else:
        archive_path = out_dir / f"{bundle_name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(staging_root, arcname=bundle_name)

    created_artifacts.append(archive_path)
    shutil.rmtree(out_dir / "_staging", ignore_errors=True)

    for p in created_artifacts:
        print(f"Created release artifact: {p} ({p.stat().st_size:,} bytes)")
    return created_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Package QuakeSnail release bundle.")
    parser.add_argument("--version", default="v0.1.0", help="Release version tag (default: v0.1.0)")
    parser.add_argument("--out-dir", default="dist", help="Output directory (default: dist)")
    args = parser.parse_args()
    package_release(args.version, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
