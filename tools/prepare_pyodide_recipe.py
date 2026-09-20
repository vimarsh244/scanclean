#!/usr/bin/env python3
"""Pin the official Pyodide OpenCV recipe to ScanClean's OpenCV release.

The recipe itself stays owned by Pyodide.  Release CI checks out a pinned
pyodide-recipes commit and this script makes the deliberately small source and
Emscripten port adjustments needed by ScanClean.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "packaging" / "pyodide" / "versions.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path, help="path to opencv-python/meta.yaml")
    args = parser.parse_args()

    config = json.loads(VERSIONS.read_text(encoding="utf-8"))
    opencv = config["opencv"]
    text = args.recipe.read_text(encoding="utf-8")

    old_version = f"  version: {opencv['base_recipe_version']}"
    if old_version not in text:
        raise SystemExit(
            f"Pinned recipe no longer contains {old_version!r}; review it before building"
        )

    source_lines = text.splitlines()
    in_source = False
    found_url = found_hash = found_jpeg_link = found_png_link = False
    found_build_args = False
    output = []
    for line in source_lines:
        if line == "source:":
            in_source = True
        elif line and not line.startswith(" "):
            in_source = False

        if line == old_version:
            line = f"  version: {opencv['version']}"
        elif in_source and line.startswith("  url:"):
            line = f"  url: {opencv['source_url']}"
            found_url = True
        elif in_source and line.startswith("  sha256:"):
            line = f"  sha256: {opencv['source_sha256']}"
            found_hash = True
        elif line.strip() == "-ljpeg":
            # Let Emscripten select the PIC port archive for a SIDE_MODULE.
            # A plain -ljpeg can select the non-PIC sysroot archive on a clean
            # GitHub runner, which wasm-ld correctly refuses to link.
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}-sUSE_LIBJPEG=1"
            found_jpeg_link = True
        elif line.strip() == "-lpng-legacysjlj":
            # This is the port variant compatible with Pyodide's WASM longjmp
            # ABI. Emscripten does not build its PIC archive automatically.
            found_png_link = True
        elif line.strip() == "source $PKGDIR/extras/build_args.sh":
            output.append("    embuilder build libjpeg --pic")
            output.append("    embuilder build libpng-legacysjlj --pic")
            found_build_args = True
        output.append(line)

    if not (
        found_url
        and found_hash
        and found_jpeg_link
        and found_png_link
        and found_build_args
    ):
        raise SystemExit(
            "Could not find the expected OpenCV source, build, or PNG linker fields"
        )

    args.recipe.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"Prepared OpenCV {opencv['version']} recipe at {args.recipe}")


if __name__ == "__main__":
    main()
