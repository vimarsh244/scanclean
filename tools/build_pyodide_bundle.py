#!/usr/bin/env python3
"""Assemble an ABI-locked ScanClean Pyodide release archive."""

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "packaging" / "pyodide" / "versions.json"
REQUIRED_PACKAGES = ("numpy", "opencv-python", "pillow")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_metadata(path: Path):
    with zipfile.ZipFile(path) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        return BytesParser().parsebytes(wheel.read(metadata_name))


def locate(root: Path, filename: str) -> Path:
    matches = list(root.rglob(Path(filename).name))
    if len(matches) != 1:
        raise SystemExit(f"Expected one {filename!r} below {root}, found {len(matches)}")
    return matches[0]


def package_entry(name: str, version: str, filename: str, digest: str, depends: list[str]):
    imports = {"scanclean": ["scanclean"], "opencv-python": ["cv2"]}.get(
        name, ["PIL"] if name == "pillow" else [name]
    )
    return {
        "depends": depends,
        "file_name": f"wheels/{filename}",
        "imports": imports,
        "install_dir": "site",
        "name": "Pillow" if name == "pillow" else name,
        "package_type": "package",
        "sha256": digest,
        "tool": {},
        "unvendored_tests": False,
        "version": version,
    }


def normalise_requirement(value: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", value)
    if not match:
        raise SystemExit(f"Cannot parse wheel requirement {value!r}")
    return match.group(0).lower().replace("_", "-")


def add_to_zip(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    info = zipfile.ZipInfo(str(PurePosixPath(name)), (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, source.read_bytes(), compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--wheel-dir", required=True, type=Path)
    parser.add_argument("--scanclean-wheel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    config = json.loads(VERSIONS.read_text(encoding="utf-8"))
    source_lock = json.loads(args.lock.read_text(encoding="utf-8"))
    expected_info = {
        "abi_version": config["pyodide"]["abi"],
        "arch": "wasm32",
        "platform": f"emscripten_{config['pyodide']['emscripten'].replace('.', '_')}",
        "python": config["pyodide"]["python"],
    }
    if source_lock["info"] != expected_info:
        raise SystemExit(
            f"Built lock targets {source_lock['info']!r}, expected {expected_info!r}"
        )

    metadata = wheel_metadata(args.scanclean_wheel)
    version = metadata["Version"]
    if args.tag != f"v{version}":
        raise SystemExit(f"Tag {args.tag!r} does not match ScanClean {version}")
    requirements = {
        normalise_requirement(item) for item in (metadata.get_all("Requires-Dist") or [])
    }
    missing = set(REQUIRED_PACKAGES) - requirements
    if missing:
        raise SystemExit(f"ScanClean wheel is missing runtime requirements: {sorted(missing)}")

    staging = args.output_dir / f"scanclean-pyodide-{args.tag}"
    if staging.exists():
        shutil.rmtree(staging)
    wheels = staging / "wheels"
    wheels.mkdir(parents=True)

    packages = {}
    manifest_files = {}
    for name in REQUIRED_PACKAGES:
        entry = source_lock["packages"].get(name)
        if entry is None:
            raise SystemExit(f"Built lock does not contain {name}")
        expected_version = (
            config["opencv"]["version"]
            if name == "opencv-python"
            else config["packages"][name]
        )
        if entry["version"] != expected_version:
            raise SystemExit(
                f"Built {name} {entry['version']}, expected {expected_version}"
            )
        source = locate(args.wheel_dir, entry["file_name"])
        destination = wheels / source.name
        shutil.copy2(source, destination)
        digest = sha256(destination)
        copied = dict(entry)
        copied["file_name"] = f"wheels/{destination.name}"
        copied["sha256"] = digest
        packages[name] = copied
        manifest_files[copied["file_name"]] = {
            "sha256": digest,
            "size": destination.stat().st_size,
        }

    scanclean_destination = wheels / args.scanclean_wheel.name
    shutil.copy2(args.scanclean_wheel, scanclean_destination)
    scanclean_digest = sha256(scanclean_destination)
    packages["scanclean"] = package_entry(
        "scanclean",
        version,
        scanclean_destination.name,
        scanclean_digest,
        list(REQUIRED_PACKAGES),
    )
    manifest_files[f"wheels/{scanclean_destination.name}"] = {
        "sha256": scanclean_digest,
        "size": scanclean_destination.stat().st_size,
    }

    lock = {"info": expected_info, "packages": packages, "tool": {}}
    lock_path = staging / "pyodide-lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "format": 1,
        "version": version,
        "tag": args.tag,
        "pyodide": {k: config["pyodide"][k] for k in ("version", "python", "emscripten", "abi")},
        "packages": {name: entry["version"] for name, entry in packages.items()},
        "files": manifest_files,
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = staging / "README.txt"
    readme.write_text(
        "ScanClean Pyodide distribution\n\n"
        f"Runtime: Pyodide {config['pyodide']['version']}\n"
        "Pass pyodide-lock.json as loadPyodide({ lockFileURL }), then load "
        "the 'scanclean' package. Run image processing in a Web Worker.\n",
        encoding="utf-8",
    )

    archive_path = args.output_dir / f"scanclean-pyodide-{args.tag}.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for source in sorted(staging.rglob("*")):
            if source.is_file():
                add_to_zip(archive, source, source.relative_to(staging).as_posix())
    print(archive_path)


if __name__ == "__main__":
    main()
