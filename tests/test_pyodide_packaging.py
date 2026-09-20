import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import scanclean


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "packaging" / "pyodide" / "versions.json"


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / name), *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_prepare_pyodide_recipe_updates_only_pinned_source(tmp_path):
    config = json.loads(VERSIONS.read_text(encoding="utf-8"))
    recipe = tmp_path / "meta.yaml"
    recipe.write_text(
        "package:\n"
        "  name: opencv-python\n"
        f"  version: {config['opencv']['base_recipe_version']}\n"
        "source:\n"
        "  url: https://example.invalid/old.tar.gz\n"
        "  sha256: old\n"
        "requirements:\n"
        "  run:\n"
        "    - numpy\n"
        "build:\n"
        "  ldflags: |\n"
        "    -lpng-legacysjlj\n"
        "  script: |\n"
        "    source $PKGDIR/extras/build_args.sh\n",
        encoding="utf-8",
    )

    run_script("prepare_pyodide_recipe.py", recipe)
    result = recipe.read_text(encoding="utf-8")

    assert f"version: {config['opencv']['version']}" in result
    assert config["opencv"]["source_url"] in result
    assert config["opencv"]["source_sha256"] in result
    assert "    - numpy" in result
    assert "    -lpng-legacysjlj\n" in result
    assert "    embuilder build libpng-legacysjlj --pic\n" in result
    assert result.index("embuilder build") < result.index("source $PKGDIR")


def make_wheel(path, name, version, requirements=()):
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
    ]
    metadata.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    distribution = name.replace("-", "_")
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"{distribution}-{version}.dist-info/METADATA", "\n".join(metadata) + "\n"
        )


def test_build_pyodide_bundle_is_locked_and_reproducible(tmp_path):
    config = json.loads(VERSIONS.read_text(encoding="utf-8"))
    wheel_dir = tmp_path / "built"
    wheel_dir.mkdir()
    package_versions = {
        "numpy": config["packages"]["numpy"],
        "opencv-python": config["opencv"]["version"],
        "pillow": config["packages"]["pillow"],
    }
    packages = {}
    for name, version in package_versions.items():
        filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        make_wheel(wheel_dir / filename, name, version)
        packages[name] = {
            "depends": ["numpy"] if name == "opencv-python" else [],
            "file_name": filename,
            "imports": ["cv2"] if name == "opencv-python" else [name],
            "install_dir": "site",
            "name": name,
            "package_type": "package",
            "sha256": "replaced-by-bundler",
            "tool": {},
            "unvendored_tests": False,
            "version": version,
        }

    lock = tmp_path / "pyodide-lock.json"
    lock.write_text(
        json.dumps(
            {
                "info": {
                    "abi_version": config["pyodide"]["abi"],
                    "arch": "wasm32",
                    "platform": "emscripten_5_0_3",
                    "python": config["pyodide"]["python"],
                },
                "packages": packages,
                "tool": {},
            }
        ),
        encoding="utf-8",
    )
    scanclean_wheel = tmp_path / f"scanclean-{scanclean.__version__}-py3-none-any.whl"
    make_wheel(
        scanclean_wheel,
        "scanclean",
        scanclean.__version__,
        ("numpy>=2.0", "opencv-python>=5,<6", "Pillow>=10"),
    )
    output = tmp_path / "out"
    output.mkdir()
    args = (
        "--lock",
        lock,
        "--wheel-dir",
        wheel_dir,
        "--scanclean-wheel",
        scanclean_wheel,
        "--output-dir",
        output,
        "--tag",
        f"v{scanclean.__version__}",
    )

    run_script("build_pyodide_bundle.py", *args)
    archive = output / f"scanclean-pyodide-v{scanclean.__version__}.zip"
    first_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    run_script("build_pyodide_bundle.py", *args)

    assert hashlib.sha256(archive.read_bytes()).hexdigest() == first_hash
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert {"manifest.json", "pyodide-lock.json", "README.txt"} <= names
        assert len([name for name in names if name.startswith("wheels/")]) == 4
        bundled_lock = json.loads(bundle.read("pyodide-lock.json"))
        assert set(bundled_lock["packages"]) == {
            "numpy",
            "opencv-python",
            "pillow",
            "scanclean",
        }
        assert bundled_lock["packages"]["scanclean"]["depends"] == [
            "numpy",
            "opencv-python",
            "pillow",
        ]
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["version"] == scanclean.__version__
        for filename, record in manifest["files"].items():
            assert hashlib.sha256(bundle.read(filename)).hexdigest() == record["sha256"]
