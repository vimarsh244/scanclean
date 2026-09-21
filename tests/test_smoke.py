import ast
import importlib
import subprocess
import sys
from functools import reduce
from pathlib import Path

import cv2


def test_public_imports():
    import scanclean
    from scanclean import Options, clean_page, make_pdf, write_pdf

    assert scanclean.__version__ == "0.1.2"
    assert all(callable(item) for item in (Options, clean_page, make_pdf, write_pdf))


def test_import_does_not_execute_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    for name in list(sys.modules):
        if name == "scanclean" or name.startswith("scanclean."):
            del sys.modules[name]

    package = importlib.import_module("scanclean")

    assert callable(package.clean_page)
    assert calls == []
    assert "scanclean.cli" not in sys.modules


def test_all_used_opencv_apis_exist():
    import scanclean.core as core

    tree = ast.parse(Path(core.__file__).read_text(encoding="utf-8"))
    paths = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parts = [node.attr]
        value = node.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name) and value.id == "cv2":
            paths.add(tuple(reversed(parts)))

    missing = []
    for path in sorted(paths):
        try:
            reduce(getattr, path, cv2)
        except AttributeError:
            missing.append("cv2." + ".".join(path))
    assert not missing, f"OpenCV is missing APIs used by ScanClean: {missing}"


def test_opencv_5_loads_bundled_detector():
    import scanclean.core as core

    major = int(cv2.__version__.split(".", 1)[0])
    assert major == 5, f"ScanClean requires OpenCV 5, found {cv2.__version__}"
    detector = cv2.dnn.readNetFromONNX(str(core.MODEL))
    assert not detector.empty()
