#!/usr/bin/env python3
"""Rewrite tests/baselines/original_metrics.json from the local sample PDFs.

    python3 tools/refresh_baselines.py

The regression test pins a handful of numbers per sample page, plus an exact
output hash for the runtime the numbers were taken on. Those numbers are the
record of what ScanClean produces; when a cleaning stage deliberately changes,
they have to be retaken, and taking them by hand is how a baseline quietly
stops describing the code.

Which page of each document is measured, and which crop within it, are choices
made once and kept: they are read from the existing file and never invented
here, so refreshing the numbers cannot silently move the test's aim. Add a new
document by adding an entry to :data:`PAGES` below.

The PDFs live in the ignored local `original/` directory and are not
distributed; this script skips any that are absent.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pypdf import PdfReader

from scanclean import Options, clean_page

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "original"
BASELINE = ROOT / "tests" / "baselines" / "original_metrics.json"

# document -> (page number, crop box within the cleaned page)
PAGES = {
    "Sample 1.pdf": (6, [250, 300, 950, 650]),
    "Sample 2.pdf": (7, [150, 150, 950, 600]),
    "Sample 3.pdf": (1, [100, 750, 1050, 1300]),
    "Sample 4.pdf": (3, [130, 300, 1000, 700]),
    "Sample 5.pdf": (4, [120, 300, 900, 700]),
    "Sample 6.pdf": (1, [200, 400, 1400, 900]),
    "Sample 7.pdf": (5, [200, 400, 1300, 900]),
}


def read_page(path, number):
    page = PdfReader(path).pages[number - 1]
    assert len(page.images) == 1, f"{path.name} p{number} is not a single scan"
    bgr = cv2.imdecode(np.frombuffer(page.images[0].data, np.uint8), cv2.IMREAD_COLOR)
    return bgr, round(bgr.shape[1] * 72 / float(page.mediabox.width), 3)


def measure(name, number, box):
    bgr, dpi = read_page(ORIGINAL / name, number)
    output, stats, _, out_dpi = clean_page(bgr, dpi, Options(audit=True))

    height, width = output.shape
    grid = []
    for row in range(4):
        for column in range(4):
            cell = output[
                row * height // 4 : (row + 1) * height // 4,
                column * width // 4 : (column + 1) * width // 4,
            ]
            grid.append(round(float((cell < 160).mean()), 8))

    x0, y0, x1, y1 = box
    crop = (output[y0:y1, x0:x1] < 160).astype(np.uint8)
    return {
        "file": name,
        "page": number,
        "input_sha256": hashlib.sha256(bgr.tobytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "dpi": out_dpi,
        "shape": list(output.shape),
        "mean": round(float(output.mean()), 6),
        "std": round(float(output.std()), 6),
        "dark_pixels": int((output < 160).sum()),
        "ink_pixels": int((output < 224).sum()),
        "grid_dark_fraction": grid,
        "text_crop": {
            "box": list(box),
            "dark_pixels": int(crop.sum()),
            "components": int(cv2.connectedComponentsWithStats(crop, 8)[0] - 1),
        },
        "stats": {
            "glyph_h": stats["glyph_h"],
            "glyph_area": stats["glyph_area"],
            "core_removed": stats["core_removed"],
        },
    }


def main():
    previous = {}
    if BASELINE.exists():
        for case in json.loads(BASELINE.read_text(encoding="utf-8"))["cases"]:
            previous[case["file"]] = (case["page"], case["text_crop"]["box"])

    cases = []
    for name, default in PAGES.items():
        if not (ORIGINAL / name).exists():
            print(f"   skipping {name}: not present locally")
            continue
        number, box = previous.get(name, default)
        case = measure(name, number, box)
        cases.append(case)
        print(f"   {name} p{number}: {case['dark_pixels']} dark px, "
              f"{case['text_crop']['components']} components in the crop")

    if len(cases) != len(PAGES):
        raise SystemExit("refusing to write a partial baseline; some samples are absent")

    BASELINE.write_text(
        json.dumps({"opencv": cv2.__version__, "numpy": np.__version__, "cases": cases},
                   indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"-> {BASELINE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
