# ScanClean

ScanClean is a classical image-processing pipeline for cleaning degraded
scanned documents while conservatively preserving text. It is tuned for
Gujarati and other Indic documents where legitimate dots and vowel signs can
be the same size as dust. The cleaning pipeline uses NumPy and OpenCV; it does
not use machine learning to generate or repaint document pixels.

## Installation

Install the package and its Python dependencies from this repository:

```bash
python -m pip install .
```

ScanClean requires OpenCV 5 so the bundled ONNX text detector is always
available. Install only one OpenCV wheel variant in an environment; the package
uses `opencv-python` by default.

For development and tests:

```bash
python -m pip install -e ".[test]"
```

Published releases can eventually be installed with `pip install scanclean`.

## Python API

`clean_page()` accepts an OpenCV BGR or grayscale NumPy array. It does not
read files or invoke desktop tools.

```python
import cv2

from scanclean import Options, clean_page, write_pdf

image = cv2.imread("page.png", cv2.IMREAD_COLOR)
cleaned, stats, audit, out_dpi = clean_page(
    image,
    dpi=300,
    opts=Options(audit=True),
)

write_pdf([cleaned], [out_dpi], "cleaned.pdf")
```

Use `make_pdf()` instead of `write_pdf()` when the caller needs PDF bytes.

## Command line

The desktop CLI accepts one or more PDFs and preserves the existing options:

```bash
scanclean input.pdf -o cleaned
scanclean input.pdf -o cleaned --audit
scanclean input.pdf -o cleaned --upscale 2 --bilevel
```

The PDF input path requires the Poppler commands `pdfimages`, `pdfinfo`, and
`pdftoppm` on `PATH`. Poppler is not needed when calling `clean_page()` with an
already-decoded image.

## Tests and builds

```bash
python -m pytest -v
python -m build
```

The ignored local `original/` directory enables full-resolution regression
tests for all seven development sample documents. When those PDFs are absent,
pytest skips only the tests that need them; no source PDF or processed version
is committed. Small numerical baseline records remain in the repository, and
`python tools/refresh_baselines.py` retakes them when a cleaning stage changes
on purpose.

The margin rules are also covered by `tests/test_margins.py`, which builds the
shapes that used to be cleaned away — type running out to the trim, a column of
a table, the side of a printed border — out of rectangles, so those cases are
tested without a source scan.

## Browser / Pyodide

Tagged releases include `scanclean-pyodide-vX.Y.Z.zip`, an ABI-locked browser
distribution containing ScanClean, NumPy, Pillow, and a custom OpenCV 5 wheel.
Use the included `pyodide-lock.json` with the Pyodide version recorded in its
manifest. OpenCV 5 is built because the upstream OpenCV 4 Pyodide package cannot
execute ScanClean's bundled PP-OCRv6 ONNX detector.

The browser distribution still accepts decoded pixel arrays: PDF decoding,
encoding, previews, ZIP creation, and downloads remain responsibilities of the
calling application. Run ScanClean in a Web Worker because image cleanup is
CPU-intensive and Pyodide otherwise executes on the UI thread.

See [experiments tried and implementation details](docs/experiments_tried.md)
for the reasoning behind the conservative cleaning stages.

## License

ScanClean is licensed under the GNU Affero General Public License v3.0.
