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
tests for all four development sample documents. When those PDFs are absent,
pytest skips only the tests that need them; no source PDF or processed version
is committed. Small numerical baseline records remain in the repository.

## Browser note

The core package is pure Python and keeps desktop PDF extraction outside the
image-processing module, so it can later be loaded into browser Python runtimes
such as Pyodide. Browser integration is not part of this repository yet.

See [experiments tried and implementation details](docs/experiments_tried.md)
for the reasoning behind the conservative cleaning stages.
