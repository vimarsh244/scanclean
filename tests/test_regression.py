import hashlib
import json
import shutil
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from pypdf import PdfReader

from scanclean import Options, clean_page, make_pdf
from scanclean.cli import page_images


ROOT = Path(__file__).parents[1]
ORIGINAL = ROOT / "original"
BASELINE = json.loads(
    (Path(__file__).parent / "baselines" / "original_metrics.json").read_text(
        encoding="utf-8"
    )
)

pytestmark = [pytest.mark.original, pytest.mark.slow]


def original_case_ids(case):
    return f"{Path(case['file']).stem.lower().replace(' ', '-')}-p{case['page']}"


def read_original_page(case):
    path = ORIGINAL / case["file"]
    if not path.exists():
        pytest.skip(f"local regression fixture is absent: {path}")
    page = PdfReader(path).pages[case["page"] - 1]
    assert len(page.images) == 1
    bgr = cv2.imdecode(
        np.frombuffer(page.images[0].data, np.uint8), cv2.IMREAD_COLOR
    )
    dpi = round(bgr.shape[1] * 72 / float(page.mediabox.width), 3)
    return bgr, dpi


@pytest.mark.parametrize("case", BASELINE["cases"], ids=original_case_ids)
def test_original_page_regression(case):
    bgr, dpi = read_original_page(case)
    output, stats, audit, out_dpi = clean_page(bgr, dpi, Options(audit=True))

    assert list(output.shape) == case["shape"]
    assert out_dpi == case["dpi"]
    assert audit.shape == (*output.shape, 3)
    assert abs(float(output.mean()) - case["mean"]) < 0.5
    assert abs(float(output.std()) - case["std"]) < 0.5
    assert "det_boxes" in stats
    assert int((output < 160).sum()) == pytest.approx(case["dark_pixels"], rel=0.015)
    assert int((output < 224).sum()) == pytest.approx(case["ink_pixels"], rel=0.015)

    height, width = output.shape
    grid = []
    for row in range(4):
        for column in range(4):
            cell = output[
                row * height // 4 : (row + 1) * height // 4,
                column * width // 4 : (column + 1) * width // 4,
            ]
            grid.append(float((cell < 160).mean()))
    assert np.max(np.abs(np.array(grid) - case["grid_dark_fraction"])) < 0.008

    x0, y0, x1, y1 = case["text_crop"]["box"]
    crop = (output[y0:y1, x0:x1] < 160).astype(np.uint8)
    components = cv2.connectedComponentsWithStats(crop, 8)[0] - 1
    assert int(crop.sum()) == pytest.approx(case["text_crop"]["dark_pixels"], rel=0.02)
    assert components == pytest.approx(case["text_crop"]["components"], rel=0.04)

    assert stats["glyph_h"] == pytest.approx(case["stats"]["glyph_h"], abs=2)
    assert stats["glyph_area"] == pytest.approx(case["stats"]["glyph_area"], rel=0.08)
    assert stats["core_removed"] <= max(case["stats"]["core_removed"] + 5, 2)

    input_hash = hashlib.sha256(bgr.tobytes()).hexdigest()
    same_runtime = (
        input_hash == case["input_sha256"]
        and cv2.__version__ == BASELINE["opencv"]
        and np.__version__ == BASELINE["numpy"]
    )
    if same_runtime:
        assert hashlib.sha256(output.tobytes()).hexdigest() == case["output_sha256"]

    pdf = make_pdf([output], [out_dpi])
    rendered = PdfReader(BytesIO(pdf))
    assert pdf.startswith(b"%PDF-")
    assert len(rendered.pages) == 1
    assert float(rendered.pages[0].mediabox.width) > 0
    assert float(rendered.pages[0].mediabox.height) > 0


def test_baselines_cover_every_local_sample_document():
    expected = {f"Sample {number}.pdf" for number in range(1, 8)}
    assert {case["file"] for case in BASELINE["cases"]} == expected


def test_original_pdf_end_to_end(tmp_path):
    source = ORIGINAL / "Sample 1.pdf"
    if not source.exists():
        pytest.skip(f"local regression fixture is absent: {source}")

    reader = PdfReader(source)
    cleaned_pages, dpis = [], []
    for page in reader.pages:
        bgr = cv2.imdecode(
            np.frombuffer(page.images[0].data, np.uint8), cv2.IMREAD_COLOR
        )
        dpi = round(bgr.shape[1] * 72 / float(page.mediabox.width), 3)
        output, _, _, out_dpi = clean_page(bgr, dpi, Options())
        cleaned_pages.append(output)
        dpis.append(out_dpi)

    destination = tmp_path / "sample-1.cleaned.pdf"
    destination.write_bytes(make_pdf(cleaned_pages, dpis))

    assert len(cleaned_pages) == len(reader.pages) == 7
    assert destination.read_bytes().startswith(b"%PDF-")
    assert destination.stat().st_size > 1000
    assert len(PdfReader(destination).pages) == 7


@pytest.mark.skipif(
    not all(shutil.which(command) for command in ("pdfimages", "pdfinfo", "pdftoppm")),
    reason="Poppler command-line tools are not installed",
)
def test_original_pdf_end_to_end_through_desktop_extractor(tmp_path):
    source = ORIGINAL / "Sample 1.pdf"
    if not source.exists():
        pytest.skip(f"local regression fixture is absent: {source}")

    cleaned_pages, dpis = [], []
    for _, bgr, dpi in page_images(source):
        output, _, _, out_dpi = clean_page(bgr, dpi, Options())
        cleaned_pages.append(output)
        dpis.append(out_dpi)
    destination = tmp_path / "sample-1.cleaned.pdf"
    destination.write_bytes(make_pdf(cleaned_pages, dpis))

    assert len(cleaned_pages) == 7
    assert destination.stat().st_size > 1000
    assert len(PdfReader(destination).pages) == 7
