from io import BytesIO

import numpy as np
from pypdf import PdfReader

from scanclean import make_pdf, write_pdf


def test_make_pdf_single_page_has_valid_header_and_dimensions():
    page = np.full((100, 200), 255, dtype=np.uint8)
    pdf = make_pdf([page], [300])
    reader = PdfReader(BytesIO(pdf))

    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 100
    assert len(reader.pages) == 1
    assert float(reader.pages[0].mediabox.width) == 48
    assert float(reader.pages[0].mediabox.height) == 24


def test_make_pdf_multiple_pages_and_bilevel():
    pages = [
        np.full((80, 120), 255, dtype=np.uint8),
        np.pad(np.zeros((20, 20), dtype=np.uint8), 20, constant_values=255),
    ]
    pdf = make_pdf(pages, [300, 150], bilevel=True)

    assert len(PdfReader(BytesIO(pdf)).pages) == 2
    assert b"/BitsPerComponent 1" in pdf


def test_transparent_pdf_paths_are_valid():
    page = np.tile(np.arange(64, dtype=np.uint8), (64, 1))

    gray_pdf = make_pdf([page], [300], transparent=True, matte=True)
    bilevel_pdf = make_pdf([page], [300], bilevel=True, transparent=True)

    assert len(PdfReader(BytesIO(gray_pdf)).pages) == 1
    assert b"/SMask" in gray_pdf
    assert len(PdfReader(BytesIO(bilevel_pdf)).pages) == 1
    assert b"/ImageMask true" in bilevel_pdf


def test_write_pdf_writes_exact_make_pdf_bytes(tmp_path):
    page = np.arange(100, dtype=np.uint8).reshape(10, 10)
    destination = tmp_path / "page.pdf"

    write_pdf([page], [300], destination)

    assert destination.read_bytes() == make_pdf([page], [300])
