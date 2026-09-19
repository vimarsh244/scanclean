import cv2
import numpy as np
import pytest

from scanclean import Options, clean_page
from scanclean.core import flatten, glyph_metrics, ink_mask, luminance


def conservative_options(**changes):
    values = dict(detect=False, deskew=False, no_crop=True)
    values.update(changes)
    return Options(**values)


def test_luminance_leaves_neutral_ink_and_lifts_colour():
    bgr = np.array([[[30, 30, 30], [40, 40, 180]]], dtype=np.uint8)
    result, chroma = luminance(bgr)

    assert result[0, 0] == 30
    assert result[0, 1] > 180
    assert chroma.tolist() == [[0, 140]]


def test_mask_and_glyph_metrics_find_repeated_print_shapes():
    page = np.full((180, 320), 245, dtype=np.uint8)
    for y in (45, 85, 125):
        for x in range(45, 276, 38):
            cv2.rectangle(page, (x, y), (x + 15, y + 13), 20, -1)
    normalized = flatten(page, 1.0)
    mask = ink_mask(normalized, 1.0)
    glyph_height, glyph_area, *_ = glyph_metrics(mask)

    assert 10 <= glyph_height <= 20
    assert glyph_area > 100


def test_tiny_marks_near_glyphs_survive_while_remote_dust_does_not():
    page = np.full((300, 500, 3), 245, dtype=np.uint8)
    marks = []
    for y in (70, 120, 170, 220):
        for x in range(80, 421, 40):
            cv2.rectangle(page, (x, y), (x + 18, y + 16), (25, 25, 25), -1)
            cv2.line(page, (x, y + 8), (x + 25, y + 8), (25, 25, 25), 2)
            cv2.circle(page, (x + 8, y - 7), 2, (25, 25, 25), -1)
            marks.append((x + 8, y - 7))
    dust = [(10, 10), (470, 20), (20, 280), (480, 270)]
    for point in dust:
        cv2.circle(page, point, 1, (80, 80, 80), -1)

    output, stats, _, _ = clean_page(
        page, 300, conservative_options(sharpen=0)
    )

    assert all(np.count_nonzero(output[y - 2 : y + 3, x - 2 : x + 3] < 160) >= 9 for x, y in marks)
    assert all(np.count_nonzero(output[y - 2 : y + 3, x - 2 : x + 3] < 160) == 0 for x, y in dust)
    assert stats["core_removed"] == 0


@pytest.mark.parametrize(
    "page",
    [
        np.full((128, 160, 3), 255, dtype=np.uint8),
        np.zeros((128, 160, 3), dtype=np.uint8),
        np.full((8, 8, 3), 255, dtype=np.uint8),
        np.full((128, 160), 255, dtype=np.uint8),
        np.full((1800, 2400, 3), 255, dtype=np.uint8),
    ],
    ids=["white", "black", "very-small", "grayscale", "high-resolution"],
)
def test_edge_case_pages_do_not_crash(page):
    output, stats, audit, out_dpi = clean_page(page, 300, conservative_options())

    assert output.shape == page.shape[:2]
    assert output.size > 0
    assert stats
    assert audit is None
    assert out_dpi == 300


def test_colour_page_with_one_component_and_audit():
    page = np.full((128, 160, 3), 255, dtype=np.uint8)
    page[60:64, 78:82] = (0, 0, 0)
    output, stats, audit, out_dpi = clean_page(
        page, 240, conservative_options(audit=True)
    )

    assert output.shape == (128, 160)
    assert stats["glyph_h"] > 0
    assert stats["glyph_area"] > 0
    assert stats["core_removed"] >= 0
    assert audit.shape == (128, 160, 3)
    assert out_dpi == 240
