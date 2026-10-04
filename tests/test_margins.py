"""The margin stages, on synthetic pages.

Every case here is a shape the real scans produce and that used to be cleaned
away as damage: type that runs out to the trim, a column of a table, the side
of a printed border. They are built from rectangles so the test says what it
means without a source scan, which is never committed.
"""

import cv2
import numpy as np
import pytest

from scanclean.core import clear_bands, edge_bands, frame_rules, line_cover, scratches
from scanclean import core as sc


GH = 20.0
GA = 252.0


def ruled_page(text_left, width=400, height=760, lines=10, pitch=60):
    """A page of type: (mask, runs). Each line is a row of glyph blocks."""
    mask = np.zeros((height, width), np.uint8)
    for index in range(lines):
        top = 50 + index * pitch
        for left in range(text_left, width - 20, 30):
            cv2.rectangle(mask, (left, top), (left + 13, top + 17), 255, -1)
    return mask, mask.copy()


def test_line_cover_counts_lines_not_pixels():
    mask, runs = ruled_page(text_left=20)

    cover, lines, _ = line_cover(runs, GH)

    assert lines == 10
    assert cover[:20].max() == 0
    assert cover[200] == 10


def test_edge_bands_stands_down_where_the_lines_run_into_the_band():
    """Dense leading puts ink in the inter-line gaps at a column of ordinary
    type. Left to itself the stage reads that as a strip and condemns every
    first word on the page along with it."""
    mask, runs = ruled_page(text_left=20)
    mask[:, 60:70] = 255                      # ink in every gap, inside the text

    assert edge_bands(mask, runs, GH) == (0, mask.shape[1])


def test_edge_bands_still_finds_a_strip_outside_the_type_area():
    mask, runs = ruled_page(text_left=120)
    mask[:, :40] = 255                        # a strip down the trim, clear of type

    left, right = edge_bands(mask, runs, GH)

    assert 40 <= left <= 60
    assert right == mask.shape[1]


def test_clear_bands_keeps_a_line_initial_word_a_band_cuts_off():
    mask, runs = ruled_page(text_left=20)

    _, removed = clear_bands(mask, runs, GH, left=45, right=mask.shape[1])

    assert not removed.any()


def test_clear_bands_keeps_printed_rules_it_cannot_chain_to_a_line():
    mask, runs = ruled_page(text_left=120)
    rule = np.zeros_like(mask)
    rule[40:700, 60:64] = 255                 # a border side, alone in the margin
    mask = cv2.bitwise_or(mask, rule)

    _, without = clear_bands(mask, runs, GH, left=90, right=mask.shape[1])
    _, keeping = clear_bands(mask, runs, GH, left=90, right=mask.shape[1], keep=rule)

    assert without.any()
    assert not keeping.any()


def test_scratches_spares_a_column_of_type():
    """A table column is a chain many glyphs tall and one wide - the crease
    shape exactly - but it is made of letters, and a crease is not."""
    mask = np.zeros((760, 400), np.uint8)
    for index in range(8):
        top = 50 + index * 50
        cv2.rectangle(mask, (100, top), (113, top + 17), 255, -1)
    runs = np.zeros_like(mask)

    _, removed = scratches(mask, runs, GH, GA, core=np.zeros_like(mask))

    assert not removed.any()


def test_scratches_still_takes_a_shattered_crease():
    mask = np.zeros((760, 400), np.uint8)
    for index in range(16):
        top = 50 + index * 30
        cv2.rectangle(mask, (200, top), (201, top + 5), 255, -1)
    runs = np.zeros_like(mask)

    _, removed = scratches(mask, runs, GH, GA, core=np.zeros_like(mask))

    assert removed.any()


def test_frame_rules_keeps_a_drawn_rule():
    mask = np.zeros((760, 400), np.uint8)
    mask[40:700, 60:64] = 255

    assert frame_rules(mask, GH)[300, 61] == 255


@pytest.mark.parametrize("flaw", ["ragged", "wandering"])
def test_frame_rules_refuses_edge_damage_of_the_same_height(flaw):
    rng = np.random.default_rng(7)
    mask = np.zeros((760, 400), np.uint8)
    for row in range(40, 700):
        if flaw == "ragged":
            start, width = 60, int(rng.integers(2, 18))
        else:
            start, width = 60 + int(rng.integers(0, 14)), 4
        mask[row, start:start + width] = 255

    assert not frame_rules(mask, GH).any()


def test_detected_page_number_survives_the_final_crop(monkeypatch):
    image = np.full((760, 400, 3), 255, np.uint8)
    for top in range(150, 651, 50):
        for left in range(80, 321, 30):
            cv2.rectangle(image, (left, top), (left + 12, top + 17), (0, 0, 0), -1)
    cv2.putText(image, "3", (195, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    detected = np.zeros(image.shape[:2], np.uint8)
    detected[35:80, 185:225] = 255
    detected[140:690, 70:345] = 255
    monkeypatch.setattr(sc, "text_block", lambda *args: (60, 120, 350, 710))
    monkeypatch.setattr(sc, "detect_text", lambda *args: (
        detected, [(185, 35, 40, 45), (70, 140, 275, 550)]))

    output, stats, _, _ = sc.clean_page(image, 300, sc.Options(deskew=False))

    assert stats["box"][1] < 70
    assert (output[35:80, 185:225] < 160).sum() > 50
    assert not (output[:30] < 160).any()


def test_despeckle_keeps_a_faint_fragment_attached_by_its_halo():
    mask = np.zeros((100, 180), np.uint8)
    core = np.zeros_like(mask)
    norm = np.full_like(mask, 255)
    core[40:58, 60:74] = 255
    mask[:] = core
    norm[core > 0] = 40
    # Beyond the direct dilation reach, but connected through faint ink.
    mask[44:49, 82:87] = 255
    norm[44:49, 74:87] = 225
    norm[44:49, 82:87] = 170
    # An equally faint isolated speck must still be removed.
    mask[44:49, 140:145] = 255
    norm[44:49, 140:145] = 170

    output, removed, _ = sc.despeckle(mask, norm, core, np.zeros_like(mask), GH, GA)

    assert output[44:49, 82:87].all()
    assert not removed[44:49, 82:87].any()
    assert removed[44:49, 140:145].all()


def test_erasure_audit_counts_a_page_number_lost_only_to_cropping(monkeypatch):
    from audit_glyphs import erased

    mask = np.zeros((200, 200), np.uint8)
    mask[20:38, 90:104] = 255
    analysis = dict(gh=GH, ga=GA, mask=mask, box=(20, 60, 180, 180),
                    stages={"specks": np.zeros_like(mask)})
    monkeypatch.setattr(sc, "detect_text", lambda *args: (mask, [(90, 20, 14, 18)]))

    marks, _ = erased(np.full((200, 200, 3), 255, np.uint8), 300,
                      sc.Options(), analysis=analysis)

    assert marks == [("crop", 90, 20, 14, 18)]


def test_paper_floor_preserves_connected_faint_strokes_but_whitens_isolated_texture():
    norm = np.full((100, 180), 255, np.uint8)
    ink = np.zeros_like(norm)
    ink[40:58, 60:74] = 255
    norm[ink > 0] = 40
    norm[44:49, 74:90] = 170           # faint stroke missing from the ink mask
    norm[44:49, 140:145] = 170        # equally faint, isolated paper texture
    kill = np.zeros_like(norm)
    kill[40:58, 85:100] = 255

    output = sc.compose(norm, np.zeros_like(norm), (0, 0, 180, 100), ink=ink)
    removed = sc.compose(norm, kill, (0, 0, 180, 100), ink=ink)

    assert output[46, 80] < 200
    assert output[46, 142] == 255
    assert removed[46, 88] > 250       # halo support must not undo a deletion
