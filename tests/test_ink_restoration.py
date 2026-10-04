"""Ink enhancement must darken measured strokes without painting white gaps."""

import cv2
import numpy as np
import pytest

from scanclean import Options, clean_page
from scanclean import core
from scanclean.cli import build_parser


def faint_page():
    norm = np.full((120, 220), 255, np.uint8)
    # Two faint hollow letters separated by a clean inter-letter gap.
    cv2.rectangle(norm, (30, 40), (50, 65), 220, 2)
    cv2.rectangle(norm, (65, 40), (85, 65), 220, 2)
    cv2.rectangle(norm, (100, 40), (120, 65), 20, 2)
    # Same shape outside the confirmed text area.
    cv2.rectangle(norm, (160, 40), (180, 65), 220, 2)
    text = np.zeros_like(norm)
    text[25:80, 20:140] = 255
    return norm, text, np.zeros_like(norm)


def test_enhancement_darkens_faint_strokes_without_filling_openings_or_gaps():
    norm, text, kill = faint_page()
    enhanced = core.restore_ink(norm, text, kill, 24)

    assert enhanced[50, 30] < 170
    assert np.array_equal(enhanced[norm == 255], norm[norm == 255])
    assert enhanced[50, 40] == 255  # letter opening
    assert enhanced[50, 58] == 255  # inter-letter gap
    assert np.array_equal(enhanced[:, 150:], norm[:, 150:])
    before = core.compose(norm, kill, (0, 0, 220, 120))
    after = core.compose(enhanced, kill, (0, 0, 220, 120))
    assert before[50, 100] == after[50, 100] == 0


def test_enhancement_ignores_flat_paper_and_shallow_grain():
    rng = np.random.default_rng(4)
    norm = rng.integers(240, 243, size=(120, 220), dtype=np.uint8)
    text = np.full_like(norm, 255)

    enhanced = core.restore_ink(norm, text, np.zeros_like(norm), 24)

    assert np.array_equal(enhanced, norm)


def test_enhancement_respects_explicit_deletions_and_does_not_mutate_inputs():
    norm, text, kill = faint_page()
    kill[35:70, 25:55] = 255
    previous = norm.copy()

    enhanced = core.restore_ink(norm, text, kill, 24)

    assert np.array_equal(enhanced[kill > 0], norm[kill > 0])
    assert enhanced[50, 65] < norm[50, 65]
    assert np.array_equal(norm, previous)


def test_strength_zero_disables_restoration_and_strength_is_monotonic():
    norm, text, kill = faint_page()
    disabled = core.restore_ink(norm, text, kill, 24, strength=0)
    medium = core.restore_ink(norm, text, kill, 24, strength=0.5)
    full = core.restore_ink(norm, text, kill, 24, strength=1)

    assert np.array_equal(disabled, norm)
    assert np.all(full <= medium)
    assert np.all(medium <= norm)
    assert full[50, 30] < medium[50, 30] < norm[50, 30]


def test_faint_glyph_rescue_groups_fragments_with_a_surviving_stroke():
    norm = np.full((120, 220), 255, np.uint8)
    cv2.rectangle(norm, (30, 40), (50, 65), 220, 2)
    removed = np.zeros_like(norm)
    kept = np.zeros_like(norm)
    # The upper half failed the darkness test; only the bottom stroke survived.
    removed[39:53, 29:52] = ((norm[39:53, 29:52] < 236) * 255).astype(np.uint8)
    kept[64:67, 29:52] = ((norm[64:67, 29:52] < 236) * 255).astype(np.uint8)
    text = np.full_like(norm, 255)
    # An equally faint isolated speck in the same text box must stay deleted.
    norm[45:49, 80:84] = 220
    removed[45:49, 80:84] = 255

    recovered = core.rescue_faint_glyphs(removed, text, norm, kept, 24, 135)

    assert np.array_equal(recovered[39:53, 29:52], removed[39:53, 29:52])
    assert not recovered[45:49, 80:84].any()
    assert not recovered[removed == 0].any()
    assert not core.rescue_faint_glyphs(removed, np.zeros_like(text), norm, kept, 24, 135).any()
    assert not core.rescue_faint_glyphs(removed, text, norm, np.zeros_like(kept), 24, 135).any()


def test_clean_page_applies_restoration_before_tonal_clipping(monkeypatch):
    norm, text, kill = faint_page()
    mask = ((norm < 224) * 255).astype(np.uint8)
    analysis = dict(norm=norm, mask=mask, kill=kill, text=text, gh=24,
                    box=(0, 0, 220, 120), stats={})
    monkeypatch.setattr(core, "analyse", lambda *args: dict(analysis, stats={}))
    image = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    previous, _, _, _ = clean_page(image, 300, Options(sharpen=0))
    enhanced, stats, audit, _ = clean_page(
        image, 300, Options(restore_ink=1, sharpen=0, audit=True))

    assert previous[50, 30] > 220
    assert enhanced[50, 30] < 160
    assert stats["ink_enhanced_px"] > 0
    assert audit[50, 30, 0] == norm[50, 30]  # audit keeps the measured source
    assert enhanced[50, 40] == enhanced[50, 58] == 255


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_api_strength_is_rejected_even_on_a_blank_page(strength):
    with pytest.raises(ValueError, match="between 0 and 1"):
        clean_page(np.full((30, 30), 255, np.uint8), opts=Options(restore_ink=strength))


def test_cli_ink_restoration_defaults_and_adjustment():
    parser = build_parser()
    assert parser.parse_args(["input.pdf"]).restore_ink == Options().restore_ink == 0
    assert parser.parse_args(["input.pdf", "--restore-ink"]).restore_ink == 1
    assert parser.parse_args(["input.pdf", "--restore-ink", "0.5"]).restore_ink == 0.5
    with pytest.raises(SystemExit):
        parser.parse_args(["input.pdf", "--restore-ink", "1.5"])


@pytest.mark.original
@pytest.mark.slow
def test_sample5_page2_faint_strokes_become_visible_without_darkening_blank_margins():
    from pathlib import Path
    from pypdf import PdfReader

    path = Path(__file__).resolve().parents[1] / 'original' / 'Sample 5.pdf'
    if not path.exists():
        pytest.skip('local Sample 5 regression fixture is absent')
    page = PdfReader(path).pages[1]
    image = cv2.imdecode(np.frombuffer(page.images[0].data, np.uint8), cv2.IMREAD_COLOR)
    dpi = round(image.shape[1] * 72 / float(page.mediabox.width), 3)
    previous, _, _, _ = clean_page(image, dpi, Options())
    enhanced, _, _, _ = clean_page(image, dpi, Options(restore_ink=1))
    before = previous[900:1400, 60:860]
    after = enhanced[900:1400, 60:860]

    # This is the faint-word region in the user's first screenshot. More of
    # its existing strokes must be readable, rather than merely counting the
    # same glyph components that survived analysis in both versions.
    assert (after < 160).sum() > 1.25 * (before < 160).sum()
    assert ((before >= 245) & (after < 224)).sum() > 500
    assert np.array_equal(enhanced[:90], previous[:90])
    # The top and middle of "દૂ" in "દૂર કર્યું હતું" used to be removed as
    # specks even with restoration enabled. Check those strokes specifically.
    assert (enhanced[1186:1194, 355:370] < 160).sum() >= 20
    assert (enhanced[1196:1203, 355:369] < 160).sum() >= 15
