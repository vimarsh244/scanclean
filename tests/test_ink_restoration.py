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

    recovered, support = core.recover_faint_text(removed, text, norm, kept, 24, 135)

    assert np.array_equal(recovered[39:53, 29:52], removed[39:53, 29:52])
    assert not recovered[45:49, 80:84].any()
    assert not recovered[removed == 0].any()
    assert not support[norm == 255].any()
    assert not core.recover_faint_text(removed, np.zeros_like(text), norm, kept, 24, 135)[0].any()
    assert not core.recover_faint_text(removed, text, norm, np.zeros_like(kept), 24, 135)[0].any()


@pytest.mark.parametrize("scale", [0.5, 1, 2])
def test_context_recovery_groups_broken_curves_without_painting_the_gaps(scale):
    norm = np.full((120, 180), 255, np.uint8)
    cv2.rectangle(norm, (40, 40), (60, 68), 225, 2)
    # White breaks split the upper and lower curve into separate components.
    norm[52:55, 35:65] = 255
    kept = np.zeros_like(norm)
    kept[67:70, 39:62] = ((norm[67:70, 39:62] < 248) * 255).astype(np.uint8)
    removed = ((norm < 248) & (kept == 0)).astype(np.uint8) * 255
    text = np.full_like(norm, 255)
    norm, kept, removed, text = [cv2.resize(a, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_NEAREST)
                                  for a in (norm, kept, removed, text)]

    recovered, support = core.recover_faint_text(
        removed, text, norm, kept, 24 * scale, 135 * scale * scale)

    assert (recovered > 0).sum() > 0.8 * (removed > 0).sum()
    assert not support[norm == 255].any()


def test_context_recovery_can_restore_a_wholly_faint_letter_on_the_same_line():
    norm = np.full((120, 240), 255, np.uint8)
    kept = np.zeros_like(norm)
    cv2.rectangle(norm, (30, 40), (50, 68), 30, 2)
    kept[norm < 100] = 255
    cv2.rectangle(norm, (75, 40), (95, 68), 225, 2)
    # A similar mark away from the line must not inherit its neighbour's support.
    cv2.rectangle(norm, (170, 85), (190, 113), 225, 2)
    removed = ((norm == 225) * 255).astype(np.uint8)
    text = np.full_like(norm, 255)

    recovered, support = core.recover_faint_text(removed, text, norm, kept, 24, 135)

    assert recovered[39:70, 74:97].any()
    assert not recovered[84:115, 169:192].any()
    assert not support[norm == 255].any()
    blocked = removed.copy()
    recovered, support = core.recover_faint_text(removed, text, norm, kept, 24, 135, blocked)
    assert not recovered.any()
    assert not support[blocked > 0].any()


def test_context_recovery_ignores_grain_inside_text():
    rng = np.random.default_rng(4)
    norm = rng.integers(240, 243, size=(120, 220), dtype=np.uint8)
    text = np.full_like(norm, 255)

    recovered, support = core.recover_faint_text(text, text, norm, text, 24, 135)

    assert not recovered.any()
    assert not support.any()


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
    # Selective recovery must improve faint strokes without requiring the
    # blanket darkening that previously thickened already-legible print.
    assert (after < 160).sum() > 1.15 * (before < 160).sum()
    assert ((before >= 245) & (after < 224)).sum() > 500
    assert np.array_equal(enhanced[:90], previous[:90])
    # The top and middle of "દૂ" in "દૂર કર્યું હતું" used to be removed as
    # specks even with restoration enabled. Check those strokes specifically.
    assert (enhanced[1186:1194, 355:370] < 160).sum() >= 20
    assert (enhanced[1196:1203, 355:369] < 160).sum() >= 15
    # The disconnected inner curve in "રહેવું" must survive too.
    assert (enhanced[1052:1057, 393:404] < 160).sum() >= 18
    # Recovery should not amplify the tiny scattered peaks seen in this crop.
    weak = enhanced[975:1240, 315:535]
    _, _, components, _ = cv2.connectedComponentsWithStats((weak < 160).astype(np.uint8), 8)
    assert np.count_nonzero(components[1:, cv2.CC_STAT_AREA] <= 3) <= 20


@pytest.mark.parametrize("darkness", [20, 60, 100])
def test_restoration_does_not_thicken_dark_dots_or_vowel_marks(darkness):
    norm = np.full((120, 180), 255, np.uint8)
    cv2.circle(norm, (40, 35), 3, darkness, -1, cv2.LINE_AA)
    cv2.line(norm, (80, 30), (80, 53), darkness, 2, cv2.LINE_AA)
    norm = cv2.GaussianBlur(norm, (3, 3), 0.6)
    enhanced = core.restore_ink(norm, np.full_like(norm, 255), np.zeros_like(norm), 24)

    # Antialiased rims must retain their original width as well as their core.
    assert np.array_equal(enhanced < 224, norm < 224)
    assert np.array_equal(enhanced[norm < 160], norm[norm < 160])


def test_fade_gate_ignores_dark_rims_but_develops_weak_strokes():
    norm, text, kill = faint_page()
    enhanced = core.restore_ink(norm, text, kill, 24)

    assert np.array_equal(enhanced[35:70, 95:125], norm[35:70, 95:125])
    assert enhanced[50, 30] < 170


def test_restoration_does_not_amplify_isolated_faint_grain_peaks():
    norm = np.full((150, 260), 255, np.uint8)
    cv2.rectangle(norm, (30, 50), (60, 85), 220, 2)
    xs = [90, 115, 140, 165, 190, 215]
    norm[60, xs] = [180, 190, 200, 210, 220, 230]
    enhanced = core.restore_ink(norm, np.full_like(norm, 255), np.zeros_like(norm), 24)

    assert np.array_equal(enhanced[60, xs], norm[60, xs])
    assert enhanced[65, 30] < 170  # coherent ink is developed at the same time


def test_restoration_repairs_grey_pinholes_without_filling_white_counters():
    norm = np.full((120, 180), 255, np.uint8)
    cv2.rectangle(norm, (40, 40), (70, 80), 60, 5)
    norm[55, 40] = 225  # pale dropout inside a strong stroke
    norm[65, 40] = 255  # genuinely white opening must remain white
    norm[50, 44] = 225  # grey outside rim without surrounding ink
    text = np.full_like(norm, 255)
    kill = np.zeros_like(norm)
    enhanced = core.restore_ink(norm, text, kill, 24)
    half = core.restore_ink(norm, text, kill, 24, 0.5)

    assert enhanced[55, 40] == 100
    assert enhanced[55, 40] < half[55, 40] < norm[55, 40]
    assert enhanced[50, 44] == norm[50, 44]
    assert np.array_equal(enhanced[norm == 255], norm[norm == 255])
    assert np.array_equal(enhanced[norm < 160], norm[norm < 160])
    kill[55, 40] = 255
    assert core.restore_ink(norm, text, kill, 24)[55, 40] == 225
    text[:] = 0
    assert np.array_equal(core.restore_ink(norm, text, kill, 24), norm)


@pytest.mark.parametrize("scale", [1, 2])
def test_restoration_keeps_a_weak_stroke_continuous_across_pale_sections(scale):
    norm = np.full((150, 180), 255, np.uint8)
    cv2.rectangle(norm, (40, 50), (70, 85), 220, 2)
    # Periodic pale sections formerly looked like holes beside dark fragments.
    for y in (53, 58, 63, 68, 73, 78):
        norm[y, 40:42] = 233
    norm = cv2.resize(norm, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    enhanced = core.restore_ink(norm, np.full_like(norm, 255), np.zeros_like(norm), 24 * scale)

    assert np.all(enhanced[53 * scale:79 * scale, 40 * scale] < 170)
    assert np.array_equal(enhanced[norm == 255], norm[norm == 255])
    assert enhanced[65 * scale, 55 * scale] == 255  # hollow letter stays hollow
