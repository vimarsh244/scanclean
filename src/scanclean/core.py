#!/usr/bin/env python3
"""
ScanClean core - print-ready cleanup for scanned books, tuned for Indic scripts.

Design rule: text is sacred. Diacritics (anusvara, nukta, matra dots) in these
scans are 3-6 px wide, i.e. exactly the size of the dust speckles we want gone.
So noise is NEVER removed on size alone. A small blob is removed only when it
sits outside the "protection zone" grown around real glyph bodies. The failure
mode is therefore "a speck survives", never "a bindi disappeared".

Pipeline per page (see README.md for the reasoning behind each):
  0. deskew                     -> projection-profile, before any analysis
  1. max-channel luminance      -> drops coloured ink (library stamps) + yellowing
  2. background estimation      -> flattens paper to uniform white
  3. Sauvola binarisation       -> ink mask (analysis only, never the output)
  4. border/dust/blot bodies    -> things no glyph can be, removed before text
                                   detection so they cannot be protected as text
  5. text lines + margins       -> RLSA line membership; tears, creases, blots
  6. contextual despeckle       -> the careful bit (see above)
  7. text detector (optional)   -> PP-OCRv6 tiny via cv2.dnn: rescue marks inside
                                   detected lines, clear ink outside the local
                                   column that no line owns
  8. tonal composition          -> white paper, full-greyscale antialiased text
  9. PDF assembly               -> lossless FlateDecode, own writer
"""

import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Options:
    """Options for :func:`clean_page`, matching the command-line defaults."""

    deskew: bool = True
    detect: bool = True
    paper_floor: bool = True
    no_crop: bool = False
    keep_colour: bool = False
    band: float = 0.09
    ink_floor: float = 130
    max_stamp: float = 0.04
    chroma: float = 42
    colour_kill: float = 1.4
    black: float = 40
    white: float = 224
    speck: float = 0.55
    up: float = 0.85
    down: float = 0.65
    side: float = 0.55
    soften: float = 1.0
    sharpen: float = 0.8
    upscale: int = 1
    faint: float = 150
    bilevel: bool = False
    audit: bool = False


# ---------------------------------------------------------------- stages

def luminance(bgr, drop_colour=True, colour_kill=1.4):
    """Max-channel luminance.

    Printing ink on these pages is neutral: measured saturation of body text is
    2-5, while the violet library stamp runs 44-77. So take max(R,G,B) - which
    already lifts any saturated colour - then lift it further in proportion to
    chroma. Black type is untouched; the stamp ring goes to paper white.

    Note this spares a stamp's hand-written accession number when that is in
    black ink, which is correct: neutral hand-writing is indistinguishable from
    type by colour, and guessing would put real marks at risk.
    """
    if not drop_colour:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None
    f = bgr.astype(np.float32)
    mx = f.max(axis=2)
    chroma = mx - f.min(axis=2)
    # Gate the lift on brightness. Applied flat, it lightens ANY coloured ink -
    # including the brown type on a sepia scan, whose chroma runs 20-40. That
    # silently washes out the faintest words on exactly the pages that can least
    # afford it. A stamp's strokes sit well above this range (measured ~160)
    # while type sits below it, so fading the lift in over mid-tones suppresses
    # the stamp and leaves dark ink untouched.
    gate = np.clip((mx - 90.0) / 60.0, 0.0, 1.0)
    lum = np.clip(mx + chroma * colour_kill * gate, 0, 255).astype(np.uint8)
    return lum, chroma.astype(np.uint8)


def colour_ink(chroma, lum, protect, ink, scale, thresh=42, min_area_px=200,
               max_page_frac=0.04, dark_floor=130, max_chromatic_ink=20.0):
    """Regions of saturated ink: library stamps, rubber-stamp dates, blue pen.

    Chroma alone leaves a ghost, because a stamp's densest strokes are dark
    enough to survive the lift in `luminance`. So grow a region from confident
    coloured pixels and clear it wholesale.

    Three guards, because this stage can do real damage. A page that is merely
    DISCOLOURED - foxed, browned, water-stained - has weak chroma everywhere
    rather than strong chroma somewhere, and without these it reads as one
    enormous stamp and takes the text with it:

      * a threshold set above discolouration. Measured: real stamp ink runs
        44-77, while a badly browned page peaks at 37.
      * `protect` - the detected lines of type - is never cleared. Colour is
        not a reliable witness here: on a sepia scan the faded type carries
        MORE chroma than the threshold (measured p90=43, p99=62 on one page),
        so a colour test alone erases exactly the words that are hardest to
        read. Only structure can tell a stamp from brown type, so a stamp is
        cleared where it crosses blank paper and left where it crosses text.
      * a page-level cap. A stamp is bounded; if the "stamp" covers more of the
        sheet than any stamp could, it is a colour cast and the stage stands
        down rather than guessing.
    """
    blank = np.zeros(lum.shape, np.uint8)
    if chroma is None:
        return blank, 0.0

    # Is this page's own TYPE chromatic? On a sepia or foxed scan the ink
    # carries colour, and then no chroma threshold can separate a stamp from the
    # text - the faint words are both the most coloured and the most fragile.
    # Measured: a clean page runs 0.9-1.0% of ink above the threshold whatever
    # stamps it carries, a sepia page 5.4%. Above the cut, stand down entirely
    # rather than delete words to erase a stamp.
    ink_px = ink > 0
    if ink_px.sum() > 1000:
        # The 90th percentile of INK chroma, not the fraction above the
        # threshold. The fraction conflates "a stamp sits on this page" with
        # "this page's type is brown" - measured 2.1% for a genuine violet stamp
        # against 3.8% for a merely sepia page, which is too close to act on.
        # The percentile asks whether the BULK of the ink is coloured, and
        # separates cleanly: 13-14 for stamped pages, 24-37 for sepia ones.
        bulk = float(np.percentile(chroma[ink_px], 90))
        if bulk > max_chromatic_ink:
            return blank, bulk / 100.0
    seed = ((chroma > thresh) & (lum < 250)).astype(np.uint8) * 255
    k = max(3, int(9 * scale) | 1)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    frac = float((seed > 0).sum()) / seed.size
    if frac > max_page_frac:
        return blank, frac                      # colour cast: do nothing at all

    n, lab, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    region = np.zeros_like(seed)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px * scale * scale:
            region[lab == i] = 255
    region = cv2.dilate(region, np.ones((k, k), np.uint8))
    # Protect ink up to the SAME threshold used to call something coloured.
    # A fixed low bar (chroma < 20) is wrong on a sepia scan, where the type
    # itself carries chroma 20-40: the type then fails the protection test while
    # sitting inside the dilated region, and gets cleared. Anything below the
    # threshold is by definition not "coloured ink", so it must be protected.
    # Protect a halo, not just the strokes. The region was dilated by k before
    # being cleared; clearing right up to a glyph's edge eats its antialiased
    # rim, which reads as the word fading out.
    halo = max(3, int(5 * scale) | 1)
    region[cv2.dilate(protect, np.ones((halo, halo), np.uint8)) > 0] = 0

    # Never clear a pixel dark enough to be ink. Measured on these pages, a
    # flagged region is ~97% near-white paper; the chroma lift in luminance()
    # has already taken the stamp's strokes to near-white by this point, so
    # clearing the region adds almost nothing there - while the few dark pixels
    # it does contain are the text. Without this the stage is net harmful on a
    # sepia page, deleting whole words to erase a stamp that was already gone.
    region[lum < dark_floor] = 0
    return region, frac


def _background(plane, scale, ds=4):
    """Low-frequency paper level. Estimated downscaled: it is smooth by
    definition, and this makes the morphology ~16x cheaper."""
    small = cv2.resize(plane, None, fx=1.0 / ds, fy=1.0 / ds, interpolation=cv2.INTER_AREA)
    k = max(3, int(31 * scale / ds) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), max(2.0, 12 * scale / ds))
    return cv2.resize(bg, (plane.shape[1], plane.shape[0]), interpolation=cv2.INTER_LINEAR)


def flatten(gray, scale):
    """Divide out slowly-varying illumination / paper tone."""
    bg = _background(gray, scale)
    out = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def white_balance(bgr, scale):
    """Flatten each channel against its own paper level.

    Essential before any colour test. Raw chroma cannot separate a violet stamp
    from cream paper - sample 3's paper has chroma 45, the stamp 53. Once each
    channel is divided by its own background, paper of any tint normalises to
    neutral white and only genuinely coloured ink keeps its chroma.
    """
    planes = [flatten(bgr[:, :, c], scale) for c in range(3)]
    return cv2.merge(planes)


def sauvola(gray, win, k=0.25, r=128.0):
    """Local adaptive threshold; robust where a global one clips faint strokes."""
    win = max(3, int(win) | 1)
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
    sq = cv2.boxFilter(g * g, -1, (win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thr = mean * (1 + k * (std / r - 1))
    return ((g < thr) * 255).astype(np.uint8)


def ink_mask(norm, scale, aggressive=False):
    """Union of a local and a global threshold - deliberately over-inclusive.

    Over-inclusion is safe: this mask only decides what is *protected* and what
    is *examined*, never what is drawn.
    """
    win = max(15, int(45 * scale) | 1)
    a = sauvola(norm, win, k=0.20 if aggressive else 0.25)
    otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.bitwise_or(a, otsu)


def glyph_metrics(mask):
    """Median height/area of the components that behave like glyph bodies."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None, None, stats, lab, n
    areas = stats[1:, cv2.CC_STAT_AREA]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    # glyph bodies = upper half by area, excluding runaway blobs (tears, borders)
    cut = np.percentile(areas, 60)
    sel = (areas >= cut) & (areas < np.percentile(areas, 99.5) * 6)
    if sel.sum() < 5:
        sel = areas >= cut
    return float(np.median(heights[sel])), float(np.median(areas[sel])), stats, lab, n


def drop_border_artifacts(mask, gh, scale):
    """Remove big dark things fused to the page edge: tears, spine, shadows."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    out = mask.copy()
    touch = max(4, int(0.02 * min(H, W)))   # a band, not a hairline: strips
                                            # often stop just short of the frame
    removed = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, a = stats[i, :5]
        at_edge = (x <= touch or y <= touch or x + w >= W - touch or y + h >= H - touch)
        if w >= 3 * gh and h <= 0.4 * gh:
            # A printed rule - the double rule and wavy ornament under a running
            # head, say. A tight scan crop runs them into the frame, which makes
            # them look exactly like the scanner-edge strips this stage exists
            # for. What separates them is levelness: the page is deskewed by
            # now, so print lies flat, while a book edge seen past the trim
            # slants or curls and is several strokes tall end to end.
            continue
        # only genuinely large things; a letter grazing the crop edge is spared
        if at_edge and (a > 12 * gh * gh or w > 0.5 * W or h > 0.5 * H):
            removed[lab == i] = 255
    out[removed > 0] = 0
    return out, removed


def line_cover(runs, gh):
    """Per column, how many separate lines of type carry ink there.

    The honest measure of "how far does the text reach". A pixel count cannot
    say it: a torn edge that the smear has chained onto a few lines contributes
    plenty of pixels from very few lines, while the true column edge is reached
    by nearly every line on the page.

    Returns (cover, lines, components) - the per-column count, how many runs
    were long enough to count as lines, and how many the smear produced in all.
    """
    W = runs.shape[1]
    smear = cv2.dilate(runs, np.ones((1, max(3, int(2.0 * gh) | 1)), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(smear, 8)
    cover = np.zeros(W, np.int32)
    lines = 0
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_WIDTH] < 3 * gh:
            continue
        lines += 1
        x, y, w, h = stats[j, :4]
        cover[x:x + w] += (runs[y:y + h, x:x + w] > 0).any(axis=0)
    return cover, lines, n


def edge_junk(mask, core, runs, zone, gh, ga, band_frac=0.09, keep=None):
    """Clear non-text ink in the margins: tears, folds, spine shadow.

    "In the margin" means beyond the sheet-edge band OR beyond the robust text
    column - neither frame alone is enough, since a tear can stop short of the
    trim while sitting far outside the type area.

    What survives out there needs care, because the margins are also where page
    numbers live. Geometry alone cannot tell a page number from a tear fragment:
    both are compact, isolated and glyph-shaped. What separates them is extent.
    Damage runs DOWN the edge as a chain of fragments spanning many lines, while
    a page number is a lone mark with white space above and below. So mirror the
    horizontal line-detection vertically: smear the margin ink downward, and
    treat any chain taller than a few lines as damage.
    """
    H, W = mask.shape
    bw, bh = int(band_frac * W), int(band_frac * H)
    sx0, sx1, sy0, sy1 = bw, W - bw, bh, H - bh
    ys, xs = np.nonzero(runs)
    if len(xs) > 500:
        # percentiles, so sparse damage cannot widen the column it is measured against
        # The MEASURED column wins outright; the 9% band is only a fallback for
        # when line detection found too little to trust. Intersecting the two
        # (the obvious reading of "margin") lets the fixed band cut into text on
        # a page whose type runs out to the trim - and then the margin rule
        # deletes the last letters of every line.
        # Side edges by LINE COUNT, not pixel percentile: a torn edge that
        # the smear has chained onto a few lines contributes plenty of pixels
        # but only a handful of lines, while the column edge is reached by
        # many. Line members are never deleted here whatever the column, so
        # erring narrow cannot cost a letter - it only lets the margin rules
        # look at more of the margin.
        cover, _, ln = line_cover(runs, gh)
        need = 3 if ln > 8 else 1
        cc = np.nonzero(cover >= need)[0]
        if len(cc):
            sx0 = int(cc.min() - 0.5 * gh)
            sx1 = int(cc.max() + 0.5 * gh)
        else:
            sx0 = int(np.percentile(xs, 0.1) - 1.5 * gh)
            sx1 = int(np.percentile(xs, 99.9) + 1.5 * gh)
        sy0 = int(np.percentile(ys, 0.1) - 1.5 * gh)
        sy1 = int(np.percentile(ys, 99.9) + 1.5 * gh)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    # vertical chain analysis, restricted to the side margins
    side = np.zeros_like(mask)
    side[:, :max(0, sx0)] = 255
    side[:, min(W, sx1):] = 255
    side_ink = cv2.bitwise_and(mask, side)
    # A torn edge breaks up with gaps of a line or two; bridge three glyph
    # heights so its pieces still read as one chain down the page.
    vsmear = cv2.dilate(side_ink, np.ones((max(3, int(3.0 * gh) | 1), 1), np.uint8))
    vn, vlab, vstats, _ = cv2.connectedComponentsWithStats(vsmear, 8)
    fl, fv = lab.ravel(), vlab.ravel()
    sel = (fl > 0) & (fv > 0)
    chain = np.zeros(n, np.int32)
    chain[fl[sel]] = fv[sel]
    chain_h = np.zeros(n, np.int32)
    for i in range(1, n):
        if chain[i]:
            chain_h[i] = vstats[chain[i], cv2.CC_STAT_HEIGHT]

    removed = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, a = stats[i, :5]
        if w >= 3 * gh and h <= 0.5 * gh:
            continue                          # a printed rule: furniture, not damage
        sub = (lab[y:y + h, x:x + w] == i)
        if keep is not None and keep[y:y + h, x:x + w][sub].any():
            continue                          # printed furniture: a border side
        cx, cy = x + w / 2.0, y + h / 2.0
        in_side = cx < sx0 or cx > sx1
        in_vert = cy < sy0 or cy > sy1
        if not (in_side or in_vert):
            continue                          # inside the type area; not our business
        if runs[y:y + h, x:x + w][sub].any():
            continue                          # part of a real line of type
        if a <= ga and h <= 1.2 * gh and zone[y:y + h, x:x + w][sub].any():
            # A diacritic-sized mark sitting in the protection zone. Running
            # heads live inside the top band, and their matras are too small to
            # be promoted to line members - so without this the band rule eats
            # them and a heading loses its vowel signs. The zone is the
            # mechanism that protects diacritics; consult it here too.
            continue
        # glyph-shaped AND glyph-sized: nothing a page number is made of is
        # more than about one and a half glyphs tall
        glyphy = core[y:y + h, x:x + w][sub].any() and a <= 3 * ga and h <= 1.6 * gh
        if in_side and not in_vert:
            if glyphy and chain_h[i] < 4 * gh:
                continue                      # lone marginal mark: keep it
        elif glyphy:
            continue                          # page number / catchword
        # Delete the defect, not the component: in a side margin a tear often
        # touches the last letters of a line, and those must survive it.
        piece = (sub * 255).astype(np.uint8)
        defect, _ = split_merge(piece, gh, ga)
        removed[y:y + h, x:x + w][defect > 0] = 255
    out = mask.copy()
    out[removed > 0] = 0
    return out, removed


def _letters(stats, gh, ga):
    """Components shaped and sized like a letter of this page's type."""
    w, h, a = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    sel = ((h >= 0.45 * gh) & (h <= 2.0 * gh) & (a >= 0.3 * ga)
           & (w >= 0.15 * gh) & (h <= 4 * np.maximum(w, 1)))
    sel[0] = False
    return sel


def split_merge(sub, gh, ga):
    """Separate a long linear defect from letters it has merged with.

    A crease that grazes a word becomes ONE connected component with it, and
    deleting components wholesale then deletes the word. Nothing about size or
    position can catch this - the merged blob is not glyph-shaped, so it slips
    past every glyph test, which is exactly how it stays invisible.

    Letters and creases have the same stroke width, so thickness cannot separate
    them. Length can: opening with a line taller than a glyph keeps only runs no
    letter can produce, since a glyph's tallest stem is one glyph high.

    Returns (defect, residue). A non-empty residue means the component was a
    merge, and only `defect` may be deleted.
    """
    L = max(3, int(1.5 * gh) | 1)
    tol = 7                                   # allow the line to wander +/-3 px

    def run(axis):
        # Widen ACROSS the run before opening ALONG it. A crease is never
        # perfectly straight, so a bare line element finds nothing - measured
        # zero surviving pixels on a real merged crease, which is precisely how
        # the whole component, letters included, came to be deleted.
        k_wide = np.ones((1, tol), np.uint8) if axis == 0 else np.ones((tol, 1), np.uint8)
        k_long = np.ones((L, 1), np.uint8) if axis == 0 else np.ones((1, L), np.uint8)
        # Pad first. OpenCV erodes as if everything beyond the array were ink,
        # and `sub` is cropped tight to its component - so any stroke spanning
        # the crop's full height, i.e. the stem of a lone letter, looked
        # infinitely long and was classed as a crease. That is how a letter
        # lost its stem to a fold line that merely touched it.
        p = L + tol
        padded = cv2.copyMakeBorder(sub, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        wide = cv2.dilate(padded, k_wide)
        lin = cv2.morphologyEx(wide, cv2.MORPH_OPEN, k_long)
        lin = cv2.erode(lin, k_wide)[p:-p, p:-p]
        return cv2.bitwise_and(lin, sub)

    linear = cv2.bitwise_or(run(0), run(1))
    if not linear.any():
        return sub, np.zeros_like(sub)

    k3 = np.ones((3, 3), np.uint8)
    seed = cv2.bitwise_and(sub, cv2.bitwise_not(cv2.dilate(linear, k3)))
    n, _, st, _ = cv2.connectedComponentsWithStats(seed, 8)
    if not any(st[i, cv2.CC_STAT_AREA] >= 0.25 * ga for i in range(1, n)):
        return sub, np.zeros_like(sub)        # nothing but defect: delete whole

    # Grow the non-linear mass back until it meets the line itself, rather than
    # taking the line plus a fixed margin. A fixed margin is what cost this page
    # its vowel sign and full stop: they sit against the crease, so any dilation
    # of it swallows them whole. Reconstruction stops exactly at the line.
    allowed = cv2.bitwise_and(sub, cv2.bitwise_not(linear))
    rec = cv2.bitwise_and(seed, allowed)
    for _ in range(64):
        grown = cv2.bitwise_and(cv2.dilate(rec, k3), allowed)
        if cv2.countNonZero(cv2.absdiff(grown, rec)) == 0:
            break
        rec = grown
    # Where the line runs THROUGH a letter - along its stem, say - no shape test
    # can tell the two strokes apart, and taking the line cuts the letter in
    # half. So inside each letter's own box the line stays: what is left is a
    # stroke-length remnant that reads as part of the glyph, while the line is
    # still removed everywhere above and below it.
    n, lab, st, _ = cv2.connectedComponentsWithStats(rec, 8)
    for i in range(1, n):
        x, y, w, h, a = st[i, :5]
        if a >= 0.25 * ga:
            rec[y:y + h, x:x + w] = sub[y:y + h, x:x + w]
    defect = cv2.bitwise_and(sub, cv2.bitwise_not(rec))
    return defect, rec


def blots(mask, core, gh, ga):
    """Remove things too big to be a letter that are not part of any line.

    The margin rules cannot catch these. On a page whose text runs out to the
    sheet edge - speaker labels set flush left, say - the margin band collapses
    to nothing, and a tear streak sitting in that same column is never even
    examined. So judge by what the thing IS, not where it sits.

    `core` already holds every glyph-shaped component and every member of a
    line of type, so anything outside it is neither. If such a component is also
    far larger than a letter, it is damage: a tear, a spine streak, an ink blot.
    Merged text is safe here - a word that blobs into one component still sits
    in a line, so it is in `core`.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    W = mask.shape[1]
    removed = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, a = stats[i, :5]
        if w >= 3 * gh and h <= 0.8 * gh:
            continue                          # printed rule
        sub = (lab[y:y + h, x:x + w] == i)
        if core[y:y + h, x:x + w][sub].any():
            continue                          # glyph, or part of a line
        if w <= 1.0 * gh and 0.12 * W < x + w / 2 < 0.88 * W:
            # A short thin line well inside the sheet: a crease fragment or a
            # printed frame's side. scratches() judges those by the whole
            # structure they join into - a frame's side meets its corners and
            # is not a thin streak - which one piece seen alone cannot show.
            continue
        if h > 3 * gh or w > 3 * gh or a > 2 * ga:
            piece = (sub * 255).astype(np.uint8)
            defect, residue = split_merge(piece, gh, ga)
            tgt = removed[y:y + h, x:x + w]
            tgt[defect > 0] = 255             # residue (letters) is left alone
    out = mask.copy()
    out[removed > 0] = 0
    return out, removed


def scratches(mask, runs, gh, ga, core=None, keep=None, min_len=5.0, max_agg=3.2,
              min_aspect=4.0, min_column=4, max_letter_frac=0.6):
    """Remove long thin creases and fold lines running down the page.

    These defeat every other rule by being FAINT: thresholding shatters a single
    crease into a chain of small fragments, each of which is individually
    glyph-sized and so passes as text. Judged alone they are letters; judged
    together they are a line six glyph-heights long and one wide, which no
    script produces.

    So bridge the gaps with a vertical closing, measure the reconstructed
    object, and if it is long and thin delete the fragments that compose it -
    except any that belong to a real line of type, so a crease crossing a word
    takes the crease and leaves the word.

    The width threshold can afford to be generous because deletion is no longer
    wholesale: split_merge() reconstructs any letter mass inside a candidate and
    removes only the linear part. A torn flap's ragged boundary is much wider
    than a crease, and catching it would have been unsafe while whole components
    were being deleted.
    """
    k = max(3, int(2.0 * gh) | 1)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, 1), np.uint8))
    jn, jlab, jstats, _ = cv2.connectedComponentsWithStats(joined, 8)

    tall = np.zeros(jn, bool)
    for j in range(1, jn):
        w, h = jstats[j, cv2.CC_STAT_WIDTH], jstats[j, cv2.CC_STAT_HEIGHT]
        if h >= min_len * gh and w <= max_agg * gh and h >= min_aspect * max(w, 1):
            tall[j] = True
    if not tall.any():
        return mask, np.zeros_like(mask)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    fl, fj = lab.ravel(), jlab.ravel()
    sel = fl > 0
    owner = np.zeros(n, np.int32)
    owner[fl[sel]] = fj[sel]

    # Script DOES produce a line six glyph-heights long and one wide: a column
    # of a table, the figures down the last column of a table of contents, a
    # file of ditto marks. The vertical closing reassembles one exactly as it
    # reassembles a shattered crease, and the line finder cannot object - the
    # entries stand too far apart to smear into the rows they belong to. What
    # tells them apart is what the chain is MADE of. A crease shatters into
    # slivers and crumbs; a column is a stack of letters. Measured over these
    # samples: a chain that is 62-100% glyph-shaped, glyph-sized pieces is
    # always type, one that is 0-19% always damage, with nothing in between.
    letter = _letters(stats, gh, ga)
    pieces = np.bincount(owner[1:], minlength=jn)
    letters = np.bincount(owner[1:], weights=letter[1:].astype(np.float64), minlength=jn)
    tall &= ~((pieces >= min_column) & (letters >= max_letter_frac * pieces))
    if not tall.any():
        return mask, np.zeros_like(mask)

    # Letters we are keeping, and a narrow reach either side of them. The
    # vertical closing that reassembles a shattered crease also chains in any
    # letter stroke sitting in the same column - measured on a folded flap,
    # where it took the right stem of a ``ma`` clean off. A fragment hugging a
    # kept letter is that letter's stroke, not a piece of the crease.
    beside = None
    if core is not None:
        in_tall = np.where(tall[jlab], 255, 0).astype(np.uint8)
        kept = cv2.bitwise_and(core, cv2.bitwise_not(in_tall))
        reach = max(3, int(0.25 * gh) * 2 + 1)
        beside = cv2.dilate(kept, np.ones((1, reach), np.uint8))

    removed = np.zeros_like(mask)
    for i in range(1, n):
        if not tall[owner[i]]:
            continue
        x, y, w, h = stats[i, :4]
        sub = (lab[y:y + h, x:x + w] == i)
        if keep is not None and keep[y:y + h, x:x + w][sub].any():
            continue                          # printed furniture: a border side
        if runs[y:y + h, x:x + w][sub].any():
            continue                          # real type crossed by the crease
        if (beside is not None and h <= 1.2 * gh
                and core[y:y + h, x:x + w][sub].any()
                and beside[y:y + h, x:x + w][sub].any()):
            continue                          # a stroke of the letter beside it
        piece = (sub * 255).astype(np.uint8)
        defect, residue = split_merge(piece, gh, ga)
        if not residue.any() and h < 2 * w:
            # No line inside it, and not shaped like a piece of one. The
            # closing chains in anything sharing the crease's column - a
            # letter, an anusvara - and a crease shatters into vertical
            # slivers, not into round or wide marks.
            continue
        removed[y:y + h, x:x + w][defect > 0] = 255
    out = mask.copy()
    out[removed > 0] = 0
    return out, removed


def _per_component(lab, n, values, how):
    """Max or min of `values` over each labelled component, vectorised."""
    out = np.full(n, -np.inf if how == "max" else np.inf, np.float64)
    (np.maximum if how == "max" else np.minimum).at(out, lab.ravel(), values.ravel())
    return out


def stroke_radius(mask, gh):
    """Half-width of a typical printed stroke: the median inscribed radius of
    body-sized glyphs. The yardstick for "too thick to be type"."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    mx = _per_component(lab, n, dt, "max")
    h, a = stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    sel = (h >= 0.45 * gh) & (h <= 1.6 * gh) & (a >= 0.12 * gh * gh)
    sel[0] = False
    return (float(np.median(mx[sel])) if sel.any() else max(1.5, gh / 9.0)), dt, lab, n, stats


def dust_discs(mask, norm, gh, sr, dt, lab, n, stats):
    """Out-of-focus dust on the scanner glass: soft grey discs.

    These are glyph-sized, so the size rules call them letters and protect
    them - worse, their presence is what props the text box open. Three things
    together separate them from print, and each was measured against every
    component in the samples:
      * round and solid        - no letter fills 60% of its box at ~1:1
      * thicker than any stroke - inscribed radius >= 1.6x the page's stroke,
                                  so an anusvara (radius ~ one stroke) can't match
      * never ink-black         - printed dots (full stops, the ``0`` sign) bottom
                                  out at 0-6; dust sits at 50-160, being a blur
    """
    mx = _per_component(lab, n, dt, "max")
    mn = _per_component(lab, n, norm.astype(np.float64), "min")
    removed = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, a = stats[i, :5]
        if max(w, h) > 1.5 * gh or a < 9:
            continue
        if a < 0.6 * w * h or not 0.6 <= w / h <= 1.6:
            continue
        if mx[i] < 1.6 * sr or mn[i] <= 45:
            continue
        removed[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i] = 255
    out = mask.copy()
    out[removed > 0] = 0
    return out, removed


def blob_bodies(mask, gh, ga, sr, factor=3.5):
    """Delete the parts of any mark that are too thick to be type.

    Open the ink with a disc wider than any stroke can hold: whatever survives
    is a region no letter could contain - an ink blot, a torn edge, a thumb
    mark. Measured: the thickest real letters (bold speaker labels, a heavy
    ``tI``) reach 2.6x the page's stroke radius; blots start at 3.9x.

    This is safe even when a blot has swallowed a letter, because an opening
    removes nothing thin: the letter's strokes are left standing where they
    were. Only the blot's ragged rim is then tidied - fragments that were
    part of the same mark and are too small to be a letter.
    """
    r = int(np.ceil(factor * sr))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    body = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if not body.any():
        return mask, np.zeros_like(mask)
    k3 = np.ones((3, 3), np.uint8)
    body = cv2.bitwise_and(cv2.dilate(body, k3), mask)      # its antialiased rim
    rest = cv2.bitwise_and(mask, cv2.bitwise_not(body))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest, 8)
    near = cv2.dilate(body, k3)          # 8-adjacent = was the same mark
    touch = _per_component(lab, n, (near > 0).astype(np.float64), "max")
    removed = body.copy()
    for i in range(1, n):
        if touch[i] > 0 and stats[i, cv2.CC_STAT_AREA] < 0.25 * ga:
            x, y, w, h = stats[i, :4]
            removed[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i] = 255
    out = mask.copy()
    out[removed > 0] = 0
    return out, removed


def is_sliver(w, h, gh):
    """A crease or sheet-edge fragment: tall and hairline-thin. Measured on the
    samples, the tallest thin print (parentheses, the long i-matra) is 1.6x
    glyph height; edge slivers run 1.8-3x."""
    return h >= 1.8 * gh and w <= 0.35 * gh


def edge_bands(mask, runs, gh, thresh=0.15, reach=0.2, max_line_frac=0.2, keep=None):
    """Find the damaged strips down each side of the sheet, by structure.

    Text columns are blank BETWEEN lines of type; a torn edge, fold or spine
    shadow is inked continuously down the page. So measure, for every column,
    how often it carries ink in the gaps between lines. Measured: edge strips
    run 0.2-0.8, text columns 0.01-0.08 (0.14 where a crease crosses). This is
    immune to what defeats every per-component rule at the edge - damage that
    breaks into letter-sized fragments and chains itself onto the lines.

    That inter-line evidence is only as good as the gaps, though, and on a
    densely leaded Indic page there is barely a gap to read: the upper matras
    and conjunct stems of the next line stand in it, and a column of ordinary
    type then scores 0.10-0.21 - above the threshold. The scan is picked apart
    column by column, so one such column anywhere in the outer fifth of the
    sheet drags the band's inner edge out to meet it, and everything between
    the trim and that column is condemned along with it. Measured on these
    samples that is the first word of every line on the page, or the last.

    So the band must also answer to the text: a strip of damage lies OUTSIDE
    the type area, and the lines of type stop short of it. Where instead the
    lines run on into the proposed band, whatever the inter-line reading says,
    the band has been drawn across the text and the side stands down. Measured:
    a true edge strip is reached by 0-12% of the page's lines (the few whose
    ends the damage has chained itself onto), a misplaced band by 31-76%.

    `keep` is printed furniture, and is not evidence. The side of a border box
    is inked in every gap between every line, which is the whole signature this
    stage hunts for, so a page printed inside a border reads as having a strip
    of damage down each side of it - ending exactly where the border is, with
    the type in between condemned.

    Returns (left, right): the inner edge of each band, 0 / W when none.
    """
    if keep is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(keep))
    H, W = mask.shape
    rp = (runs > 0).sum(axis=1)
    if not rp.any():
        return 0, W
    band = rp > 0.2 * np.percentile(rp[rp > 0], 75)
    # grow line rows by a matra's reach so ascenders are not read as gap ink
    band = cv2.dilate(band.astype(np.uint8)[:, None],
                      np.ones((int(0.9 * gh) | 1, 1), np.uint8))[:, 0] > 0
    ry = np.nonzero(rp)[0]
    gap = np.nonzero(~band)[0]
    gap = gap[(gap > ry.min()) & (gap < ry.max())]
    if len(gap) < 3 * gh:
        return 0, W                              # too few gaps to judge by
    ink = cv2.dilate((mask[gap] > 0).astype(np.uint8), np.ones((1, 5), np.uint8))
    prof = cv2.blur(ink.mean(axis=0).astype(np.float32)[None, :], (9, 1))[0]
    r = int(reach * W)
    lc = np.nonzero(prof[:r] >= thresh)[0]
    rc = np.nonzero(prof[W - r:] >= thresh)[0]
    left = int(lc.max()) + 1 if len(lc) else 0
    right = W - r + int(rc.min()) if len(rc) else W

    cover, lines, _ = line_cover(runs, gh)
    limit = max(3, int(max_line_frac * lines))
    if left > 0 and cover[:left].max() >= limit:
        left = 0
    if right < W and cover[right:].max() >= limit:
        right = W
    return left, right


def clear_bands(mask, runs, gh, left, right, keep=None, ga=None, max_letter_frac=0.03):
    """Delete marks lying wholly inside an edge band.

    Wholly: a first letter that a strip runs into crosses the band's inner
    edge and is kept. And anything chained to kept text across a word gap is
    kept too, so a line-initial or line-final word that falls inside the band
    stays with its line.

    `keep` is ink that must stay wherever it lies - the printed rules and
    ornaments. The side of a border box is the one thing here that cannot be
    saved by chaining it to a line: it stands alone in the margin by design,
    a clear word's width outside the type it encloses.

    Finally, each side is asked what it cost. This stage deletes wholesale by
    position, on evidence read column by column, and every way that evidence
    can mislead ends the same way - the band drawn across type instead of
    beside it. Rather than enumerate the ways, count the letters: a strip of
    damage holds none of this page's type and measures 0.0-2.2% of it, while
    a band that has reached into a column of figures or a line of words takes
    3.7-16%. Over that, the side stands down and leaves the margin dirty,
    which is this tool's standing preference over leaving it short of words.
    """
    H, W = mask.shape
    if left <= 0 and right >= W:
        return mask, np.zeros_like(mask)
    slack = int(0.25 * gh)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    x, w = st[:, cv2.CC_STAT_LEFT], st[:, cv2.CC_STAT_WIDTH]
    inside = (x + w <= left + slack) | (x >= right - slack)
    inside[0] = False
    if keep is not None and keep.any():
        inside &= ~(_per_component(lab, n, (keep > 0).astype(np.float64), "max") > 0)
    is_run = _per_component(lab, n, (runs > 0).astype(np.float64), "max") > 0
    # Grow "kept text" along each line, one WORD gap at a time. A letter gap is
    # not enough: what sits inside the band is a whole word - the first on its
    # line or the last - and what it has to reach across to be recognised as
    # part of that line is the space before or after it. Reaching only 0.8 of a
    # glyph height stops short of the 1.2-2.0 these pages set, which is why a
    # line-initial word inside the band was never chained to the line it
    # belongs to. text_runs bridges word gaps at 2.0; match it.
    k = np.ones((max(3, int(0.3 * gh)), max(3, int(2.0 * gh) | 1)), np.uint8)
    for _ in range(8):
        kept_text = np.where((is_run & ~inside)[lab], 255, 0).astype(np.uint8)
        near = _per_component(lab, n, (cv2.dilate(kept_text, k) > 0).astype(np.float64), "max") > 0
        freed = inside & is_run & near
        if not freed.any():
            break
        inside &= ~freed

    if ga is not None:
        letter = _letters(st, gh, ga)
        budget = max_letter_frac * max(int(letter.sum()), 1)
        on_left = st[:, cv2.CC_STAT_LEFT] + st[:, cv2.CC_STAT_WIDTH] <= left + slack
        for side in (on_left, ~on_left):
            if (inside & letter & side).sum() > budget:
                inside &= ~side
    removed = np.where(inside[lab], 255, 0).astype(np.uint8)
    out = mask.copy()
    out[removed > 0] = 0
    return out, removed


def guard_box(box, runs, gh):
    """Widen the crop until it contains every confirmed line of type.

    The margin clip is the one stage that could silently delete a whole word,
    so it is not allowed to be the last word on its own geometry: whatever the
    percentiles decided, no pixel that belongs to a real text line may fall
    outside the box.
    """
    ys, xs = np.nonzero(runs)
    if len(xs) == 0:
        return box
    pad = int(0.4 * gh)
    x0, y0, x1, y1 = box
    return (min(x0, max(0, int(xs.min()) - pad)),
            min(y0, max(0, int(ys.min()) - pad)),
            max(x1, min(runs.shape[1], int(xs.max()) + 1 + pad)),
            max(y1, min(runs.shape[0], int(ys.max()) + 1 + pad)))


def text_block(mask, gh, scale):
    """Robust bounding box of the type area.

    Driven by confirmed text runs, not raw ink: a scanner-edge strip or a torn
    flap breaks into dozens of glyph-sized fragments, and those would otherwise
    drag the box out to the full sheet and keep the junk in frame.
    """
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    keep = []
    for i in range(1, n):
        x, y, w, h, a = stats[i, :5]
        if 0.35 * gh <= h <= 6 * gh and a >= 0.15 * gh * gh and w <= 0.9 * W:
            keep.append((x, y, x + w, y + h))
    if len(keep) < 20:
        return (0, 0, W, H)
    k = np.array(keep)
    # percentiles, not extremes: one surviving blob must not blow up the box
    x0 = int(np.percentile(k[:, 0], 0.5))
    y0 = int(np.percentile(k[:, 1], 0.5))
    x1 = int(np.percentile(k[:, 2], 99.5))
    y1 = int(np.percentile(k[:, 3], 99.5))
    pad = int(1.5 * gh)
    return (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))


def text_runs(mask, gh):
    """Mark every component that belongs to a run of type, at ANY point size.

    Size alone misidentifies small type: footnotes and the parenthetical
    tune-headings in these books are half the body height, so a height test
    brands them as dust. Text however is *serial* - many similar blobs strung
    along a line - while dirt is isolated. So smear horizontally (RLSA) and
    keep whatever lands in a long, well-populated run.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    # The smear must bridge WORD gaps, not just letter gaps. At 0.8x glyph
    # height it stops short of them, so every word forms its own run and short
    # words - the first or last on a line - fail the member count and lose their
    # protection. Inter-word gaps here run to ~1.2x glyph height.
    smear = cv2.dilate(mask, np.ones((1, max(3, int(2.0 * gh) | 1)), np.uint8))
    sn, slab, sstats, _ = cv2.connectedComponentsWithStats(smear, 8)

    # Map component -> smeared-line via an actual member pixel. Sampling the
    # bounding-box centre instead looks equivalent and is not: for a hollow or
    # open glyph the centre lands on background, the component gets filed under
    # label 0, and it is then not recognised as part of its own line. That is
    # silent, and it costs real words at the margins.
    flat_lab, flat_slab = lab.ravel(), slab.ravel()
    sel = flat_lab > 0
    comp_to_line = np.zeros(n, np.int32)
    comp_to_line[flat_lab[sel]] = flat_slab[sel]
    members = {}
    for i in range(1, n):
        members.setdefault(int(comp_to_line[i]), []).append(i)

    is_core = np.zeros(n, bool)
    is_run = np.zeros(n, bool)
    for i in range(1, n):
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        # Isolated components must be body-sized AND compact. Both bounds earn
        # their keep: a scanner-edge strip arrives as a picket fence of slivers
        # (w=3-6, h=24-88) that would otherwise pass as glyphs and drag the crop
        # box out to the full sheet. Anything inside a real line of type is
        # admitted below regardless of shape, so narrow strokes are not at risk.
        if (0.45 * gh <= h <= 4 * gh and a >= 0.12 * gh * gh and h <= 6 * max(w, 1)
                and not is_sliver(w, h, gh)):
            is_core[i] = True
    for s, mem in members.items():
        if s == 0:
            continue
        if sstats[s, cv2.CC_STAT_WIDTH] >= 3 * gh and len(mem) >= 2:
            for i in mem:                   # a genuine line of type, any size
                # ...but line membership must not launder damage into "text".
                # The smear is wide enough to bridge word gaps, so it also
                # reaches a tear streak sitting beside a flush-left line. No
                # letter is several times a glyph tall, so refuse those.
                ch = stats[i, cv2.CC_STAT_HEIGHT]
                if ch > 3 * gh:
                    continue
                # ...and it must not launder DIRT into text either. The smear
                # reaches half a glyph either side, so a speck lying near a line
                # would otherwise be promoted to "confirmed text" and become
                # undespeckleable. Diacritics do not need this: they are held by
                # the protection zone, which is what that mechanism is for.
                if ch < 0.3 * gh:
                    continue
                # ...nor a crease. A sheet edge breaks into hairline slivers a
                # few glyphs tall; the smear strings them onto the nearest line,
                # after which the whole margin reads as text and is never cleaned.
                if is_sliver(stats[i, cv2.CC_STAT_WIDTH], ch, gh):
                    continue
                is_core[i] = is_run[i] = True
    # label -> mask by lookup table: one pass over the page, not one per component
    core = np.where(is_core[lab], 255, 0).astype(np.uint8)
    runs = np.where(is_run[lab], 255, 0).astype(np.uint8)
    return core, runs


def ornament_rules(mask, gh, min_len=8.0, max_h=0.5, min_pieces=6):
    """Printed rules and ornaments: long, level, thin chains of marks.

    A wavy rule under a running head breaks into dozens of speck-sized pieces
    at this resolution, each of which the despeckler rightly calls dust. Taken
    together they are a level band many glyphs long and a fraction of one
    tall, which dust never forms - the same serial argument as for text, run
    at a thinner gauge. Bands touching the sheet edge are left out: that is
    where scanner shadow lies, and it is the one other thing this shape fits.
    """
    H, W = mask.shape
    smear = cv2.dilate(mask, np.ones((1, max(3, int(0.6 * gh) | 1)), np.uint8))
    sn, slab, sst, _ = cv2.connectedComponentsWithStats(smear, 8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    fl, fs = lab.ravel(), slab.ravel()
    sel = fl > 0
    owner = np.zeros(n, np.int32)
    owner[fl[sel]] = fs[sel]
    count = np.bincount(owner[1:], minlength=sn)
    band = np.zeros(sn, bool)
    edge = max(4, int(0.02 * min(H, W)))
    for j in range(1, sn):
        x, y, w, h = sst[j, :4]
        if (w >= min_len * gh and h <= max_h * gh and count[j] >= min_pieces
                and y > edge and y + h < H - edge):
            band[j] = True
    keep = np.zeros(n, bool)
    keep[1:] = band[owner[1:]]
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def _straight_rules(mask, gh, min_len, max_thick, max_wobble, max_ragged, min_rows):
    """Components that are a drawn line down the array: see frame_rules."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        if h < min_len * gh or w > max_thick * gh:
            continue
        sub = (lab[y:y + h, x:x + w] == i)
        rows = np.nonzero(sub.any(axis=1))[0]
        if len(rows) < min_rows * h:
            continue                          # gappy: a chain of marks, not a line
        spans = [np.nonzero(sub[r])[0] for r in rows]
        centre = np.array([s.mean() for s in spans])
        thick = np.array([float(np.ptp(s)) + 1.0 for s in spans])
        fit = np.polyval(np.polyfit(rows, centre, 1), rows)
        if np.std(centre - fit) > max_wobble * gh:
            continue                          # wanders: a crease
        if thick.mean() > 0.35 * gh or np.std(thick) > max_ragged * thick.mean():
            continue                          # ragged or fat: a tear, not a rule
        out[y:y + h, x:x + w][sub] = 255
    return out


def frame_rules(mask, gh, min_len=9.0, max_thick=1.0, max_wobble=0.08,
                max_ragged=0.35, min_rows=0.7):
    """Solid printed rules: the border round a page, the rule under a head.

    `ornament_rules` above finds the case that arrives in PIECES - a wavy
    ornament, shattered by thresholding into dozens of marks. A plain rule
    arrives whole instead, as one component tens of glyphs long and a few
    pixels through, and no count of pieces will ever find it. Its shape is
    also the shape of everything this pipeline takes out of a margin - a
    crease, the edge of a torn flap, the shadow down a spine - and it lies
    where they lie. So stage after stage deletes it, and a book printed
    inside a ruled border comes back with three sides of it, or none.

    What no damage reproduces is that a rule was DRAWN. It holds one line and
    one thickness along its whole length; a tear is ragged by nature and a
    crease wanders. Measured on these samples a printed rule strays 0.03-0.06
    of a glyph height from its own centre line and varies 0.17-0.24 in
    thickness, while a strip of edge damage of the same length strays 0.17
    and varies 0.98.

    Kept apart from the ornaments above, which the caller folds into the lines
    of type: those are shaped like lines and cost nothing there, but a rule
    standing the height of the page is not one, and filing it as one wrecks
    the line finding it joins. Smeared to bridge word gaps it welds every line
    on the sheet into a single run, after which the page reads as having four
    lines on it and every rule that counts them is blind.
    """
    down = _straight_rules(mask, gh, min_len, max_thick, max_wobble, max_ragged, min_rows)
    across = _straight_rules(mask.T, gh, min_len, max_thick, max_wobble,
                             max_ragged, min_rows).T
    return cv2.bitwise_or(down, np.ascontiguousarray(across))


def protection_zone(mask, gh, up=0.85, down=0.65, side=0.55):
    """Grow a zone around real text. Anything inside is never touched.

    Asymmetric and anisotropic: more headroom above (anusvara, upper matras)
    than below, and a modest horizontal reach so that a speck sitting in the
    margin on the same line as text is still reachable.
    """
    core, runs = text_runs(mask, gh)
    dy_up, dy_dn, dx = int(up * gh), int(down * gh), int(side * gh)
    kern = np.ones((dy_up + dy_dn + 1, 2 * dx + 1), np.uint8)
    return cv2.dilate(core, kern, anchor=(dx, dy_up)), core, runs


# ---------------------------------------------------------------- text detector

MODEL = Path(__file__).with_name("models") / "ppocrv6_tiny_det.onnx"
_detector = None


def detect_text(bgr, thresh=0.2, box_thresh=0.4, unclip=1.4, max_side=960):
    """Lines of type, as a learned detector sees them. (mask, boxes) or None.

    PP-OCRv6 tiny (PaddleOCR, Apache-2.0): a 1.8 MB DB text detector run
    through OpenCV's own DNN module, so it adds no dependency. It is used as a
    second opinion, never to paint pixels: it only says where lines of type
    are. That is the one question the RLSA line finder cannot answer at a torn
    edge, where damage breaks into letter-sized pieces and chains itself onto
    the lines. The detector was trained on print and does not box it.

    Post-processing is PaddleOCR's DBPostProcess: probability > 0.2, keep
    regions whose min-area box averages > 0.4, grow each box by
    area * 1.4 / perimeter. Run at 960 px on the long side, as PaddleOCR does -
    at full resolution it starts boxing ragged edge strips as text.
    """
    global _detector
    if _detector is None:
        try:
            _detector = cv2.dnn.readNetFromONNX(MODEL)
        except (cv2.error, AttributeError):
            _detector = False
    if _detector is False:
        return None
    H, W = bgr.shape[:2]
    r = min(1.0, max_side / max(H, W))
    h, w = max(32, int(round(H * r / 32)) * 32), max(32, int(round(W * r / 32)) * 32)
    x = cv2.resize(bgr, (w, h)).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    _detector.setInput(x.transpose(2, 0, 1)[None])
    prob = cv2.resize(_detector.forward()[0, 0], (W, H))
    cs, _ = cv2.findContours((prob > thresh).astype(np.uint8), cv2.RETR_LIST,
                             cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((H, W), np.uint8)
    boxes = []
    for c in cs[:3000]:
        if len(c) < 4:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if min(rw, rh) < 3:
            continue
        pts = cv2.boxPoints(rect)
        x0, y0 = np.floor(pts.min(0)).astype(int).clip(0)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x1, y1 = min(x1, W - 1), min(y1, H - 1)
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(m, [(pts - (x0, y0)).astype(np.int32)], 1)
        if cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], m)[0] < box_thresh:
            continue
        d = rw * rh * unclip / (2 * (rw + rh))
        poly = cv2.boxPoints(((cx, cy), (rw + 2 * d, rh + 2 * d), ang)).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
        boxes.append(cv2.boundingRect(poly))
    return mask, boxes


def rescue_marks(removed, det, norm, chroma, kept, gh, near=False,
                 dark=60, size=0.7, reach=0.35, colour=42, colour_frac=0.01):
    """Ink-black marks the detector places INSIDE a line of type go back.

    The dot-sized members of a line - an anusvara over the first letter, a
    full stop set high and loose at a line end, a short dash - are too small
    to count as line members and can fall outside the protection zone, where
    the despeckler takes them. The detector's line box says they belong to
    the line. What it cannot tell is a dust dot in the same box, so some of
    those come back too: the usual trade.

    Excluded: anything not ink-black (dust is a grey blur), anything larger
    than `size`, and anything in a coloured neighbourhood (the black flecks
    inside a violet stamp). With `near`, which is used for marks the edge and
    band stages took, the mark must also sit within a letter gap of ink that
    survived, so a dot stranded in an edge band stays gone.

    `size` is a diacritic by default because the caller that rescues specks has
    no other guard: widen it there and page dirt comes back with the dots. The
    edge and band stages are a different case and are given a whole glyph. What
    they take is not a speck but a region of the margin, and what a margin
    holds besides damage is type the line finder could not confirm - a page
    number, the figures in the last column of a table of contents. Those are
    full-sized letters, and geometry cannot tell them from tear fragments; the
    file says as much where it weighs the same question in `edge_junk`. The
    detector can, that being the reason it is consulted at all, and its verdict
    is still hedged here by darkness, by colour, and by `near`.
    """
    n, lab, st, cen = cv2.connectedComponentsWithStats(removed, 8)
    out = np.zeros_like(removed)
    H, W = removed.shape
    r, R = int(reach * gh), int(3 * gh)
    vy = int(max(0.8, reach) * gh)            # reach is a reach, in both axes
    ok = []
    for j in range(1, n):
        x, y, w, h, _ = st[j]
        cx, cy = int(cen[j][0]), int(cen[j][1])
        if w > size * gh or h > size * gh or not det[cy, cx]:
            continue
        sub = lab[y:y + h, x:x + w] == j
        if norm[y:y + h, x:x + w][sub].min() >= dark:
            continue
        if chroma is not None:
            win = chroma[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
            if (win > colour).mean() > colour_frac:
                continue
        ok.append((x, y, w, h, sub))
    if not near:
        for x, y, w, h, sub in ok:
            out[y:y + h, x:x + w][sub] = 255
        return out

    # A mark vouched for by the detector still has to sit beside surviving ink,
    # or a speck stranded in an edge band comes back with the words. But the
    # ink it sits beside may be another rescued mark: what these stages strand
    # in a margin is often a whole column of type - the figures down the last
    # column of a table of contents - and taken one at a time every figure in
    # it looks equally alone. So let the rescue support itself, line-finding's
    # own serial argument, seeded from ink that survived on its own account.
    near_ink = kept.copy()
    pending = list(ok)
    while pending:
        rest = []
        for item in pending:
            x, y, w, h, sub = item
            if near_ink[max(0, y - vy):y + h + vy, max(0, x - r):x + w + r].any():
                out[y:y + h, x:x + w][sub] = 255
            else:
                rest.append(item)
        if len(rest) == len(pending):
            break
        near_ink = cv2.bitwise_or(kept, out)
        pending = rest
    return out


def detector_gate(mask, det, boxes, keep, gh, pad=0.5):
    """Remove ink outside the detected text column that touches no line.

    The column is the span of the detected lines of type (boxes at least four
    glyphs wide), and runs from the topmost to the bottommost box of any size,
    so running heads, page numbers and footnotes are inside it. Ink wholly
    within the column is left to the rest of the pipeline. Outside it, a mark
    has to touch a detected line to stay, which is true of text on a folded
    flap and of nothing else found out there: tear fragments, edge pepper,
    shadow.
    `keep` is ink that must stay regardless (printed rules and ornaments,
    which the detector does not box because they are not text).
    """
    lines = [b for b in boxes if b[2] >= 4 * gh]
    if len(lines) < 3:
        return mask, np.zeros_like(mask)
    p = int(pad * gh)
    T = min(b[1] for b in boxes) - p
    B = max(b[1] + b[3] for b in boxes) + p
    # The column is LOCAL: verse set short beside a torn edge must not borrow
    # the width of prose further down the page, or the damage beside the verse
    # counts as inside the column. Each row takes the extent of the lines
    # within `reach` of it, falling back to the whole page's in the margins.
    H = mask.shape[0]
    reach = int(8 * gh)
    lo = np.full(H, min(b[0] for b in lines) - p)
    hi = np.full(H, max(b[0] + b[2] for b in lines) + p)
    ll = np.full(H, 10 ** 6); rr = np.full(H, -1)
    for bx, by, bw, bh in lines:
        a, b = max(0, by - reach), min(H, by + bh + reach)
        ll[a:b] = np.minimum(ll[a:b], bx - p)
        rr[a:b] = np.maximum(rr[a:b], bx + bw + p)
    have = rr >= 0
    lo[have], hi[have] = ll[have], rr[have]
    # a full glyph of reach sideways: a loose full stop can sit that far out
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(2 * gh) | 1, int(0.7 * gh) | 1))
    near = cv2.dilate(det, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    touch = np.zeros(n, bool)
    touch[np.unique(lab[near > 0])] = True
    touch[np.unique(lab[keep > 0])] = True
    x, y, w, h = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    yc = np.clip(y + h // 2, 0, H - 1)
    inside = (x >= lo[yc]) & (x + w <= hi[yc]) & (y >= T) & (y + h <= B)
    rule = (w >= 3 * gh) & (h <= 0.4 * gh)
    drop = ~(touch | inside | rule)
    drop[0] = False
    gone = np.where(drop[lab], 255, 0).astype(np.uint8)
    return cv2.bitwise_and(mask, cv2.bitwise_not(gone)), gone


def attached_halo(norm, core, halo):
    """Marks joined to a letter by ink too faint to threshold but not paper.

    Tier 2 below rests on "ink bottoms out near 0, so a mark that never
    approaches ink is dirt". That reads the mark alone, and it is wrong
    wherever the IMPRESSION was weak: the lightly inked middle of a conjunct
    breaks off from its own letter, is grey rather than black, stands a stroke
    away from anything the line finder confirmed, and so answers to every test
    for a smudge. Whole words came back with their insides eaten out.

    What the mark alone cannot show, its surroundings can. Follow it at the
    level where paper stops being paper: a detached piece of a letter is
    joined to the rest of that letter by ink faint enough to have missed the
    ink mask and far too dark to be the sheet, while a speck of dirt sits on
    clean paper and its halo is an island. Measured on these pages, 75% of the
    marks this tier took out of weakly printed words are joined to a letter
    this way, against 0-35% of the marks it took off open paper.

    `halo` is that level. It sits below the sheet - the background having been
    flattened, paper reads 250-255 - and well above anything the ink mask
    reached, which is the whole point: the bridge is made of exactly the ink
    the mask could not see. It is deliberately not the tone curve's white
    point, which answers a different question (what shall print as paper).
    """
    soft = ((norm < halo) * 255).astype(np.uint8)
    n, lab = cv2.connectedComponents(soft, 8)[:2]
    attached = np.zeros(max(n, 1), bool)
    attached[np.unique(lab[core > 0])] = True
    attached[0] = False
    return attached[lab]


def despeckle(mask, norm, core, zone, gh, ga,
              max_area_frac=0.55, max_dim_frac=0.9, faint=150, halo=236):
    """Two tiers, both conservative.

    Tier 1 - outside the protection zone: any small blob goes. Nothing that
             belongs to a line of type can be here, by construction - and
             where that construction fails, the halo test catches it. The
             zone is grown from the components the line finder confirmed, so
             a word printed too weakly to confirm leaves a hole in it, and
             tier 1 applies no test of darkness at all: measured on one such
             word, pieces bottoming out at 44 - ink by any standard - were
             deleted for sitting in the hole.
    Tier 2 - inside the zone: a blob goes only if it is provably not print.
             Letterpress ink on these pages bottoms out near 0; a blob whose
             darkest pixel never approaches ink is paper texture or a smudge.

    Either way a mark is kept if it stands against real ink, or is joined to
    a letter by its own halo (see :func:`attached_halo`), so that a faint
    fragment of an actual stroke is never taken for dirt.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    near_ink = cv2.dilate(core, np.ones((5, 5), np.uint8))
    joined = attached_halo(norm, core, halo) if halo else None
    removed = np.zeros_like(mask)
    spared = 0
    for i in range(1, n):
        x, y, w, h, a = stats[i, :5]
        if a > max_area_frac * ga or h > max_dim_frac * gh or w > max_dim_frac * gh:
            continue                                   # too big to be dust
        sub = (lab[y:y + h, x:x + w] == i)
        if core[y:y + h, x:x + w][sub].any():
            continue                                   # it IS text
        if w >= 2.5 * h and w >= 0.5 * gh:
            # A dash or hyphen. These sit in the gap after a speaker name, far
            # enough from any glyph to fall outside the protection zone, and
            # small enough to pass as dust - so they were being deleted as
            # specks. Dirt is rarely 4:1 and horizontal; printed rules are.
            continue
        touching = (near_ink[y:y + h, x:x + w][sub].any()
                    or (joined is not None and joined[y:y + h, x:x + w][sub].any()))
        if not zone[y:y + h, x:x + w][sub].any():
            if touching:
                spared += 1
            else:
                removed[y:y + h, x:x + w][sub] = 255   # tier 1
            continue
        px = norm[y:y + h, x:x + w][sub]
        if faint and px.min() > faint and not touching:
            removed[y:y + h, x:x + w][sub] = 255       # tier 2
        else:
            spared += 1
    out = mask.copy()
    out[removed > 0] = 0
    return out, removed, spared


def compose(norm, kill, box, black=40, white=224, soften=1.0, ink=None):
    """Paper to pure white, ink kept as antialiased greyscale.

    We paint from the *continuous-tone* image, not from the binary mask, so no
    thresholding decision can thin a stroke. `kill` marks pixels proven to be
    dirt; those are forced white.

    `ink` is the surviving ink mask. Anything more than two pixels from it is
    paper by construction - the mask is a deliberately over-inclusive union of
    a local and a global threshold, so it already holds every stroke down to
    the faintest - and is painted white. Without this, grey mottling lighter
    than any ink (a crumbling sheet edge, foxing, show-through) passes the tone
    curve as a visible grey haze, because it was never "dirt" to remove: it was
    never ink to begin with.
    """
    g = norm.astype(np.float32)
    g = (g - black) / max(1.0, (white - black)) * 255.0
    g = np.clip(g, 0, 255)
    if soften and soften > 0:
        # feather removals so we do not leave a hard hole in the paper
        k = cv2.dilate(kill, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(k.astype(np.float32) / 255.0, (0, 0), soften)
        g = g + (255.0 - g) * np.clip(alpha, 0, 1)
    if ink is not None:
        # two pixels of reach keeps each stroke's antialiased rim intact
        keep = cv2.dilate(ink, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0
        keep = cv2.GaussianBlur(keep, (0, 0), 0.8)
        g = 255.0 - (255.0 - g) * np.clip(keep, 0, 1)
    out = np.full_like(g, 255.0)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = g[y0:y1, x0:x1]
    return np.clip(out, 0, 255).astype(np.uint8)


def sharpen(gray, amount=0.8, radius=1.0, threshold=6):
    """Unsharp mask, applied only where there is an actual edge.

    The scan is optically soft, so strokes carry a gradient that reads as blur
    at print size. A plain unsharp would also amplify the paper grain left
    between letters, so gate it on local gradient: flat paper stays flat.
    """
    if amount <= 0:
        return gray
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), radius)
    detail = g - blur
    if threshold > 0:
        detail[np.abs(detail) < threshold] = 0.0
    return np.clip(g + amount * detail, 0, 255).astype(np.uint8)


def supersample(gray, factor):
    """Resample the page up by an integer factor.

    This adds no information - the scan's detail is fixed - but it does stop the
    printer's own halftoning from landing on top of a coarse pixel grid, and it
    gives the following sharpen a finer grid to place stroke edges on. The page
    keeps its physical size; only the pixel density changes.
    """
    if factor <= 1:
        return gray
    return cv2.resize(gray, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_LANCZOS4)


def skew_angle(gray, limit=3.0):
    """Page skew in degrees, from the text lines themselves.

    Rotate the ink through candidate angles and keep the one whose row profile
    is sharpest: when lines are level, each is a tall peak with clean gaps
    between; tilted, they smear into each other. This is what ImageMagick's
    -deskew and Leptonica do. Unlike fitting a rectangle round all the ink it
    is blind to what is NOT text - a vertical edge streak adds the same amount
    to every row, so it cannot pull the answer.
    """
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ink = ink.astype(np.float32)
    h, w = ink.shape

    def sharpness(a):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
        prof = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_LINEAR).sum(axis=1)
        return float(np.sum(np.diff(prof) ** 2))

    best = max(np.arange(-limit, limit + 1e-6, 0.2), key=sharpness)
    best = max(np.arange(best - 0.2, best + 0.2 + 1e-6, 0.02), key=sharpness)
    return round(float(best), 2)


def straighten(bgr, scale, min_angle=0.12):
    """Rotate the scan level before any analysis.

    Done first, not last: every later stage reasons about horizontal lines of
    type and vertical margins, and all of them work better on a level page.
    The exposed corners are filled with the page's own paper colour so they
    read as paper, not as a dark wedge to be cleaned up.
    """
    gray = flatten(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), scale)
    ang = skew_angle(gray)
    if abs(ang) < min_angle:
        return bgr, 0.0
    h, w = bgr.shape[:2]
    paper = [int(np.median(bgr[:, :, c])) for c in range(3)]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=paper), ang


# ---------------------------------------------------------------- driver

def analyse(bgr, dpi, opt):
    """Every decision, no rendering: what is ink, what is text, what goes.

    Shared by clean_page() and audit_lines.py, so the audit measures exactly
    the code that ships rather than a copy of it that can drift.
    Returns None when the page has no text to reason about.
    """
    scale = dpi / 300.0                      # kernels were tuned at 300 dpi
    ang = 0.0
    if getattr(opt, "deskew", True):
        bgr, ang = straighten(bgr, scale)
    wb = white_balance(bgr, scale) if not opt.keep_colour else bgr
    gray, chroma = luminance(wb, drop_colour=not opt.keep_colour,
                             colour_kill=opt.colour_kill)
    norm = flatten(gray, scale)

    # Provisional pass: find the text BEFORE deciding what is a stamp, so the
    # stamp stage can be told where it must not touch. Costs one extra
    # threshold+RLSA, and it is the only thing standing between a sepia page
    # and having its faintest words erased as "coloured ink".
    prov = ink_mask(norm, scale)
    pgh, _, *_ = glyph_metrics(prov)
    protect = text_runs(prov, pgh)[0] if pgh and pgh >= 3 else prov
    stamp, cast = colour_ink(chroma, norm, protect, prov, scale, thresh=opt.chroma,
                             max_page_frac=opt.max_stamp, dark_floor=opt.ink_floor)
    norm[stamp > 0] = 255                    # clear coloured ink before analysis

    mask = ink_mask(norm, scale)
    gh, ga, *_ = glyph_metrics(mask)
    if gh is None or gh < 3:
        return dict(norm=norm, gh=None, stats={"note": "no text found; passed through"})
    stats = {"glyph_h": round(gh, 1), "glyph_area": round(ga, 1)}
    if ang:
        stats["deskew"] = ang

    mask, border = drop_border_artifacts(mask, gh, scale)
    # Before text detection, so neither can be mistaken for type and protected.
    sr, dt, lab, n, cst = stroke_radius(mask, gh)
    mask, dust = dust_discs(mask, norm, gh, sr, dt, lab, n, cst)
    mask, thick = blob_bodies(mask, gh, ga, sr)
    zone, core, runs = protection_zone(mask, gh, up=opt.up, down=opt.down, side=opt.side)
    orn = ornament_rules(mask, gh)
    # Printed furniture that must survive every margin rule. The horizontal
    # ornaments join the lines of type as well, being lines themselves; the
    # border sides only join `core`, since counting them as lines is what
    # blinds the line-based tests (see frame_rules).
    keep = cv2.bitwise_or(orn, frame_rules(mask, gh))
    core, runs = cv2.bitwise_or(core, keep), cv2.bitwise_or(runs, orn)
    box = text_block(core, gh, scale) if not opt.no_crop else (0, 0, mask.shape[1], mask.shape[0])
    # Printed furniture is inside the frame too, literally: a border is the
    # outermost thing on the sheet, so a crop drawn round the type alone paints
    # it white having gone to the trouble of keeping it.
    box = guard_box(box, cv2.bitwise_or(runs, keep), gh)
    mask, edge = edge_junk(mask, core, runs, zone, gh, ga, band_frac=opt.band, keep=keep)
    mask, blot = blots(mask, core, gh, ga)
    mask, crease = scratches(mask, runs, gh, ga, core, keep=keep)
    bl, br = edge_bands(cv2.bitwise_or(mask, border), runs, gh, keep=keep)
    mask, band = clear_bands(mask, runs, gh, bl, br, keep=keep, ga=ga)
    mask, specks, spared = despeckle(mask, norm, core, zone, gh, ga,
                                     max_area_frac=opt.speck, faint=opt.faint)

    # Second opinion from a text detector: put back line members too small to
    # be protected, then clear what lies outside the column and no line owns.
    det = detect_text(bgr) if getattr(opt, "detect", True) else None
    rescued = gate = np.zeros_like(mask)
    if det is not None:
        dmask, dboxes = det
        # any surviving ink will do as the neighbour: a letter fused with a
        # tear is no longer glyph-shaped, but its full stop still belongs to it
        kept = mask
        rescued = cv2.bitwise_or(
            rescue_marks(specks, dmask, norm, chroma, kept, gh),
            rescue_marks(cv2.bitwise_or(band, edge), dmask, norm, chroma, kept, gh,
                         near=True, size=1.6, reach=1.2))
        back = cv2.bitwise_not(rescued)
        specks, band, edge = (cv2.bitwise_and(m, back) for m in (specks, band, edge))
        mask = cv2.bitwise_or(mask, rescued)
        mask, gate = detector_gate(mask, dmask, dboxes, keep, gh)
        stats.update(det_boxes=len(dboxes),
                     rescued=int(cv2.connectedComponentsWithStats(rescued, 8)[0] - 1),
                     gate_px=int((gate > 0).sum()))

    stats.update(border_px=int((border > 0).sum()),
                 specks_removed=int(cv2.connectedComponentsWithStats(specks, 8)[0] - 1),
                 specks_spared=spared, box=box, stamp_px=int((stamp > 0).sum()),
                 dust_px=int((dust > 0).sum()), thick_px=int((thick > 0).sum()),
                 edge_px=int((edge > 0).sum()), blot_px=int((blot > 0).sum()),
                 crease_px=int((crease > 0).sum()), band_px=int((band > 0).sum()),
                 bands=(bl, br))
    if cast > opt.max_stamp:
        stats["stamp_stood_down"] = f"{cast:.2f} (colour cast or chromatic ink)"

    kill = border
    for m in (stamp, dust, thick, edge, blot, crease, band, specks, gate):
        kill = cv2.bitwise_or(kill, m)
    # Anything glyph-shaped that we deleted is worth flagging: it is the one
    # error class that matters here, and it is invisible in a thumbnail.
    stats["core_removed"] = int(cv2.connectedComponentsWithStats(
        cv2.bitwise_and(kill, core), 8)[0] - 1)
    stages = dict(border=border, stamp=stamp, dust=dust, thick=thick, edge=edge,
                  blot=blot, crease=crease, band=band, specks=specks, gate=gate,
                  rescued=rescued)
    return dict(norm=norm, mask=mask, kill=kill, core=core, runs=runs, zone=zone,
                box=box, gh=gh, ga=ga, stats=stats, stages=stages)


def clean_page(bgr, dpi=300, opts=None):
    """Clean one BGR image and return ``(image, stats, audit, output_dpi)``."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    opt = opts if opts is not None else Options()
    a = analyse(bgr, dpi, opt)
    if a["gh"] is None:
        return a["norm"], a["stats"], None, dpi
    norm, kill, box, stats = a["norm"], a["kill"], a["box"], a["stats"]
    scale = dpi / 300.0
    out = compose(norm, kill, box, black=opt.black, white=opt.white, soften=opt.soften,
                  ink=a["mask"] if opt.paper_floor else None)

    # sharpen at native resolution first - that is where the real stroke edges
    # are - then resample, then a lighter pass to crisp the interpolation
    out = sharpen(out, amount=opt.sharpen, radius=1.0 * scale)
    if opt.upscale > 1:
        out = supersample(out, opt.upscale)
        out = sharpen(out, amount=opt.sharpen * 0.5, radius=1.0 * scale * opt.upscale)
        dpi = dpi * opt.upscale
    if opt.bilevel:
        out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    stats["out_px"] = f"{out.shape[1]}x{out.shape[0]}"

    audit = None
    if opt.audit:
        audit = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        fat = cv2.dilate(kill, np.ones((5, 5), np.uint8))
        audit[fat > 0] = (0, 0, 255)                  # every deleted pixel, in red
        x0, y0, x1, y1 = box
        cv2.rectangle(audit, (x0, y0), (x1 - 1, y1 - 1), (255, 128, 0), 3)
    return out, stats, audit, dpi


def _flate_stream(arr, bilevel):
    """Compress a page image, packing to 1 bit per pixel when bilevel."""
    if bilevel:
        bits = (arr > 127).astype(np.uint8)
        packed = np.packbits(bits, axis=1)      # rows padded to byte boundary
        return zlib.compress(packed.tobytes(), 9), 1
    return zlib.compress(arr.tobytes(), 9), 8


def make_pdf(pages, dpi_list, bilevel=False, transparent=False, matte=False):
    """Return PDF bytes containing losslessly compressed page images.

    Not via Pillow: its PDF plugin hard-codes DCTDecode for greyscale, and JPEG
    rings around exactly the high-contrast edges this whole tool exists to keep
    crisp. The text is black on white with hard edges - the worst case for a DCT
    - so the artefacts read as softening and grey haloes at print size.

    `transparent` emits the ink alone, with the paper genuinely absent rather
    than painted white: a 1-bit stencil (/ImageMask) when bilevel, an alpha
    channel (/SMask) when greyscale, which keeps the antialiasing. `matte` then
    paints a white rectangle behind it, so the page looks exactly like the
    opaque version while the ink stays a separate layer.
    """
    objs = [None]                                # 1-indexed

    def add(body):
        objs.append(body)
        return len(objs) - 1

    catalog = add(b"")                           # patched once Pages is numbered
    pages_obj = add(b"")
    kids = []
    for arr, dpi in zip(pages, dpi_list):
        h, w = arr.shape
        extra = ""
        if transparent and bilevel:
            # stencil: sample 0 paints the current fill colour, 1 paints nothing
            data, _ = _flate_stream(arr, True)
            hdr = (f"<</Type/XObject/Subtype/Image/Width {w}/Height {h}"
                   f"/ImageMask true/Decode[0 1]"
                   f"/Filter/FlateDecode/Length {len(data)}>>")
        elif transparent:
            alpha = (255 - arr)                  # opacity of the ink
            adata, _ = _flate_stream(alpha, False)
            smask = add((f"<</Type/XObject/Subtype/Image/Width {w}/Height {h}"
                         f"/ColorSpace/DeviceGray/BitsPerComponent 8"
                         f"/Filter/FlateDecode/Length {len(adata)}>>\nstream\n").encode()
                        + adata + b"\nendstream")
            black = np.zeros_like(arr)
            data, _ = _flate_stream(black, False)
            hdr = (f"<</Type/XObject/Subtype/Image/Width {w}/Height {h}"
                   f"/ColorSpace/DeviceGray/BitsPerComponent 8"
                   f"/SMask {smask} 0 R"
                   f"/Filter/FlateDecode/Length {len(data)}>>")
        else:
            data, bpc = _flate_stream(arr, bilevel)
            hdr = (f"<</Type/XObject/Subtype/Image/Width {w}/Height {h}"
                   f"/ColorSpace/DeviceGray/BitsPerComponent {bpc}"
                   f"/Filter/FlateDecode/Length {len(data)}>>")
        img = add(hdr.encode() + b"\nstream\n" + data + b"\nendstream")

        # points, so the sheet keeps its physical size however many pixels it has
        pw, ph = w * 72.0 / dpi, h * 72.0 / dpi
        ops = []
        if matte:
            ops.append(f"1 1 1 rg 0 0 {pw:.4f} {ph:.4f} re f")
        if transparent and bilevel:
            ops.append("0 g")                    # stencil paints in this colour
        ops.append(f"q {pw:.4f} 0 0 {ph:.4f} 0 0 cm /Im0 Do Q")
        content = "\n".join(ops).encode()
        cont = add(f"<</Length {len(content)}>>\nstream\n".encode()
                   + content + b"\nendstream")
        page = add((f"<</Type/Page/Parent {pages_obj} 0 R"
                    f"/MediaBox[0 0 {pw:.4f} {ph:.4f}]"
                    f"/Resources<</XObject<</Im0 {img} 0 R>>>>"
                    f"/Contents {cont} 0 R>>").encode())
        kids.append(page)

    objs[catalog] = f"<</Type/Catalog/Pages {pages_obj} 0 R>>".encode()
    objs[pages_obj] = ("<</Type/Pages/Kids[" + " ".join(f"{k} 0 R" for k in kids)
                       + f"]/Count {len(kids)}>>").encode()

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, body in enumerate(objs[1:], start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<</Size {len(objs)}/Root {catalog} 0 R>>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def write_pdf(pages, dpi_list, path, bilevel=False, transparent=False, matte=False):
    """Write :func:`make_pdf` output to *path*."""
    with open(path, "wb") as f:
        f.write(make_pdf(pages, dpi_list, bilevel, transparent, matte))
