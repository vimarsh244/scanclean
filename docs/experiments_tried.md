# Experiments tried and implementation details

Print-ready cleanup for scanned books, tuned for Indic scripts. White background,
noise and stamps gone, **every diacritic preserved**.

```bash
scanclean original/*.pdf -o cleaned                        # greyscale, safest
scanclean original/*.pdf -o out --upscale 2 --bilevel      # best for printing
scanclean original/*.pdf -o cleaned --audit                # + overlay of deletions
```

Requires Python 3.10+, `opencv-python` 5.0.0.93 or newer (but below 6),
`pillow`, `numpy`, and Poppler (`pdfimages`, `pdfinfo`, `pdftoppm`) on `PATH`.
OpenCV 5 is required because its DNN importer loads the bundled
`scanclean/models/ppocrv6_tiny_det.onnx` detector. The detector can still be
disabled explicitly with `--no-detect`.

Install the Python dependencies with `python -m pip install -r requirements.txt`.

## The core problem

These scans are ~220 dpi against the real page. At that resolution a Gujarati
anusvāra is **3–6 px across — the same size as the dust speckles.** Every
off-the-shelf despeckler (ImageMagick `-despeckle`, ScanTailor, `unpaper`,
Acrobat's "remove background") filters by blob size, so on this material they
strip bindis along with the dirt, silently. That is why this is a custom tool
and not a one-line ImageMagick call.

So **nothing is ever removed on size alone.** A blob is removed only when its
*context* proves it is not text:

| Signal | Rule |
| --- | --- |
| Position | Outside a protection zone grown around every real line of type |
| Line membership | Text is serial — many blobs strung along a line; dirt is isolated |
| Density | Letterpress ink bottoms out near 0; paper texture never gets below ~150 |
| Chroma | Type is neutral (saturation 2–5); stamps are violet (44–77) |
| Extent | Damage runs down an edge as a chain; a page number stands alone |
| Shape | A 4:1 horizontal mark is a dash, not dust |
| Thickness | No letter holds a disc 3.5× its stroke width; a blot does |
| Softness | Printed dots are ink-black; scanner-glass dust is a grey blur |
| Detector | A 1.8 MB text detector boxes lines of type and never boxes torn edges |

The failure mode is deliberately asymmetric: **a speck survives** rather than
**a bindi disappears**. Some dirt close to letters is therefore left behind.
That is the intended trade, not an oversight.

## Verifying, rather than trusting

`--audit` writes `<name>.audit.pdf` alongside the output, painting **every
deleted pixel red** over the original with the crop box in blue. Flip through it
and any lost text is obvious at a glance.

Each page also reports `core_removed` — how many *glyph-shaped* things were
deleted. It should be small and confined to torn edges. A jump means a page hit
a case the heuristics misread, and the audit will show which.

This is how the tool was validated. Rendering thumbnails of every glyph-shaped
deletion across the 29 sample pages caught bugs that page-level inspection missed
entirely: small-type quotation lines taken for dirt, a page number losing a digit,
short words at line ends deleted at the margins, and a running head losing its
vowel signs (`ઓપેરા.` → `ઓપગ`) because a matra is too small to register as a line
member and the margin rule then ate it. None were visible at thumbnail scale.

`compare/index.html` (rebuild with `python3 build_compare.py`) is the other half of
this: every page as a before/after slider at native resolution, which is how the
surviving creases and the clipped running head were spotted in the first place.

Run BOTH audits after changing any threshold. Between them they have caught every
real defect found so far; neither catches all of them alone:

```bash
scanclean original/*.pdf -o cleaned --audit              # per-pixel overlay
python3 audit_lines.py                                   # ink lost from lines of type
```

### Counting erased characters

Both audits above are read by eye, and both measure the pipeline against its own
opinion of where the text is — which is no help at all when that opinion is what
went wrong. Three further documents (samples 5–7) were added later and broke
exactly that way: whole first words, last words and table columns were deleted,
and every internal counter read clean, because the stages that deleted them had
first decided the text was not text.

So score against a witness that has no stake in it. `audit_glyphs.py` counts the
deleted marks that are glyph-shaped *and* that the bundled PP-OCRv6 detector
independently places inside a line of type, and attributes each to the stage
that took it:

```bash
python3 audit_glyphs.py                                  # characters erased, by stage
```

It is blunt — an ornament inside a detected box counts, and a character the
detector missed does not — but it is a number, it needs no eye, and across the
46 sample pages it moved from 236 to 38. That is the measurement every claim
below rests on.

## Where a neural network helps, and where it does not

**It does not paint pixels.** A generative model (GAN, diffusion) produces
plausible pixels - on a page where a bindi is 4 px, plausible means it can add
or drop bindis, which breaks the one hard rule. See *Tried and rejected* below:
DocDiff was tested, both as published and fine-tuned on these books.

**It does answer one question well: where are the lines of type?** That was
the question the classical line finder could not answer at a torn edge, where
damage shatters into letter-sized pieces and chains itself onto the lines
(the "laundering" that made edge junk look like text). PP-OCRv6 tiny - a
0.43M-parameter DB text detector, Apache-2.0, 1.8 MB as ONNX, run through
OpenCV's own DNN module in ~0.15 s a page - boxes every line on all 29 sample
pages and none of the edge damage. It is used strictly as a second opinion:

- **Gate.** Ink outside the *local* text column (the extent of the detected
  lines within 8 glyph heights of it) that touches no detected line, within a
  glyph's reach, is removed. Local, because verse set short beside a torn
  edge must not borrow the width of prose lower down the page. Printed rules
  and ornaments are exempt - the detector does not box them.
- **Rescue.** Ink-black, diacritic-sized marks that the detector places
  *inside* a line but the despeckler, edge or band stages removed are put back.
  It found real losses the other audits had missed: an anusvara (`પાંઉમેસ`
  had become `પાઉમેસ`), two full stops set high and loose at line ends, and a
  short dash. It also brings back a few dust dots that sit inside lines - the
  usual trade.

Every gate removal (151 regions) and every rescue (26) across the 29 pages was
reviewed as a before/after contact sheet (`experiments/verify_stage.py`).

## Why not ImageMagick?

Checked, and the useful parts are already here in OpenCV:

- **Deskew** — ImageMagick's `-deskew` (and Leptonica's) scores candidate
  angles by how sharp the row profile of the ink is. `scanclean` does exactly
  that, on by default. Measured: S2 pages 4 and 6 were tilted 1.5–1.6°.
- **Despeckle / morphology / `-trim`** — size- or bounding-box-based, so they
  hit the anusvāra problem above, or crop to junk at the sheet edge.
- **Background removal** (`-level`, `-fuzz`, `-white-threshold`) — global
  thresholds; the per-pixel flatten + paper floor here is local and ink-aware.

Adding it would be a second image stack for no capability gain.

## What it does per page

0. **Straighten.** Projection-profile deskew (≤3°), before any analysis, so
   lines are level and margins vertical for every rule that follows.
1. **White-balance each channel** against its own paper level. Required before
   any colour test — raw chroma cannot tell a violet stamp (53) from cream paper
   (45); after normalising, paper of any tint goes neutral and only real
   coloured ink keeps chroma.
2. **Suppress coloured ink.** Stamps, blue pen. Black hand-writing is kept — it
   is indistinguishable from type by colour, so guessing would risk real marks.
3. **Flatten the background** to uniform white, keeping ink density.
4. **Find text lines** (RLSA) at *any* point size, so footnotes and small-type
   headings are protected like body text.
5. **Drop dust discs and blot bodies** before text detection, so neither can be
   mistaken for a glyph and protected. Dust: round, solid, thicker than any
   stroke and never ink-black. Blots: whatever survives an opening with a disc
   no letter could contain — which removes nothing thin, so a letter a blot has
   swallowed keeps its strokes.
6. **Clear the edge bands.** A text column is blank *between* lines; a torn
   edge is inked continuously down the sheet. Columns whose inter-line gaps are
   ≥15% inked (text columns measure 1–8%) form a band, and marks lying wholly
   inside it go — unless chained to kept text within a *word* gap, so a
   line-initial or line-final word stays with its line. This is what finally
   handles damage that shatters into letter-sized pieces and chains itself onto
   the lines. Three guards stop it doing the opposite (see below): lines of
   type that run on into the band, printed furniture, and a cap on how many of
   the page's letters a side may cost.
7. **Clear the margins** of tears, folds and spine shadow.
8. **Despeckle contextually** — the careful bit, per the table above.
9. **Remove blots** — anything too big to be a letter that belongs to no line.
10. **Remove creases** — long thin fold lines. These defeat every other rule by
   being faint: thresholding shatters one crease into a chain of fragments, each
   individually glyph-sized. Reconstructed with a vertical closing, the object is
   six glyph-heights long and one wide — which script *does* produce, in the
   narrow column of a table, so a chain made mostly of letters is spared.
11. **Consult the text detector** (see above): rescue line members the
   despeckler took, then clear what lies outside the local column that no
   detected line owns.
12. **Compose** from the continuous-tone image, never from the binary mask, so no
   thresholding decision can thin a stroke. Paper → pure white, text → full
   antialiased greyscale. Anything more than 2 px from surviving ink is set to
   white — the "paper floor" — which removes grey mottle at worn edges that was
   never dark enough to be ink, and so was never removed as dirt.
13. **Sharpen** with an edge-gated unsharp mask, optionally supersampling first.
14. **Write the PDF losslessly** (FlateDecode).

## Options worth knowing

| Flag | Default | Effect |
| --- | --- | --- |
| `--bilevel` | off | 1-bit. Much smaller; slightly bolder |
| `--upscale` | 1 | Integer resample factor. `2` doubles pixel density, same physical page |
| `--sharpen` | 0.8 | Unsharp amount on stroke edges. `0` disables |
| `--audit` | off | Overlay PDF of every deletion |
| `--speck` | 0.55 | Max speck area, as a fraction of median glyph area. **Lower = safer** |
| `--faint` | 150 | In-zone blobs never darker than this are dirt. `0` disables |
| `--up/--down/--side` | .85/.65/.55 | Protection zone reach, in glyph heights. **Raise = safer** |
| `--band` | 0.09 | Margin width cleared of non-text ink |
| `--no-crop` | off | Keep full sheet, do not clip margins |
| `--keep-colour` | off | Leave stamps alone |
| `--no-deskew` | — | Leave tilt alone (deskew is on by default, ≤3°) |
| `--no-paper-floor` | — | Keep sub-ink grey instead of whitening it |
| `--no-detect` | — | Skip the text-detector rescue and gate |
| `--transparent` | off | Ink only, paper genuinely absent |
| `--matte` | off | With `--transparent`, paint white behind the ink |
| `--max-stamp` | 0.04 | Above this page fraction, "stamp" is a colour cast; stage stands down |

To be maximally conservative: `--speck 0.25 --faint 0 --up 1.2 --down 1.0`.

## On sharpness

Three separate things were costing edge quality, only one of which is resolution:

1. **The output was JPEG.** Pillow's PDF writer hard-codes lossy DCTDecode for
   greyscale, and a DCT rings around hard black-on-white edges — visible as grey
   haloes and softening. The tool now writes its own PDF with **FlateDecode**,
   so the pixels are exactly what the pipeline produced.
2. **The scan is optically soft.** An edge-gated unsharp mask restores stroke
   definition; gating on local gradient keeps it from amplifying paper grain.
3. **Resolution is genuinely capped.** The embedded JPEGs are the originals —
   `--upscale` cannot add detail that was never captured. What it does buy is a
   finer grid for the sharpened edges and for the printer's halftoning, so it
   looks better on paper while being honest about the source.

For print I'd use `--upscale 2 --bilevel`.

## Transparent background — does it make sense?

For printing on white paper, **no**: transparent and white look identical, because
"white" is just unprinted paper either way. `--transparent` is worth it for a
different reason — it makes the background a *rendering* decision instead of baked
pixels, which matters if pages get composited, reprinted on tinted stock, or
overlaid on a template.

It is implemented the way a scanner would:

- `--transparent --bilevel` → a **1-bit /ImageMask stencil**. The ink is painted
  in the current fill colour and the paper is never painted at all. Measured 71%
  of the page genuinely absent, at the same file size as opaque 1-bit.
- `--transparent` (greyscale) → an **/SMask alpha channel**, so antialiasing is
  preserved. The base image is constant black and compresses to under 2 KB;
  all the information lives in the mask.
- Add `--matte` to paint white behind the ink layer. This is the "transparent,
  then add a white background" case — it renders identically to the default while
  keeping the ink as its own layer.

Caveat worth knowing: some older print RIPs and PDF viewers handle transparency
badly, showing grey or a checkerboard. The default stays opaque white for that
reason — reach for `--transparent` when you actually want to composite.

## Tried and rejected: learned restoration

Each was run on these pages (`experiments/`, upstream code in `third_party/`):

| Method | What it is | Result here |
| --- | --- | --- |
| DocDiff, published weights | 8M-parameter residual diffusion model (seal-removal and deblur checkpoints) | Left the violet stamp in place - it was trained on red Chinese seals - and added visible 128 px tile seams |
| DocDiff, fine-tuned | Same model, 5,000 steps on 24 pages, scan -> scanclean output; scored on 5 pages it never saw | Dropped two real full stops (`ઓપેરા.`, `છે.`), turned two grey dust spots into ink-black dots that read as bindis, reshaped one matra; left 150-370 px of edge ink per page where scanclean leaves none |
| ZigZag | ML-free two-pass local-mean background normalisation | As good as scanclean's own flatten on paper; edge damage survives as foreground, as it does for any thresholder |
| SauvolaNet | 40K-parameter learned Sauvola binariser | Clean binarisation, no better than `--bilevel`; keeps dust and stamps, since it has no notion of what text is |

The fine-tune is the telling result. Taught by this tool, on these books, it is
worse than its teacher on unseen pages in exactly the class of mark that
matters: to a network, a full stop and a speck of dust are both small dark dots,
and nothing in the pixels says which one to keep. That distinction only exists
in context - which line the dot belongs to - and context is what the detector
supplies without ever drawing a pixel.

## Tried and rejected: detecting tears by paper level

A tear is a piece of page that is missing or folded, so what shows through sits
at a different brightness from the intact sheet. Detecting that — rather than
reasoning about ink, as every other stage does — looked like the way to remove a
torn flap's ragged boundary, which survives all the ink-based rules.

It works on the page it was designed for and is unsafe everywhere else. Measured
share of each page claimed as "damaged":

| Page | Claimed | Reality |
| --- | --- | --- |
| Sample 1 p6 | 1.6% | the actual tear |
| Sample 3 p1, p3, p5, p7 | 6–8% | merely yellowed paper |
| Sample 2 p7 | **39%** | a sepia page with uneven tone |

Whitening 39% of a page to remove a tear is not a trade worth making, and the
confound is the same one that has already caused two bugs here: **these books'
discolouration looks like damage to any global appearance measure.** Chroma hit
it, brightness hits it. Only local structure — is this thing long and thin, does
it belong to a line of type — has proved safe on this material.

So the ragged edge of a torn flap stays, on the pages that have one.

## What samples 5–7 broke, and why

Three more documents were added after the tool was working: a small-format
prose book set to the trim (5), a book printed inside a ruled border (6), and
one with an ornamental border and several errata tables (7). Nearly every erased
character was on one of those three: the 29 pages the tool was tuned on
contributed 11 of the 248 counted at the time, and the 17 new ones the rest.
Every one of the causes is the same mistake in a different costume — **a stage
that deletes by position, acting on evidence that had quietly stopped meaning
what it was measured to mean.**

**The margin band drawn across the text**, which accounted for six of every
seven. `edge_bands` scores each
column by how often it carries ink *between* lines of type, which separates a
torn edge (0.2–0.8) from a text column (0.01–0.08) — on the documents it was
measured on. Set Gujarati densely and there is barely an inter-line gap to read:
the upper matras and conjunct stems of the next line stand in it, and a column of
ordinary type scores 0.10–0.21. The stage reads columns one at a time and takes
the *innermost* one over the threshold, so a single such column anywhere in the
outer fifth of the sheet pulls the band's inner edge out to meet it, condemning
everything between. On sample 5 that was the first word of every line on page 4,
and the last word of every line on page 5.

Three guards, because one was not enough and each catches a different shape of
the mistake:

- *The lines must stop short of it.* Damage lies outside the type area, so the
  lines of type end before it starts. Measured: a true edge strip is reached by
  0–12% of the page's lines (the few whose ends the damage has chained onto), a
  misplaced band by 31–76%.
- *Printed furniture is not evidence.* The side of a border box is inked in
  every gap between every line — the signature exactly — so a bordered page
  reads as damaged down both sides, with the band stopping where the border is.
- *And then count the cost.* Rather than enumerate the remaining ways the
  column evidence can mislead, measure what each side actually deleted: a strip
  of damage holds none of the page's type and takes 0.0–2.2% of its letters,
  while a band that has reached a column of figures takes 3.7–16%. Over 3%, the
  side stands down and the margin stays dirty.

The rescue was widened to match. A word inside the band has to reach across the
space before or after it to be recognised as part of its line, and the reach was
0.8 glyph heights against word gaps of 1.2–2.0. And where the line finder cannot
confirm the type at all — the figures down the last column of a table of
contents are each too short to be a line — the detector's verdict is taken
instead, for whole glyphs and not just diacritics, still hedged by darkness,
colour, and having to stand beside surviving ink. A column of such figures
vouches for itself, one figure at a time, which is the same serial argument the
line finder is built on.

**Printed borders deleted as damage.** A vertical rule is tens of glyphs tall
and a few pixels wide, standing in the margin — the shape of a crease, a torn
flap, a spine shadow, in the place they occur. So `edge_junk`, `blots`,
`scratches` and the band stage each took it in turn, and sample 6 came back with
three sides of its border, or none. What no damage reproduces is that a rule was
*drawn*: measured, a printed rule strays 0.03–0.06 of a glyph height from its own
centre line and varies 0.17–0.24 in thickness, while edge damage of the same
length strays 0.17 and varies 0.98. `frame_rules` finds them in both directions
and the margin stages are given the result to leave alone.

It has to be kept *apart* from the horizontal ornaments, which are folded into
the lines of type. Doing the obvious thing and folding the border in with them
cost more than it fixed: smeared to bridge word gaps, a rule the height of the
page welds every line on the sheet into one run, the page reads as having four
lines on it, and every rule that counts lines goes blind. That mistake alone put
13 characters back on the floor before it was caught.

**A table column read as a crease.** `scratches` exists because a faint crease
shatters into glyph-sized fragments that are individually indistinguishable from
letters, and argues that together they form "a line six glyph-heights long and
one wide, which no script produces". Script does produce it: the narrow column
of an errata table, a file of ditto marks. The distinction is what the chain is
*made of* — measured, a chain that is 62–100% glyph-shaped, glyph-sized pieces is
always type and one that is 0–19% is always damage, with nothing in between.

**Words eaten hollow where the impression was weak.** The despeckler's second
tier rests on "ink bottoms out near 0, so a mark that never approaches ink is
dirt", and its first tier on "nothing that belongs to a line of type can be
outside the protection zone" — which applies no darkness test at all. Both
readings are of the mark alone, and both fail on a lightly inked word: its
letters break into pieces, none is confirmed, the zone grown from the confirmed
ones has a hole where the word is, and pieces bottoming out at 44 — ink by any
standard — were deleted for sitting in the hole.

What the mark alone cannot show, its surroundings can. Followed at the level
where paper stops being paper, a detached piece of a letter is joined to the
rest of that letter by ink too faint for the ink mask and far too dark to be the
sheet; a speck of dirt sits on clean paper and its halo is an island. Measured,
75% of the marks taken out of weakly printed words are joined this way against
0–35% of those taken off open paper. Both tiers now consult it.

## Known limits

- **Text on a folded flap is kept.** A torn neighbouring page folded over
  this one carries real print; it stays, fold line and all where the line
  runs through a letter's stem.
- **Page size.** These PDFs declare a 1.86″×3.36″ page, so they print tiny. The
  tool preserves the original geometry; scale at print time, or ask and it can
  be retargeted to A4/A5.
- **Black hand-written accession numbers survive** stamp removal, by design (2).
- **Dirt touching a letter stays.** Removing it means risking the letter.
- **A stamp printed over text** is cleared around the text, not reconstructed
  underneath it — the strokes beneath are kept, so the text stays readable.
- **A stamp whose own ring carries text** (a circular library stamp) may survive,
  because the line detector correctly recognises it as text and protects it. This
  is the safe side of the trade.
- **Sepia pages disable stamp removal.** Where the type itself carries chroma
  above the threshold, colour cannot identify a stamp; the tool stands down and
  reports `colour_cast` rather than guessing.
- **Letters missing from the scan stay missing.** Sample 5 page 4 is clipped at
  the left in the source: its lines begin mid-character and the first letters
  were never captured. The clipped remnants are kept, and nothing can restore
  what the scanner did not see.
- **A faint impression prints faint.** Where a word took less ink than its
  neighbours, its strokes survive but stay pale — the tool has no business
  darkening ink it did not measure. `--white 236` develops the mid-tones if a
  printer needs more punch; it costs some paper texture in exchange.
- **Corner ornaments on a border may go.** `frame_rules` finds the straight
  sides, not the fleurons at the corners, which are compact marks far outside
  the type area and indistinguishable from a blot by shape.
- **An ornamental border made of repeating motifs is not protected** — only
  plain drawn rules are. Sample 7's title page loses its frame.
