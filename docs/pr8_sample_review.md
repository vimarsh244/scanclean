# PR #8: review against the supplied PDFs

Compared all 17 pages of the supplied Sample 5 (5 pages), Sample 6 (6 pages),
and Sample 7 (6 pages) against `main` at `ece59af`. The initial PR was
`7b6ebc7` on `improve-char-erasure`. Originals are stored as `original/Sample
5.pdf`, etc.; originals, generated PDFs, and comparison images remain ignored.

## What improved

Sample 5's first and last words, its table-of-contents figures, and Sample 7's
narrow table columns survive much better. The faint-print fix in the initial
PR preserves some fragments, but does not fully resolve the first screenshot.
The second screenshot's left-margin deletion improves substantially, but its
page number still disappeared in the initial PR.

Two additional rendering fixes are included in this review:

- Extend the final crop to contain surviving ink confirmed by the text
  detector. Sample 5 pages 3 and 4 retained their page numbers in the cleaning
  mask but lost them when composition whitened everything outside the crop.
- Let the paper floor follow faint ink connected to surviving strokes at the
  same 236-level halo used by despeckling. Previously a faint stroke could
  survive deletion and still disappear because it lay more than two pixels
  from the thresholded ink mask. Isolated texture remains white, and explicit
  deletions still receive their whitening/feathering.

Synthetic tests cover both rendering failures, halo-based fragment protection
versus isolated dust, and crop-only losses in the audit.

## Measurements

The audit counts glyph-shaped removed components inside independently detected
text. It now also counts ink lost to the crop, and runs its detector on the
same deskewed image as the pipeline. Earlier audit numbers cannot be compared
with this corrected instrument directly.

| Supplied document | Pages | Main | Updated PR | Reduction |
|---|---:|---:|---:|---:|
| Sample 5 | 5 | 172 | 2 | 98.8% |
| Sample 6 | 6 | 56 | 20 | 64.3% |
| Sample 7 | 6 | 47 | 7 | 85.1% |
| Total | 17 | 275 | 29 | 89.5% |

These are **suspected glyph erasures, not recognized characters or a percentage
of text correctly preserved**. The detector and component-size filters miss
some genuine losses and can count dirt inside a text box. In particular,
Sample 5 page 2 reports zero in both versions even though its faint-word
rendering visibly differs. Tone clipping and loss of small stroke fragments
require visual inspection in addition to this counter.

At default settings, the final output preserves Sample 5 page 4's numeral 3
and the source's surviving left-edge text. The source itself cuts through
some leftmost letters: their missing parts cannot be recovered.

Weakly inked words still look faint and incomplete in places. `--white 236`
retains more midtones and is provided as an additional Sample 5 output for
inspection; it may also retain more texture. Some printed border details and
ornamental corners still disappear, especially on Sample 6. Sample 6 page 6
still contributes 16 suspect marks; this review does not claim complete
preservation of all text or furniture.

## Reproduce and validate

```bash
python tools/compare_pr.py --base-ref ece59af --samples 5 6 7
python -m scanclean 'original/Sample 5.pdf' -o cleaned --white 236
python -m pytest -q
python -m build --no-isolation
```

The comparison tool writes native-resolution PNGs, base/current PDFs, a
page selector with adjustable display width in `work/pr8-review/index.html`,
and per-page JSON metrics. Nothing generated is committed.

Validation: 39 tests passed; six tests requiring unavailable Samples 1–4
were skipped. All pre-existing numerical tolerances passed before remeasuring
the three available baselines; their expected output hashes changed because
rendering intentionally changed. Samples 1–4 retain their numerical baselines,
but their now-stale exact output hashes are explicitly pending (`null`);
`tools/refresh_baselines.py` restores them when all seven originals are present.
Both the wheel and source distribution built successfully. This environment
lacks Python's `ensurepip`, so the build used the already-provisioned project
environment with `--no-isolation`.

## Follow-up: opt-in strengthening of faint ink

The next review of Sample 5 page 2 confirmed that preserving fragments alone
still leaves weak words too pale. `--restore-ink` now adds a rendering layer
before the tone curve. It detects dark ridges relative to nearby paper within
confirmed text, then increases their contrast with a soft signal gate. It
draws from measured grayscale values rather than reconstructing glyphs from
OCR or applying morphological dilation. With the detector disabled or
unavailable, the existing text protection zone provides the region.

The flag alone uses strength 1; `--restore-ink 0.5` is a gentler setting and
`Options(restore_ink=1)` enables it through the Python API. It is off by default
because it intentionally strengthens print; texture within text can also be
strengthened if it resembles a stroke. Explicit deletion masks and their
one-pixel neighbourhood are excluded from enhancement. Audit overlays retain
the source grayscale rather than the enhanced grayscale.

Compared all 17 supplied pages against the previous PR commit, `7b056a4`, with
restoration at strength 1. The erasure audit is unchanged (2, 20, and 7 for
Samples 5, 6, and 7); that instrument counts deletion decisions and does not
measure stroke visibility. The dedicated comparison is written to ignored
`work/ink-restoration/review/` with both versions of all three PDFs.

In the weak-word crop on Sample 5 page 2 (`[60, 900, 860, 1400]`), 1,375 pixels
with faint source evidence (normalized grayscale 140–235) that previously
rendered at 245 or above now render below 224. Dark pixel area below 160 grows
from 28,272 to 40,841. These figures describe the intentional contrast change,
not character accuracy. Visual inspection shows more readable weak strokes;
some genuine breaks in the impression still remain.

Validation for this follow-up: 50 tests passed, six absent-fixture tests skipped;
wheel and source distribution builds passed. Tests include the actual faint
region of Sample 5 page 2, white letter openings and gaps, flat/noisy paper,
unchanged exterior pixels, explicit deletions, strength controls, CLI parsing,
and retaining source values in audits. Default outputs and baseline records
are unchanged by this opt-in feature.

## Follow-up: the missing દૂ in દૂર કર્યું હતું

On Sample 5 page 2, the upper and middle strokes of દૂ were incorrectly removed
as specks. The source's 236-level halo joins those fragments to the surviving
lower stroke, but that lower stroke was not part of the confirmed line core.
The ordinary halo protection therefore failed, and the original per-fragment
rescue rejected the deleted strokes because they were not dark enough.

With ink restoration enabled, `rescue_faint_glyphs` now groups the source ink
inside detected text. A group must have glyph-sized dimensions, a non-solid
shape, sufficient surviving ink support, and deleted speck pixels. Only those
speck pixels are put back; the enhancement then develops their measured ink.
No isolated group without surviving ink is restored, and the other removal
stages are unchanged. The feature remains opt-in through `--restore-ink`.

Compared the previous restoration (`be729c2`, strength 1) with the new one
(strength 1) across all 17 supplied pages:

```bash
python tools/compare_pr.py --base-ref be729c2 --base-restore-ink 1 \
  --restore-ink 1 --out work/ink-restoration/review
```

The refreshed Sample 5 output retains both the upper hook and middle stroke
of દૂ. At native scan coordinates, the upper-stroke region
`[355, 1186, 370, 1194]` goes from 0 to 43 pixels below grayscale 160, and the
middle region `[355, 1196, 369, 1203]` goes from 1 to 34. The regression test
now checks these two specific parts, rather than relying only on the whole
weak-word region's dark pixel count.

Validation: 51 tests passed, six absent-fixture tests skipped; wheel and source
builds passed. A synthetic test verifies that connected letter fragments are
recovered while an equally faint isolated speck remains deleted. Default
outputs and their baselines remain unchanged.
