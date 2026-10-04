#!/usr/bin/env python3
"""Characters erased, counted by a witness with no stake in the answer.

    python3 audit_glyphs.py [n_worst]

`core_removed` and audit_lines.py both measure ScanClean against ScanClean's
own idea of where the text is. That is exactly the wrong instrument for the
failure that matters most, because the stages that delete text wholesale -
the margin band, the crease finder - delete it *precisely when* they have
decided it is not text. Every counter then reads clean while the page comes
back missing the first word of every line.

So ask something else. The bundled PP-OCRv6 detector is trained on print,
knows nothing of this pipeline's reasoning, and says where lines of type are.
A deleted mark that is shaped and sized like a letter of this page's type AND
sits inside one of its boxes is, to a good approximation, an erased character.
Each is attributed to the stage that took it, so a regression names its own
cause.

Writes audit_glyphs.png - the worst offenders, deleted pixels in red - and
prints a table by page and stage.
"""

import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scanclean.core as sc
from scanclean import Options
from scanclean.cli import page_images


def erased(bgr, dpi, opt, *, analysis=None):
    """[(stage, x, y, w, h)] for every deleted mark the detector calls type."""
    if analysis is None:
        analysis = sc.analyse(bgr, dpi, opt)
    if analysis["gh"] is None:
        return [], analysis
    # analyse() deskews internally; the witness must use that same coordinate
    # system or its boxes can miss the deleted glyphs on a tilted scan.
    aligned = sc.straighten(bgr, dpi / 300.0)[0] if getattr(opt, "deskew", True) else bgr
    detected = sc.detect_text(aligned)
    if detected is None:
        return [], analysis
    boxed = detected[0]
    gh, ga = analysis["gh"], analysis["ga"]

    # Composition also deletes surviving ink outside its crop. Those pixels
    # never enter `kill`, so an audit of removal stages alone misses page
    # numbers that survive cleaning and disappear only in the final output.
    crop = analysis["mask"].copy()
    x0, y0, x1, y1 = analysis["box"]
    crop[y0:y1, x0:x1] = 0
    stages = dict(analysis["stages"], crop=crop)
    found = []
    for stage, removed in stages.items():
        if stage == "rescued" or not removed.any():
            continue
        count, labels, stats, centres = cv2.connectedComponentsWithStats(removed, 8)
        for index in range(1, count):
            x, y, w, h, area = stats[index, :5]
            if not (0.45 * gh <= h <= 2.2 * gh and 0.2 * gh <= w <= 3 * gh):
                continue
            if area < 0.15 * ga:
                continue
            cx, cy = int(centres[index][0]), int(centres[index][1])
            if not boxed[cy, cx]:
                continue
            found.append((stage, x, y, w, h))
    return found, analysis


def main():
    worst = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    opt = Options()
    rows, tiles, total = [], [], Counter()

    for pdf in sorted(name for name in os.listdir("original") if name.endswith(".pdf")):
        for index, bgr, dpi in page_images(f"original/{pdf}"):
            marks, analysis = erased(bgr, dpi, opt)
            by_stage = Counter(stage for stage, *_ in marks)
            total.update(by_stage)
            if marks:
                rows.append((len(marks), pdf, index + 1, dict(by_stage)))
                norm, kill = analysis["norm"], analysis["kill"].copy()
                x0, y0, x1, y1 = analysis["box"]
                outside = analysis["mask"].copy()
                outside[y0:y1, x0:x1] = 0
                kill = cv2.bitwise_or(kill, outside)
                for stage, x, y, w, h in marks[:6]:
                    pad = int(1.2 * analysis["gh"])
                    y0, y1 = max(0, y - pad), min(norm.shape[0], y + h + pad)
                    x0, x1 = max(0, x - pad), min(norm.shape[1], x + w + pad)
                    tile = cv2.cvtColor(norm[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
                    tile[kill[y0:y1, x0:x1] > 0] = (0, 0, 255)
                    tile = cv2.resize(tile, None, fx=3, fy=3,
                                      interpolation=cv2.INTER_NEAREST)
                    cv2.putText(tile, f"{pdf} p{index + 1} {stage}", (3, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 0, 0), 1)
                    tiles.append(tile)
            print(f"  {pdf} p{index + 1}: {len(marks)} {dict(by_stage)}", flush=True)

    rows.sort(reverse=True)
    print(f"\n{sum(total.values())} glyph-shaped marks erased inside detected text")
    for stage, count in total.most_common():
        print(f"   {count:5d}  {stage}")
    print("\nworst pages:")
    for count, pdf, page, by_stage in rows[:worst]:
        print(f"   {count:4d}  {pdf} p{page}  {by_stage}")

    if tiles:
        tiles = tiles[:worst * 6]
        width = max(tile.shape[1] for tile in tiles)
        padded = [cv2.copyMakeBorder(tile, 0, 6, 0, width - tile.shape[1],
                                     cv2.BORDER_CONSTANT, value=(210, 210, 210))
                  for tile in tiles]
        cv2.imwrite("audit_glyphs.png", np.vstack(padded))
        print("-> audit_glyphs.png")


if __name__ == "__main__":
    main()
