#!/usr/bin/env python3
"""Per-line loss audit: catch text deleted INSIDE a line of type.

    python3 audit_lines.py [n_worst]

The `core_removed` counter in ScanClean has a blind spot, and it is the one
that matters. It asks "did we delete anything glyph-shaped?" - but when a crease
grazes a word, letter and crease become ONE component, and that merged blob is
not glyph-shaped. Deleting it wholesale takes the word with it and the counter
stays at zero. That is exactly how a page lost the last letters of two lines
while every metric read clean.

So audit by LINE instead of by component: find the lines of type, then measure
how much ink each one lost. Text lives in lines; anything that removes a chunk
from inside one is suspect regardless of what shape it had when deleted.

Writes audit_lines.png - the worst offenders, deleted pixels in red - and prints
a ranked table. Zero rows means no line lost a meaningful amount of ink.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scanclean.core as sc
from scanclean.cli import page_images


class Opt:
    keep_colour = False
    colour_kill = 1.4
    chroma = 42
    max_stamp = 0.04
    ink_floor = 130
    band = 0.09
    up, down, side = 0.85, 0.65, 0.55
    speck, faint = 0.55, 150
    deskew = True
    detect = True
    no_crop = False


def kill_mask(bgr, dpi, o=Opt):
    """(norm, kill, runs, gh) from the pipeline's own analysis - not a copy of
    it, so the audit cannot drift from the code that ships."""
    a = sc.analyse(bgr, dpi, o)
    if a["gh"] is None:
        z = np.zeros_like(a["norm"])
        return a["norm"], z, z, 1.0
    return a["norm"], a["kill"], a["runs"], a["gh"]


def line_boxes(runs, gh):
    """Bounding box of each line of type."""
    smear = cv2.dilate(runs, np.ones((1, max(3, int(2.0 * gh) | 1)), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(smear, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i, :5]
        if w >= 3 * gh and h >= 0.4 * gh:
            out.append((x, y, w, h))
    return out


def main():
    worst = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    rows, tiles = [], []
    for pdf in sorted(f for f in os.listdir("original") if f.endswith(".pdf")):
        for i, bgr, dpi in page_images(f"original/{pdf}"):
            norm, kill, runs, gh = kill_mask(bgr, dpi)
            for (x, y, w, h) in line_boxes(runs, gh):
                pad = int(0.8 * gh)
                y0, y1 = max(0, y - pad), min(kill.shape[0], y + h + pad)
                x0, x1 = max(0, x - pad), min(kill.shape[1], x + w + pad)
                # only ink that was PART OF the line: junk deleted from the
                # margin inside a line's box is the tool working, not a loss
                lost = int(((kill[y0:y1, x0:x1] > 0) & (runs[y0:y1, x0:x1] > 0)).sum())
                if lost < 120:
                    continue
                rows.append((lost, pdf, i + 1, (x0, y0, x1, y1)))
    rows.sort(reverse=True)

    print(f"{len(rows)} lines lost >=120 px of ink")
    for lost, pdf, pg, _ in rows[:worst]:
        print(f"   {lost:6d} px   {pdf} p{pg}")

    if rows:
        cache = {}
        for lost, pdf, pg, (x0, y0, x1, y1) in rows[:worst]:
            key = (pdf, pg)
            if key not in cache:
                for i, bgr, dpi in page_images(f"original/{pdf}"):
                    if i + 1 == pg:
                        cache[key] = kill_mask(bgr, dpi)[:2]
                        break
            norm, kill = cache[key]
            vis = cv2.cvtColor(norm[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
            vis[kill[y0:y1, x0:x1] > 0] = (0, 0, 255)
            scale = min(900 / max(vis.shape[1], 1), 3.0)
            vis = cv2.resize(vis, None, fx=scale, fy=scale)
            cv2.putText(vis, f"{pdf} p{pg}  {lost}px", (4, vis.shape[0] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
            tiles.append(vis)
        W = max(t.shape[1] for t in tiles)
        pad = [cv2.copyMakeBorder(t, 0, 6, 0, W - t.shape[1],
                                  cv2.BORDER_CONSTANT, value=(210, 210, 210))
               for t in tiles]
        cv2.imwrite("audit_lines.png", np.vstack(pad))
        print("-> audit_lines.png")


if __name__ == "__main__":
    main()
