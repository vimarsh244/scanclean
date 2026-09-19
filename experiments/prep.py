#!/usr/bin/env python3
"""Shared inputs for every experiment: one PNG per page, same geometry.

    in/sNpMM.png   original scan, straightened (colour)
    norm/sNpMM.png flattened greyscale (scanclean's own background removal)
    sc/sNpMM.png   scanclean output (baseline to beat)
    kill/sNpMM.png scanclean's deletion mask
    runs/sNpMM.png scanclean's text-line membership mask
"""
import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scanclean.core as sc
from scanclean.cli import page_images
from audit_lines import Opt

class O(Opt):
    black, white, soften, sharpen, upscale, bilevel = 40, 224, 1.0, 0.8, 1, False
    paper_floor, audit = True, False

for d in ("in", "norm", "sc", "kill", "runs"):
    os.makedirs(f"{HERE}/{d}", exist_ok=True)
for s in (1, 2, 3, 4):
    for i, bgr, dpi in page_images(f"{os.path.dirname(HERE)}/original/Sample {s}.pdf"):
        tag = f"s{s}p{i+1:02d}"
        straight, _ = sc.straighten(bgr, dpi / 300.0)
        a = sc.analyse(bgr, dpi, O)
        out, st, _, _ = sc.clean_page(bgr, dpi, O)
        cv2.imwrite(f"{HERE}/in/{tag}.png", straight)
        cv2.imwrite(f"{HERE}/norm/{tag}.png", a["norm"])
        cv2.imwrite(f"{HERE}/sc/{tag}.png", out)
        if a["gh"] is not None:
            cv2.imwrite(f"{HERE}/kill/{tag}.png", a["kill"])
            cv2.imwrite(f"{HERE}/runs/{tag}.png", a["runs"])
        print(tag, straight.shape, dpi, flush=True)
