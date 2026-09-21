#!/usr/bin/env python3
"""Rebuild compare/ : every page of every PDF, before and after, side by side.

    python3 build_compare.py

Images are written at the scans' NATIVE pixel size. Nothing is resampled, so
what the slider shows is what the pipeline produced; the page then displays them
at 560 CSS px, which is ~1:1 against device pixels on a Retina screen.

The page data is inlined into the HTML rather than fetched, so compare/index.html
opens straight from a file:// path with no local server.
"""

import json
import os
import shutil
import subprocess
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "compare")
IMG = os.path.join(OUT, "img")
SAMPLES = [1, 2, 3, 4, 5, 6, 7]
MARK = "pages = /*DATA*/"


def extract(pdf, tmp):
    subprocess.run(["pdfimages", "-png", pdf, f"{tmp}/p"], check=True)
    return sorted(f for f in os.listdir(tmp) if f.endswith(".png"))


def main():
    if not os.path.isfile(os.path.join(OUT, "index.html")):
        raise SystemExit("compare/index.html is missing - it is the template.")
    shutil.rmtree(IMG, ignore_errors=True)
    os.makedirs(IMG)

    pages = []
    for i in SAMPLES:
        src = {"before": os.path.join(HERE, "original", f"Sample {i}.pdf"),
               "after": os.path.join(HERE, "cleaned", f"Sample {i}.cleaned.pdf")}
        for tag in ("before", "after"):
            tmp = tempfile.mkdtemp()
            try:
                for pi, f in enumerate(extract(src[tag], tmp), start=1):
                    im = Image.open(os.path.join(tmp, f))
                    if tag == "before":
                        # a photograph of paper: JPEG is the right fit
                        im = im.convert("RGB")
                        name = f"s{i}p{pi:02d}b.jpg"
                        im.save(os.path.join(IMG, name), "JPEG",
                                quality=88, optimize=True)
                        pages.append({"s": i, "p": pi, "w": im.width, "h": im.height,
                                      "b": name, "a": f"s{i}p{pi:02d}a.png"})
                    else:
                        # near-white with hard edges: PNG, and lossless, so the
                        # comparison cannot be accused of flattering the output
                        im = im.convert("L")
                        im.save(os.path.join(IMG, f"s{i}p{pi:02d}a.png"),
                                "PNG", optimize=True)
            finally:
                shutil.rmtree(tmp)

    pages.sort(key=lambda d: (d["s"], d["p"]))
    index_path = os.path.join(OUT, "index.html")
    with open(index_path, encoding="utf-8") as stream:
        html = stream.read()
    # Anchor on the data block, not on "pages = [" - that also matches the
    # declaration `let pages = [], sample = 0` further up, and replacing from
    # there to the next "];" would swallow half the script.
    start = html.index(MARK)
    end = html.index("];", start) + 2
    html = (html[:start] + MARK
            + json.dumps(pages, separators=(",", ":")) + ";" + html[end:])
    with open(index_path, "w", encoding="utf-8", newline="") as stream:
        stream.write(html)
    with open(os.path.join(OUT, "pages.json"), "w", encoding="utf-8") as stream:
        json.dump(pages, stream)

    mb = sum(os.path.getsize(os.path.join(IMG, f)) for f in os.listdir(IMG)) / 1e6
    print(f"{len(pages)} page pairs, {len(os.listdir(IMG))} images, {mb:.1f} MB")
    print(f"open {os.path.join(OUT, 'index.html')}")


if __name__ == "__main__":
    main()
