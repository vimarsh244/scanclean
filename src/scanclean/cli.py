"""Desktop PDF command-line interface.

Poppler and filesystem handling live here so importing :mod:`scanclean` keeps
the NumPy/OpenCV cleaning core independent of desktop tools.
"""

import argparse
import os
import subprocess
import tempfile

import cv2
from PIL import Image

from .core import clean_page, write_pdf


def page_images(pdf_path, dpi=None):
    """Yield ``(index, BGR array, dpi)`` for each page of a PDF."""
    tmp = tempfile.mkdtemp(prefix="scanclean_")
    listing = subprocess.run(
        ["pdfimages", "-list", pdf_path], capture_output=True, text=True
    ).stdout.splitlines()
    rows = [line.split() for line in listing[2:] if line.strip()]
    npages = int(
        subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True)
        .stdout.split("Pages:")[1]
        .split()[0]
    )
    one_per_page = len(rows) == npages and len({row[0] for row in rows}) == npages

    if one_per_page and dpi is None:
        subprocess.run(["pdfimages", "-all", pdf_path, f"{tmp}/p"], check=True)
        files = sorted(name for name in os.listdir(tmp) if name.startswith("p-"))
        images = [cv2.imread(os.path.join(tmp, name), cv2.IMREAD_COLOR) for name in files]
        if all(image is not None for image in images):
            for index, image in enumerate(images):
                yield index, image, float(rows[index][12])
            return
        for name in files:
            os.remove(os.path.join(tmp, name))
        subprocess.run(["pdfimages", "-png", pdf_path, f"{tmp}/p"], check=True)
        for index, name in enumerate(
            sorted(item for item in os.listdir(tmp) if item.endswith(".png"))
        ):
            yield index, cv2.imread(os.path.join(tmp, name), cv2.IMREAD_COLOR), float(
                rows[index][12]
            )
    else:
        resolution = dpi or 400
        subprocess.run(
            ["pdftoppm", "-r", str(resolution), "-png", pdf_path, f"{tmp}/p"],
            check=True,
        )
        for index, name in enumerate(
            sorted(item for item in os.listdir(tmp) if item.endswith(".png"))
        ):
            yield index, cv2.imread(os.path.join(tmp, name), cv2.IMREAD_COLOR), float(
                resolution
            )


def build_parser():
    parser = argparse.ArgumentParser(description="Clean scanned book PDFs for printing.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("-o", "--outdir", default="cleaned")
    parser.add_argument("--dpi", type=float, default=None, help="force rasterise at this dpi")
    parser.add_argument("--bilevel", action="store_true", help="1-bit output (much smaller)")
    parser.add_argument(
        "--no-deskew",
        dest="deskew",
        action="store_false",
        help="do not level a tilted scan (on by default, up to 3 degrees)",
    )
    parser.add_argument(
        "--no-detect",
        dest="detect",
        action="store_false",
        help="do not consult the bundled PP-OCRv6 text detector",
    )
    parser.add_argument(
        "--no-paper-floor",
        dest="paper_floor",
        action="store_false",
        help="keep sub-ink grey (mottle, edge wear) instead of whitening it",
    )
    parser.add_argument("--no-crop", action="store_true", help="keep full page, do not clip margins")
    parser.add_argument(
        "--keep-colour", action="store_true", help="do not suppress coloured ink/stamps"
    )
    parser.add_argument(
        "--band", type=float, default=0.09, help="edge band width (fraction of page)"
    )
    parser.add_argument(
        "--ink-floor", type=float, default=130, help="stamp removal dark-pixel floor"
    )
    parser.add_argument(
        "--max-stamp", type=float, default=0.04, help="maximum coloured-ink page fraction"
    )
    parser.add_argument("--chroma", type=float, default=42, help="coloured-ink chroma threshold")
    parser.add_argument(
        "--colour-kill", type=float, default=1.4, help="strength of chroma suppression"
    )
    parser.add_argument("--black", type=float, default=40, help="tone: full-black point")
    parser.add_argument("--white", type=float, default=224, help="tone: paper-white point")
    parser.add_argument(
        "--speck", type=float, default=0.55, help="maximum speck area / median glyph area"
    )
    parser.add_argument("--up", type=float, default=0.85, help="protection above glyphs")
    parser.add_argument("--down", type=float, default=0.65)
    parser.add_argument("--side", type=float, default=0.55)
    parser.add_argument("--soften", type=float, default=1.0)
    parser.add_argument("--sharpen", type=float, default=0.8, help="unsharp amount")
    parser.add_argument("--upscale", type=int, default=1, help="integer output scale factor")
    parser.add_argument("--faint", type=float, default=150, help="faint dirt threshold")
    parser.add_argument("--transparent", action="store_true", help="emit ink on transparent paper")
    parser.add_argument("--matte", action="store_true", help="paint white behind transparent ink")
    parser.add_argument(
        "--audit", action="store_true", help="also write an overlay PDF of removed pixels"
    )
    return parser


def main(argv=None):
    opt = build_parser().parse_args(argv)
    os.makedirs(opt.outdir, exist_ok=True)
    for path in opt.inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        pages, dpis, audits, audit_dpis = [], [], [], []
        for index, bgr, dpi in page_images(path, opt.dpi):
            out, stats, audit, out_dpi = clean_page(bgr, dpi, opt)
            pages.append(out)
            dpis.append(out_dpi)
            if audit is not None:
                audits.append(cv2.cvtColor(audit, cv2.COLOR_BGR2RGB))
                audit_dpis.append(dpi)
            print(f"  {name} p{index + 1}: {stats}", flush=True)
        destination = os.path.join(opt.outdir, f"{name}.cleaned.pdf")
        write_pdf(
            pages,
            dpis,
            destination,
            opt.bilevel,
            transparent=opt.transparent,
            matte=opt.matte,
        )
        print(f"-> {destination}  ({os.path.getsize(destination) / 1e6:.2f} MB)")
        if audits:
            Image.init()
            audit_path = os.path.join(opt.outdir, f"{name}.audit.pdf")
            images = [Image.fromarray(array) for array in audits]
            images[0].save(
                audit_path,
                save_all=True,
                append_images=images[1:],
                resolution=float(audit_dpis[0]),
            )
            print(f"-> {audit_path}  (red = deleted pixels)")
