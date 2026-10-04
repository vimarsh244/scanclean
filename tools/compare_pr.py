#!/usr/bin/env python3
"""Compare local scans against a Git revision without switching branches.

    python tools/compare_pr.py --base-ref main

Source PDFs and every generated artifact stay in ignored original/ and work/.
The detector counts glyph-shaped deletions, including ink lost to cropping;
these counts are diagnostic proxies, not OCR accuracy or proof of legibility.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2

from scanclean import core
from scanclean.cli import page_images

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_glyphs  # noqa: E402


def load_revision(ref):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:src/scanclean/core.py"], cwd=ROOT, text=True
    )
    name = "scanclean._comparison_base"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = core.__file__
    sys.modules[name] = module
    exec(compile(source, f"{ref}:src/scanclean/core.py", "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--samples", nargs="+", type=int, default=[5, 6, 7])
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "pr8-review")
    parser.add_argument("--white", type=float, default=224)
    parser.add_argument("--restore-ink", type=float, default=0.0,
                        help="ink restoration strength for current code only (0-1)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = load_revision(args.base_ref)
    rows = []
    for sample in args.samples:
        outputs = {"base": [], "current": []}
        dpis = []
        for index, image, dpi in page_images(str(ROOT / "original" / f"Sample {sample}.pdf")):
            stem = f"s{sample}p{index + 1}"
            cv2.imwrite(str(args.out / f"{stem}-original.png"), image)
            row = {"sample": sample, "page": index + 1, "stem": stem}
            for tag, module in (("base", base), ("current", core)):
                opt = module.Options(white=args.white)
                if tag == "current":
                    opt.restore_ink = args.restore_ink
                output, stats, _, out_dpi = module.clean_page(image, dpi, opt)
                analysis = module.analyse(image, dpi, opt)
                marks, _ = audit_glyphs.erased(image, dpi, opt, analysis=analysis)
                row[tag] = {"erased": len(marks), "by_stage": dict(Counter(m[0] for m in marks)),
                            "stats": stats}
                cv2.imwrite(str(args.out / f"{stem}-{tag}.png"), output)
                outputs[tag].append(output)
            dpis.append(out_dpi)
            rows.append(row)
            print(f"Sample {sample} p{index + 1}: {row['base']['erased']} -> "
                  f"{row['current']['erased']} suspect erased marks", flush=True)
        for tag, pages in outputs.items():
            core.write_pdf(pages, dpis, str(args.out / f"Sample {sample}.{tag}.pdf"))

    metadata = {"base_ref": args.base_ref, "base_commit": subprocess.check_output(
        ["git", "rev-parse", args.base_ref], cwd=ROOT, text=True).strip(),
        "current_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "current_core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "working_tree_modified": bool(subprocess.check_output(
        ["git", "diff", "HEAD", "--", "src/scanclean/core.py"], cwd=ROOT)),
        "opencv": cv2.__version__, "white": args.white,
        "restore_ink": args.restore_ink, "pages": rows}
    (args.out / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    html = '''<!doctype html><meta charset="utf-8"><title>ScanClean comparison</title>
<style>body{font:16px system-ui;background:#ddd;margin:16px}header{position:sticky;top:0;background:#ddd;padding:12px;z-index:1}main{display:flex;gap:12px;align-items:flex-start}section{flex:1;min-width:0}img{width:100%;display:block}h2{font-size:18px}select{font:inherit}</style>
<header><label>Page <select id="page"></select></label> <label>Display width <input id="zoom" type="range" min="300" max="1800" value="500"></label><p id="counts"></p><div id="pdfs"></div></header>
<main><section><h2>Original</h2><img id="original"></section><section><h2>Reference version</h2><img id="base"></section><section><h2 id="current-label">Current branch</h2><img id="current"></section></main>
<script>const pages=DATA;
const restoration=RESTORATION;
if(restoration>0)document.getElementById('current-label').textContent=`Ink restoration (strength ${restoration})`;
const select=document.getElementById('page');
for(const [i,p] of pages.entries()){const o=document.createElement('option');o.value=i;o.textContent=`Sample ${p.sample}, page ${p.page}`;select.append(o)}
function show(){const p=pages[select.value];for(const tag of ['original','base','current'])document.getElementById(tag).src=`${p.stem}-${tag}.png`;document.getElementById('counts').textContent=`Suspect erased marks (including crop): ${p.base.erased} → ${p.current.erased}. This is a diagnostic proxy; inspect faint text visually.`;const links=document.getElementById('pdfs');links.replaceChildren();for(const tag of ['base','current']){const a=document.createElement('a');a.href=`Sample ${p.sample}.${tag}.pdf`;a.textContent=`Download ${tag} PDF`;a.style.marginRight='16px';links.append(a)}}
function zoom(){for(const s of document.querySelectorAll('section')){s.style.flex='none';s.style.width=document.getElementById('zoom').value+'px'}}
select.onchange=show;document.getElementById('zoom').oninput=zoom;show();zoom();</script>'''
    (args.out / "index.html").write_text(
        html.replace("DATA", json.dumps(rows)).replace("RESTORATION", str(args.restore_ink)))
    for sample in args.samples:
        subset = [r for r in rows if r["sample"] == sample]
        print(f"Sample {sample}: " + " -> ".join(str(sum(r[t]["erased"] for r in subset))
                                                for t in ("base", "current")))
    print(f"Open {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
