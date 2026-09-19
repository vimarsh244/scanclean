# experiments

Research scripts behind the choices in `../README.md`. All paths are relative to
the repo; inputs and outputs (`in/ norm/ sc/ kill/ runs/ wb/ out/`) are page
renders and are git-ignored.

```bash
python3 experiments/prep.py                         # per-page inputs + scanclean baseline
uv venv -p 3.12 .venv && uv pip install -p .venv -r experiments/requirements.txt
python3 third_party/docdiff/fetch_weights.py        # DocDiff weights (64 MB)
```

| Script | What it tests |
| --- | --- |
| `run_ppocr.py` | PaddleOCR text detectors (tiny / server), boxes per page |
| `textdet.py` | the same detector through `cv2.dnn`, parity-checked against PaddleOCR |
| `gate.py`, `lostsheet.py`, `stagebox.py`, `rescue_probe.py` | what a detector mask would add / take back on top of scanclean |
| `verify_stage.py <stage>` | before/after contact sheet of everything one stage removed or rescued |
| `run_docdiff.py`, `train_docdiff.py`, `eval_docdiff.py` | DocDiff as published, fine-tuned on these books, and scored |
| `run_zigzag.py` | ZigZag background normalisation |
| `run_sauvolanet.py` | SauvolaNet binarisation (Keras weights re-implemented in torch) |
