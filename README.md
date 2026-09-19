# scanclean

Print-ready cleanup for scanned books, tuned to preserve Indic-script diacritics.
Install: `python -m pip install -r requirements.txt` (Poppler is also required).
Run: `python scanclean.py original/*.pdf -o cleaned`
Audit: `python scanclean.py original/*.pdf -o cleaned --audit`
Print mode: add `--upscale 2 --bilevel`.
See [experiments tried and implementation details](docs/experiments_tried.md).
