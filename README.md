# scanclean

Print-ready cleanup for scanned books, tuned to preserve Indic-script diacritics.
Install: `python -m pip install -r requirements.txt` (Poppler is also required).

- Clean: `python scanclean.py original/*.pdf -o cleaned`
- Audit: `python scanclean.py original/*.pdf -o cleaned --audit`
- Print: `python scanclean.py original/*.pdf -o cleaned --upscale 2 --bilevel`

See [experiments tried and implementation details](docs/experiments_tried.md).
