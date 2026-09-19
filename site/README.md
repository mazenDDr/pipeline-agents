# The public portfolio

Three static pages: the field guide (`index.html`), fresh-data field test (`field-test/`), and recorded
credit-default tour (`tour/`). No API requests, model calls, datasets or private checkpoints are needed.

Edit `templates/` and `assets/style.css` / `assets/site.js`, then rebuild from the repository root:

```bash
pip install -e '.[dev,docs]'
python scripts/build_portfolio.py
make check
python -m http.server 8765 --bind 127.0.0.1 --directory site
# Separate terminal; requires installed Google Chrome:
python scripts/check_portfolio.py
```

The builder also creates the root README and six self-contained SVGs in `docs/assets/`. It copies those
figures into this site's assets folder. Tables and charts read the committed T11 summaries, trust scores,
field summary and reference answers. The tour and full field report reuse the committed research documents.
`tests/test_portfolio.py` checks links, anchors, SVG metadata, identical copies and deterministic rebuilding.

The GitHub Pages workflow publishes only the three generated pages and their assets, not these templates.
Repository Pages must be configured to use GitHub Actions before deployment. The workflow runs after a
relevant change merges to `main`, or manually. Until Pages is enabled and that workflow succeeds, the public
URLs in the README are not live.
