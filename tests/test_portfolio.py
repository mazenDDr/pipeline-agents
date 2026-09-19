"""The public portfolio stays reproducible, linked and tied to committed evidence."""

import importlib.util
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.feed(path.read_text())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(values[key])


def test_public_links_and_anchors() -> None:
    for path in (ROOT / "site").rglob("*.html"):
        for link in Page(path).links:
            url = urlsplit(link)
            if url.netloc == "github.com" and url.path.startswith("/mazenDDr/pipeline-agents/"):
                relative = re.sub(r"^/mazenDDr/pipeline-agents/(blob|tree)/main/", "", url.path)
                if relative != url.path:
                    assert (ROOT / unquote(relative)).exists(), (path, link)
            if url.scheme or url.netloc:
                continue
            target = path.parent / unquote(url.path) if url.path else path
            if target.is_dir():
                target /= "index.html"
            assert target.exists(), (path, link)
            if url.fragment and target.suffix == ".html":
                assert unquote(url.fragment) in Page(target).ids, (path, link)


def test_figures_are_accessible_self_contained_and_identical() -> None:
    for asset in (ROOT / "docs/assets").glob("*.svg"):
        svg = ElementTree.fromstring(asset.read_text())
        assert svg.attrib["viewBox"].startswith("0 0 880 ")
        assert svg.find("{http://www.w3.org/2000/svg}title").text
        assert "prefers-reduced-motion" in asset.read_text()
        assert asset.read_bytes() == (ROOT / "site/assets" / asset.name).read_bytes()


def test_portfolio_rebuild_is_byte_identical() -> None:
    pytest.importorskip("markdown", reason="install the docs extra to test the portfolio builder")
    paths = [ROOT / "README.md", *(ROOT / "docs/assets").glob("*.svg")]
    paths += list((ROOT / "site").rglob("*.html")) + list((ROOT / "site/assets").glob("*.svg"))
    before = {path: path.read_bytes() for path in paths}
    spec = importlib.util.spec_from_file_location("build_portfolio", ROOT / "scripts/build_portfolio.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    assert {path: path.read_bytes() for path in paths} == before
