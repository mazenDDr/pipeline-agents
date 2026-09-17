"""Download the raw benchmark datasets into data/raw/<name>/ (run on the GPU machine).

All are from the UCI Machine Learning Repository under CC BY 4.0. The sha256 of each archive is
recorded in data/raw/manifest.json on first download and checked on later runs, so a changed upstream
file cannot silently change the benchmark.
"""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "bank_marketing": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "bike_sharing": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "wine_quality": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    "credit_default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "diabetes_readmission": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "air_quality": "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
    "online_retail": "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
    "student_performance": "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
}


def unzip_all(path: Path, dest: Path) -> None:
    """Extract, including zips nested inside the archive (UCI often ships a zip of zips)."""
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)
    for inner in dest.rglob("*.zip"):
        if inner != path:
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(inner.parent / inner.stem)
            inner.unlink()


def main() -> None:
    root = Path("data/raw")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in DATASETS.items():
        dest = root / name
        archive = root / f"{name}.zip"
        if not archive.exists():
            print(f"get {name}", flush=True)
            urllib.request.urlretrieve(url, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if name in manifest and manifest[name]["sha256"] != digest:
            raise SystemExit(f"{name}: archive changed upstream ({digest} != {manifest[name]['sha256']})")
        manifest[name] = {
            "url": url,
            "sha256": digest,
            "bytes": archive.stat().st_size,
            "license": "CC BY 4.0",
        }
        if not dest.exists():
            dest.mkdir()
            unzip_all(archive, dest)
        print(f"ok  {name} {archive.stat().st_size / 1e6:.1f} MB", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
