"""Download the candidate GGUF files into models/gguf/ (run on the GPU machine).

Smallest first, so the cheap tier is usable early; files already present are skipped. A second
copy with --largest-first roughly doubles throughput; huggingface_hub's file locks keep them apart.
"""

import argparse
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/candidates.yaml")
    parser.add_argument("--out", default="models/gguf")
    parser.add_argument("--largest-first", action="store_true", help="for a second, parallel downloader")
    args = parser.parse_args()

    candidates = [c for tier in yaml.safe_load(Path(args.config).read_text()).values() for c in tier]
    api = HfApi()
    sized = []
    for c in candidates:
        info = api.get_paths_info(c["repo"], [c["file"]])[0]
        sized.append((info.lfs.size if info.lfs else info.size, c))

    for size, c in sorted(sized, key=lambda s: s[0], reverse=args.largest_first):
        target = Path(args.out) / c["file"]
        if target.exists() and target.stat().st_size == size:
            print(f"skip {c['name']} (present)", flush=True)
            continue
        print(f"get  {c['name']} {size / 2**30:.2f} GiB", flush=True)
        hf_hub_download(c["repo"], c["file"], local_dir=args.out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
