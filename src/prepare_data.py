"""Download a small WikiArt subset and COCO val2017 photographs.

No dataset is included in the delivered archive. Network access is needed.
WikiArt is sampled from a bounded Dataset Viewer scan, NOT the full corpus.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import canny_edges, edge_rgb, read_json, sha256, square_image, verify_files, write_json

API = "https://datasets-server.huggingface.co"
WIKIART = "huggan/wikiart"
COCO_ARCHIVE = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])))
    s.headers["User-Agent"] = "educational-style-transfer-stage1/1.0"
    return s


def get_json(s, url: str, **kwargs):
    r = s.get(url, timeout=90, **kwargs)
    r.raise_for_status()
    return r.json()


def hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def download(s, url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size:
        return
    tmp = target.with_suffix(target.suffix + ".part")
    with s.get(url, stream=True, timeout=(30, 180)) as r:
        r.raise_for_status()
        with tmp.open("wb") as out:
            for chunk in r.iter_content(1024 * 1024):
                out.write(chunk)
    tmp.replace(target)


def image_url(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "src" in value:
        return value["src"]
    if isinstance(value, list) and value:
        return image_url(value[0])
    raise ValueError(f"Unsupported Dataset Viewer image field: {type(value)}")


def dhash(image: Image.Image) -> int:
    a = np.asarray(image.convert("L").resize((9, 8)))
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    return sum(int(b) << i for i, b in enumerate(bits))


def prepare_wikiart(s, train_count=40, val_count=5, seed=2026, max_scan=5000) -> None:
    manifest_path = Path("data/style/manifest.json")
    if manifest_path.exists():
        old = read_json(manifest_path)
        expected = {"train_count": train_count, "val_count": val_count, "seed": seed}
        if any(old["selection"][k] != v for k, v in expected.items()):
            raise RuntimeError("Existing WikiArt selection differs. Use a separate project directory.")
        verify_files(old["images"])
        if sha256(old["reference"]["path"]) != old["reference"]["sha256"]:
            raise RuntimeError("Style reference was modified")
        print("WikiArt: existing files verified")
        return
    info = get_json(s, f"https://huggingface.co/datasets/{WIKIART}/raw/main/dataset_infos.json",
                    headers=hf_headers())
    features = next(iter(info.values()))["features"]
    style_id = features["style"]["names"].index("Impressionism")
    allowed_genres = {"landscape", "cityscape", "still_life"}
    candidates, scan_count = [], 0
    # Viewer may expose only part of the corpus. Record this limitation explicitly.
    last_response = {}
    target_pool = 2 * (train_count + val_count)
    for offset in range(0, max_scan, 100):
        j = get_json(s, f"{API}/rows", headers=hf_headers(), params={
            "dataset": WIKIART, "config": "default", "split": "train",
            "offset": offset, "length": 100})
        last_response = j
        rows = j.get("rows", [])
        if not rows:
            break
        scan_count += len(rows)
        for item in rows:
            r = item["row"]
            style = r["style"]
            is_style = style == style_id or style == "Impressionism"
            genre = r["genre"]
            if isinstance(genre, int):
                genre = features["genre"]["names"][genre]
            if is_style and genre in allowed_genres:
                artist = r["artist"]
                if isinstance(artist, int):
                    artist = features["artist"]["names"][artist]
                candidates.append({"row_idx": item["row_idx"], "genre": genre,
                                   "artist": artist, "url": image_url(r["image"])})
        if len(candidates) >= target_pool:
            break
    random.Random(seed).shuffle(candidates)
    chosen, hashes = [], []
    genres = {"landscape": "a landscape", "cityscape": "a cityscape",
              "still_life": "a still life"}
    for candidate in candidates:
        if len(chosen) >= train_count + val_count:
            break
        # Signed viewer URLs are fetched immediately, not treated as permanent URLs.
        r = s.get(candidate["url"], timeout=90)
        r.raise_for_status()
        with Image.open(io.BytesIO(r.content)) as raw:
            if min(raw.size) < 256:
                continue
            im = square_image(raw, 512)
        h = dhash(im)
        if any((h ^ old).bit_count() <= 4 for old in hashes):
            continue
        hashes.append(h)
        split = "train" if len(chosen) < train_count else "val"
        path = Path(f"data/style/{split}/wikiart_{candidate['row_idx']:06d}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path)
        chosen.append({"id": f"wikiart_{candidate['row_idx']:06d}",
                       "row_idx": candidate["row_idx"], "path": str(path), "split": split,
                       "sha256": sha256(path), "style": "Impressionism",
                       "artist": candidate["artist"], "genre": candidate["genre"],
                       "text": "an impressionist painting of " + genres[candidate["genre"]],
                       "source": WIKIART})
    if len(chosen) != train_count + val_count:
        raise RuntimeError(f"Found only {len(chosen)} usable images; increase --max-scan. "
                           "If Dataset Viewer is unavailable, follow README's manual-data protocol.")
    train = [r for r in chosen if r["split"] == "train"]
    with Path("data/style/train/metadata.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps({"file_name": Path(r["path"]).name, "text": r["text"]}) + "\n")
    reference = Path("data/style/reference.png")
    shutil.copyfile(train[0]["path"], reference)
    write_json(manifest_path, {
        "dataset": WIKIART, "style": "Impressionism", "images": chosen,
        "reference": {"path": str(reference), "source_id": train[0]["id"], "sha256": sha256(reference)},
        "selection": {"train_count": train_count, "val_count": val_count, "seed": seed,
                      "rows_scanned": scan_count, "candidate_pool": len(candidates),
                      "genres": sorted(allowed_genres), "near_duplicate_dhash_max_distance": 4},
        "viewer_metadata": {k: last_response.get(k) for k in
                            ["num_rows_total", "partial", "dataset_git_revision"]},
        "artist_counts": dict(Counter(r["artist"] for r in chosen)),
        "sampling_limitation": "Bounded viewer subset; not a representative sample of all WikiArt.",
        "caption_note": "Genre-based starter captions. Inspect and refine subjects before LoRA training.",
        "usage_note": "Consult WikiArt terms and the dataset card; non-commercial research only."})
    print(f"WikiArt: {train_count} train + {val_count} held-out; reference is train[0]")


def prepare_coco(s, count=25, dev_count=5, seed=2026) -> None:
    manifest_path = Path("data/coco/manifest.json")
    if manifest_path.exists():
        old = read_json(manifest_path)
        if (len(old["images"]), old["dev_count"], old["seed"]) != (count, dev_count, seed):
            raise RuntimeError("Existing COCO selection differs. Use a separate project directory.")
        verify_files(old["images"])
        print("COCO: existing files verified")
        return
    archive = Path("data/.cache/annotations_trainval2017.zip")
    print("Downloading COCO annotations (not the full image archive)")
    download(s, COCO_ARCHIVE, archive)
    with zipfile.ZipFile(archive) as z:
        with z.open("annotations/captions_val2017.json") as f:
            annotations = json.load(f)
    captions = {}
    for a in sorted(annotations["annotations"], key=lambda a: a["id"]):
        captions.setdefault(a["image_id"], a["caption"].strip())
    images = sorted(annotations["images"], key=lambda x: x["id"])
    selected = random.Random(seed).sample(images, count)
    records = []
    for i, row in enumerate(selected):
        url = row.get("coco_url") or f"https://images.cocodataset.org/val2017/{row['file_name']}"
        url = url.replace("http://", "https://", 1)
        raw = Path("data/coco/raw") / row["file_name"]
        url = url.replace("https://images.cocodataset.org/","http://images.cocodataset.org/")
        download(s, url, raw)
        with Image.open(raw) as image:
            processed = square_image(image, 512)
        path = Path(f"data/coco/images/{row['id']:012d}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(path)
        edges = Path(f"data/coco/edges/{row['id']:012d}.png")
        edges.parent.mkdir(parents=True, exist_ok=True)
        edge_rgb(canny_edges(processed)).save(edges)
        caption = captions[row["id"]]
        records.append({"id": row["id"], "path": str(path), "sha256": sha256(path),
                        "raw_path": str(raw), "raw_sha256": sha256(raw), "source_url": url,
                        "source_license_id": row.get("license"), "caption": caption,
                        "prompt": f"An impressionist painting. {caption}",
                        "split": "dev" if i < dev_count else "test", "edges": str(edges)})
    write_json(manifest_path, {"dataset": "COCO val2017", "seed": seed, "dev_count": dev_count,
                              "images": records, "licenses": annotations.get("licenses", []),
                              "annotations_sha256": sha256(archive),
                              "preprocessing": "EXIF correction, RGB, centered square crop, Lanczos 512"})
    print(f"COCO: {dev_count} development + {count-dev_count} test photographs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=["all", "wikiart", "coco"], default="all")
    p.add_argument("--max-scan", type=int, default=5000)
    args = p.parse_args()
    s = session()
    if args.only in {"all", "wikiart"}:
        prepare_wikiart(s, max_scan=args.max_scan)
    if args.only in {"all", "coco"}:
        prepare_coco(s)
    print("Preparation completed. Inspect the crops, references and captions BEFORE running the baseline.")


if __name__ == "__main__":
    main()
