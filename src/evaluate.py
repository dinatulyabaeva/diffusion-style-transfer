"""Compute tolerant edge F1, DINOv2 cosine and clustered uncertainty.

Run AFTER the generation subprocess ends, to free diffusion-model GPU memory.
The visual style score is deliberately NOT fabricated or replaced by DINO.
"""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np
from PIL import Image

from .common import canny_edges, read_json, sha256, verify_files, write_json
from .metrics import clustered_summary, edge_scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="outputs/ip_const_0.6")
    p.add_argument("--edges-only", action="store_true", help="Diagnostic mode; DINO remains uncomputed")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    out = Path(args.run_dir)
    run = read_json(out / "run.json")
    cfg = run["config"]
    if sha256(cfg["photos_manifest"]) != run["manifest_sha256"]:
        raise RuntimeError("The source manifest changed after generation")
    verify_files(read_json(cfg["photos_manifest"])["images"])
    records = []
    for f in sorted(out.glob("*_s*.json")):
        r = read_json(f)
        if r.get("fingerprint") != run["fingerprint"]:
            raise ValueError(f"Mixed configurations in {out}")
        if sha256(r["output_path"]) != r["image_sha256"]:
            raise ValueError(f"Output was modified: {r['output_path']}")
        records.append(r)
    if not records:
        raise RuntimeError("No generated outputs. Run generation first.")
    if not args.edges_only:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("No CUDA; pass --device cpu for metric computation")
        revision = run["model_revisions"][cfg["dino_model"]]
        processor = AutoImageProcessor.from_pretrained(cfg["dino_model"], revision=revision, use_fast=False)
        model = AutoModel.from_pretrained(cfg["dino_model"], revision=revision,
                                         use_safetensors=True).to(args.device).eval()
        feature_cache = {}

        def embed(path):
            key = sha256(path)
            if key not in feature_cache:
                with Image.open(path) as im:
                    inputs = processor(images=im.convert("RGB"), return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    feature = model(**inputs).last_hidden_state[:, 0, :].float()
                    feature = torch.nn.functional.normalize(feature, dim=-1)
                feature_cache[key] = feature[0].cpu().numpy()
            return feature_cache[key]
    results = []
    for r in records:
        row = {k: r[k] for k in ["photo_id", "seed", "split", "experiment", "method", "seconds",
                                 "unet_forward_calls", "controlnet_forward_calls", "nsfw_flagged"]}
        row.update(edge_f1=None, edge_precision=None, edge_recall=None, dino_cosine=None)
        if not r["nsfw_flagged"]:
            with Image.open(r["source_path"]) as source, Image.open(r["output_path"]) as generated:
                e1 = canny_edges(source, cfg["canny_low"], cfg["canny_high"], cfg["canny_sigma"])
                e2 = canny_edges(generated, cfg["canny_low"], cfg["canny_high"], cfg["canny_sigma"])
            scores = edge_scores(e1, e2, cfg["edge_tolerance_px"])
            row.update(edge_f1=scores["f1"], edge_precision=scores["precision"], edge_recall=scores["recall"])
            if not args.edges_only:
                row["dino_cosine"] = float(np.clip(np.dot(embed(r["source_path"]), embed(r["output_path"])), -1, 1))
        results.append(row)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summaries = {}
    for split in ["dev", "test", "all"]:
        selected = [r for r in results if split == "all" or r["split"] == split]
        summaries[split] = {"generated_images": len(selected),
            "flagged_images": sum(r["nsfw_flagged"] for r in selected),
            "edge_f1": clustered_summary(selected, "edge_f1"),
            "dino_cosine": clustered_summary(selected, "dino_cosine"),
            "seconds": clustered_summary(selected, "seconds")}
    write_json(out / "summary.json", {"fingerprint": run["fingerprint"],
        "metrics_status": "EDGE_ONLY_DIAGNOSTIC" if args.edges_only else "EDGE_AND_DINO_COMPUTED",
        "dino_feature": "last_hidden_state[:, 0, :] / L2 norm", "summaries": summaries,
        "style_score": None, "style_score_note": "Human assessment required; not inferred from DINO."})
    # This file is never overwritten, so existing human observations are preserved.
    rating_file = out / "visual_review.csv"
    if not rating_file.exists():
        with rating_file.open("w", newline="", encoding="utf-8") as f:
            fields = ["photo_id", "seed", "reviewed", "style_1_to_5", "content_1_to_5",
                      "artifacts_0_to_2", "comment"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in records:
                writer.writerow({"photo_id": r["photo_id"], "seed": r["seed"]})
    def rel(path):
        import os
        return html.escape(os.path.relpath(path, out), quote=True)
    cards = []
    for r in records:
        original = rel(r["source_path"])
        edge = rel(f"data/coco/edges/{r['photo_id']:012d}.png")
        result = rel(r["output_path"])
        style = rel(cfg["style_reference"])
        cards.append(f'<section><h3>COCO {r["photo_id"]}; seed {r["seed"]}; '
                     f'{html.escape(r["split"])}; flagged={r["nsfw_flagged"]}</h3>'
                     f'<p>{html.escape(r["prompt"])}</p><div class="row">'
                     f'<figure><img src="{original}"><figcaption>Исходник</figcaption></figure>'
                     f'<figure><img src="{edge}"><figcaption>Canny</figcaption></figure>'
                     f'<figure><img src="{style}"><figcaption>Образец стиля</figcaption></figure>'
                     f'<figure><img src="{result}"><figcaption>Результат</figcaption></figure></div></section>')
    content = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Baseline gallery</title>
<style>body{font-family:system-ui;max-width:1200px;margin:30px auto;padding:0 20px}
.row{display:flex;gap:10px}figure{margin:0;width:25%}img{width:100%;height:auto}
section{padding:20px 0;border-bottom:1px solid #ccc}figcaption{font-size:14px}</style>
<h1>Визуальная проверка baseline</h1><p>Это реальные сохранённые результаты текущего запуска.
Заполните visual_review.csv; автоматические метрики не подтверждают качество стиля.</p>''' + "\n".join(cards) + "</html>"
    (out / "gallery.html").write_text(content, encoding="utf-8")
    print(f"Saved metrics.csv, summary.json, gallery.html and visual_review.csv to {out}")


if __name__ == "__main__":
    main()
