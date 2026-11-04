"""Run IP-Adapter, LoRA or text+ControlNet with the same stored noise.

A dry run validates configuration and data without importing model libraries.
A real run requires a CUDA GPU. No generated result is bundled with this project.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import os
import platform
import time
from pathlib import Path

from PIL import Image

from .common import (adapter_callback, canny_edges, canonical_hash, edge_rgb,
                     latent_for, read_json, schedule_values, sha256, verify_files, write_json)


def lock_models(config: dict, path=Path("models.lock.json")) -> dict:
    from huggingface_hub import HfApi
    lock = read_json(path) if path.exists() else {}
    api = HfApi()
    for key in ["base_model", "controlnet_model", "ip_adapter_model", "dino_model"]:
        repo = config[key]
        if repo not in lock:
            lock[repo] = api.model_info(repo).sha
    write_json(path, lock)
    return lock


def environment(torch) -> dict:
    names = ["torch", "torchvision", "diffusers", "transformers", "peft", "accelerate",
             "numpy", "Pillow", "opencv-python-headless", "huggingface-hub"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--limit", type=int, help="Limit photos for a smoke test; not a complete baseline")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_json(args.config)
    method = cfg["method"]
    if method not in {"ip_adapter", "lora", "control_only"}:
        raise ValueError("Unknown method")
    if cfg["resolution"] != 512:
        raise ValueError("This protocol fixes resolution at 512x512")
    if cfg["steps"] < 2 or not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Need >=2 steps and distinct seeds")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    manifest = read_json(cfg["photos_manifest"])
    rows = [r for r in manifest["images"] if args.split == "all" or r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Empty selection")
    verify_files(rows)
    reference_hash = None
    if method == "ip_adapter":
        reference_hash = sha256(cfg["style_reference"])
    if method == "lora" and not cfg.get("lora_path"):
        raise ValueError("The LoRA configuration needs lora_path")
    values = schedule_values(cfg["steps"], cfg["adapter_scale"], cfg["schedule"], cfg["schedule_delta"])
    noises = []
    # All manifest noises enter the signature, so smoke/dev/test resumes stay compatible.
    for row in manifest["images"]:
        for seed in cfg["seeds"]:
            _, path = latent_for(row["id"], seed, cfg["resolution"])
            noises.append({"photo_id": row["id"], "seed": seed, "path": str(path), "sha256": sha256(path)})
    print(f"{method}: {len(rows)} photos x {len(cfg['seeds'])} seeds = {len(rows) * len(cfg['seeds'])} images")
    if args.dry_run:
        write_json("outputs/dry_run.json", {"status": "VALIDATED_WITHOUT_GENERATION", "config": cfg,
                                           "noise": noises, "weights": values})
        print("Dry run only. NO model was loaded and NO image was generated.")
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required by this reference runner. Enable GPU in Colab or use a CUDA machine.")
    from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline
    if cfg["deterministic"]:
        import random
        import numpy as np
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    lock = lock_models(cfg)
    env = environment(torch)
    signature_data = {"config": cfg, "manifest_sha256": sha256(cfg["photos_manifest"]),
                      "style_reference_sha256": reference_hash, "model_revisions": lock,
                      "environment": env, "all_noise_hash": canonical_hash(noises),
                      "implementation_sha256": {f: sha256(Path(__file__).parent / f)
                                                for f in ["run_baseline.py", "common.py"]}}
    if method == "lora":
        lora_files = sorted(Path(cfg["lora_path"]).glob("*.safetensors"))
        if not lora_files:
            raise FileNotFoundError("No LoRA .safetensors found")
        signature_data["lora_files"] = {str(f): sha256(f) for f in lora_files}
    fingerprint = canonical_hash(signature_data)
    out = Path("outputs") / cfg["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    manifest_out = out / "run.json"
    if manifest_out.exists() and read_json(manifest_out)["fingerprint"] != fingerprint:
        raise RuntimeError("Output directory belongs to another configuration/environment. Choose a new experiment name.")
    write_json(manifest_out, {**signature_data, "fingerprint": fingerprint,
                             "status": "RUNNING", "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                             "selected_split": args.split, "limited_photos": args.limit,
                             "adapter_weights": values, "noise": noises})
    dtype = torch.float16
    controlnet = ControlNetModel.from_pretrained(cfg["controlnet_model"],
        revision=lock[cfg["controlnet_model"]], torch_dtype=dtype, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(cfg["base_model"],
        revision=lock[cfg["base_model"]], controlnet=controlnet,
        torch_dtype=dtype, use_safetensors=True)
    # Keep the checkpoint's safety checker; flagged outputs are recorded, never silently replaced.
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if method == "ip_adapter":
        pipe.load_ip_adapter(cfg["ip_adapter_model"], subfolder="models",
            weight_name=cfg["ip_adapter_weight"], image_encoder_folder="image_encoder",
            revision=lock[cfg["ip_adapter_model"]])
    elif method == "lora":
        pipe.load_lora_weights(cfg["lora_path"], adapter_name="style")
    if cfg["cpu_offload"]:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.enable_vae_slicing()
    if cfg["deterministic"]:
        # Avoid fused CUDA attention kernels whose determinism varies by device/version.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    set_weight, callback = adapter_callback(method, values)
    reference = Image.open(cfg["style_reference"]).convert("RGB") if method == "ip_adapter" else None
    counters = {"unet_forward_calls": 0, "controlnet_forward_calls": 0}
    def count_unet(module, inputs):
        counters["unet_forward_calls"] += 1
    def count_control(module, inputs):
        counters["controlnet_forward_calls"] += 1
    handles = [pipe.unet.register_forward_pre_hook(count_unet),
               pipe.controlnet.register_forward_pre_hook(count_control)]

    def generate(row, seed):
        with Image.open(row["path"]) as im:
            edges = edge_rgb(canny_edges(im, cfg["canny_low"], cfg["canny_high"], cfg["canny_sigma"]))
        arr, noise_path = latent_for(row["id"], seed, cfg["resolution"])
        latents = torch.from_numpy(arr.copy()).to("cuda", dtype=dtype)
        set_weight(pipe, values[0])  # reset before EVERY image, including warm-up
        kwargs = dict(prompt=row["prompt"], negative_prompt=cfg["negative_prompt"],
            image=edges, height=512, width=512, num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance_scale"], controlnet_conditioning_scale=cfg["controlnet_scale"],
            control_guidance_start=cfg["control_guidance_start"],
            control_guidance_end=cfg["control_guidance_end"],
            latents=latents, eta=0.0, generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
        if reference is not None:
            kwargs["ip_adapter_image"] = reference
        counters.update(unet_forward_calls=0, controlnet_forward_calls=0)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            result = pipe(**kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        flagged = bool(result.nsfw_content_detected and result.nsfw_content_detected[0])
        return result.images[0], {"seconds": elapsed, **counters,
            "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "nsfw_flagged": flagged, "latent_sha256": sha256(noise_path)}

    if cfg["warmup"]:
        print("Warm-up; its output is excluded from measurements")
        generate(rows[0], cfg["seeds"][0])
    try:
        for row in rows:
            for seed in cfg["seeds"]:
                name = f"{row['id']:012d}_s{seed}"
                png, meta = out / f"{name}.png", out / f"{name}.json"
                if png.exists() and meta.exists():
                    saved = read_json(meta)
                    if saved["fingerprint"] != fingerprint or sha256(png) != saved["image_sha256"]:
                        raise RuntimeError(f"Invalid existing result: {png}")
                    continue
                image, measurement = generate(row, seed)
                image.save(png)
                write_json(meta, {"photo_id": row["id"], "seed": seed, "split": row["split"],
                    "method": method, "experiment": cfg["experiment"], "fingerprint": fingerprint,
                    "source_path": row["path"], "output_path": str(png), "prompt": row["prompt"],
                    "image_sha256": sha256(png), **measurement})
                print(name, f"{measurement['seconds']:.2f} seconds", "FLAGGED" if measurement["nsfw_flagged"] else "")
        run = read_json(manifest_out)
        run["status"] = "SELECTED_RUN_COMPLETED"
        run["baseline_scope_note"] = "A limited smoke test is not the full 25-photo baseline. Check coverage before claiming completion."
        write_json(manifest_out, run)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
