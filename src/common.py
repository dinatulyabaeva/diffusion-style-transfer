"""I/O, image preprocessing, fair schedules and reproducible latent noise."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def square_image(image: Image.Image, size: int = 512) -> Image.Image:
    """Fixed EXIF correction + central square crop, never aspect-ratio stretching."""
    if size <= 0:
        raise ValueError("Image size must be positive")
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS,
                        centering=(0.5, 0.5))


def canny_edges(image: Image.Image, low: int = 100, high: int = 200,
                sigma: float = 1.0) -> np.ndarray:
    if not 0 <= low < high <= 255:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=sigma, sigmaY=sigma)
    return cv2.Canny(gray, low, high, L2gradient=True)


def edge_rgb(edges: np.ndarray) -> Image.Image:
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


def schedule_values(n: int, base: float, kind: str = "constant",
                    delta: float = 0.3) -> list[float]:
    """Three schedules with the SAME discrete mean; no off-by-one timestep shift."""
    if n < 1 or base < 0 or delta < 0:
        raise ValueError("Invalid schedule arguments")
    if kind not in {"constant", "linear_up", "linear_down"}:
        raise ValueError(f"Unknown schedule: {kind}")
    if kind == "constant" or n == 1:
        return [float(base)] * n
    if base < delta:
        raise ValueError("Use base >= delta so all adapter weights stay nonnegative")
    vals = np.linspace(base - delta, base + delta, n, dtype=np.float64)
    if kind == "linear_down":
        vals = vals[::-1]
    return vals.tolist()


def adapter_callback(method: str, values: list[float]):
    """The callback runs AFTER a step, so set the weight for step i+1, not i."""
    def set_weight(pipe, value: float) -> None:
        if method == "ip_adapter":
            pipe.set_ip_adapter_scale(float(value))
        elif method == "lora":
            pipe.set_adapters(["style"], adapter_weights=[float(value)])
        elif method != "control_only":
            raise ValueError(method)

    def callback(pipe, step_index, timestep, callback_kwargs):
        if step_index + 1 < len(values):
            set_weight(pipe, values[step_index + 1])
        return callback_kwargs

    return set_weight, callback


def latent_for(photo_id: int | str, seed: int, size: int,
               directory: str | Path = "data/latents") -> tuple[np.ndarray, Path]:
    """Persist standard-normal float32 latents BEFORE scheduler scaling.

    NumPy PCG64 noise is generated once on CPU. Every branch loads the exact
    same .npy file; seed alone would not prove that the starting tensor matches.
    """
    if size % 8:
        raise ValueError("SD 1.5 resolution must be divisible by 8")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = f"{photo_id}|{seed}|{size}"
    sub_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    path = directory / f"{photo_id}_s{seed}_{size}.npy"
    shape = (1, 4, size // 8, size // 8)
    if not path.exists():
        arr = np.random.Generator(np.random.PCG64(sub_seed)).standard_normal(shape).astype(np.float32)
        np.save(path, arr, allow_pickle=False)
    arr = np.load(path, allow_pickle=False)
    if arr.shape != shape or arr.dtype != np.float32 or not np.isfinite(arr).all():
        raise ValueError(f"Corrupt or incompatible latent: {path}")
    return arr, path


def verify_files(rows: list[dict], path_key: str = "path") -> None:
    for row in rows:
        p = Path(row[path_key])
        if not p.is_file() or sha256(p) != row["sha256"]:
            raise RuntimeError(f"Missing or modified data file: {p}")
