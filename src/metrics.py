"""Project-defined tolerant edge F1 and photograph-level bootstrap."""
from __future__ import annotations

import cv2
import numpy as np


def edge_scores(source: np.ndarray, generated: np.ndarray,
                tolerance: float = 2.0) -> dict:
    """Symmetric edge matching within Euclidean distance <= tolerance pixels.

    Both empty sets are uninformative (None), not an artificial perfect score.
    This tolerant many-to-one matching is a proxy, not a standard benchmark.
    """
    if source.ndim != 2 or source.shape != generated.shape or tolerance < 0:
        raise ValueError("Expected matching 2-D edge maps and a nonnegative tolerance")
    src, gen = source > 0, generated > 0
    n_src, n_gen = int(src.sum()), int(gen.sum())
    counts = {"source_edges": n_src, "generated_edges": n_gen}
    if n_src == 0 and n_gen == 0:
        return {"precision": None, "recall": None, "f1": None, **counts}
    if n_src == 0 or n_gen == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, **counts}
    distance_src = cv2.distanceTransform((~src).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_gen = cv2.distanceTransform((~gen).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    precision = float(np.mean(distance_src[gen] <= tolerance))
    recall = float(np.mean(distance_gen[src] <= tolerance))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, **counts}


def clustered_summary(rows: list[dict], key: str, repetitions: int = 2000,
                      seed: int = 2026) -> dict:
    """Average seeds per photograph, then bootstrap independent photographs."""
    by_photo: dict[str, list[float]] = {}
    for r in rows:
        val = r.get(key)
        if val is not None and np.isfinite(val):
            by_photo.setdefault(str(r["photo_id"]), []).append(float(val))
    if not by_photo:
        return {"mean": None, "ci95": [None, None], "n_photos": 0}
    values = np.array([np.mean(v) for v in by_photo.values()])
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "n_photos": len(values), "bootstrap_repetitions": repetitions}
