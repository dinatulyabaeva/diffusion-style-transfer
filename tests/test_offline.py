"""Offline tests. Synthetic fixtures are NOT baseline outputs."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.common import (adapter_callback, canny_edges, latent_for, schedule_values,
                        sha256, square_image, verify_files, write_json)
from src.metrics import clustered_summary, edge_scores


class ImageTests(unittest.TestCase):
    def test_square_rgb(self):
        im = square_image(Image.new("L", (180, 100)), 64)
        self.assertEqual(im.size, (64, 64))
        self.assertEqual(im.mode, "RGB")

    def test_canny_is_repeatable(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        a[16:48, 16:48] = 255
        im = Image.fromarray(a)
        first = canny_edges(im)
        self.assertGreater(int(first.sum()), 0)
        np.testing.assert_array_equal(first, canny_edges(im))

    def test_canny_rejects_thresholds(self):
        with self.assertRaises(ValueError):
            canny_edges(Image.new("RGB", (16, 16)), 200, 100)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.edge = np.zeros((32, 32), dtype=np.uint8)
        self.edge[4:28, 12] = 255

    def test_identical_is_one(self):
        self.assertAlmostEqual(edge_scores(self.edge, self.edge)["f1"], 1)

    def test_within_tolerance(self):
        shifted = np.roll(self.edge, 2, axis=1)
        self.assertAlmostEqual(edge_scores(self.edge, shifted, 2)["f1"], 1)

    def test_outside_tolerance(self):
        shifted = np.roll(self.edge, 4, axis=1)
        self.assertAlmostEqual(edge_scores(self.edge, shifted, 2)["f1"], 0)

    def test_empty_is_not_success(self):
        empty = np.zeros((32, 32), dtype=np.uint8)
        self.assertIsNone(edge_scores(empty, empty)["f1"])
        self.assertEqual(edge_scores(empty, self.edge)["f1"], 0)

    def test_mismatched_shapes_fail(self):
        with self.assertRaises(ValueError):
            edge_scores(self.edge, np.zeros((8, 8)))

    def test_bootstrap_clusters_by_photo(self):
        rows = [{"photo_id": 1, "x": x} for x in [0, 0, 0]] + [{"photo_id": 2, "x": 1}]
        s = clustered_summary(rows, "x")
        self.assertEqual(s["n_photos"], 2)
        self.assertAlmostEqual(s["mean"], 0.5)
        self.assertEqual(s, clustered_summary(rows, "x"))

    def test_missing_values(self):
        self.assertIsNone(clustered_summary([{"photo_id": 1, "x": None}], "x")["mean"])


class ScheduleTests(unittest.TestCase):
    def test_means_match(self):
        for kind in ["constant", "linear_up", "linear_down"]:
            v = schedule_values(30, 0.6, kind, 0.3)
            self.assertEqual(len(v), 30)
            self.assertAlmostEqual(np.mean(v), 0.6)

    def test_direction(self):
        up = schedule_values(30, 0.6, "linear_up")
        down = schedule_values(30, 0.6, "linear_down")
        np.testing.assert_allclose(up, down[::-1])

    def test_bad_weights_fail(self):
        with self.assertRaises(ValueError):
            schedule_values(30, 0.1, "linear_up", 0.3)

    def test_callback_step_alignment(self):
        class FakePipeline:
            def __init__(self): self.values = []
            def set_ip_adapter_scale(self, value): self.values.append(value)
        pipe = FakePipeline()
        values = [0.3, 0.6, 0.9]
        setter, callback = adapter_callback("ip_adapter", values)
        setter(pipe, values[0])
        for step in range(3):
            payload = {"latents": "dummy"}
            self.assertIs(callback(pipe, step, None, payload), payload)
        self.assertEqual(pipe.values, values)

    def test_lora_callback_does_not_call_ip(self):
        class FakePipeline:
            def __init__(self): self.calls = []
            def set_adapters(self, names, adapter_weights):
                self.calls.append((names, adapter_weights))
        pipe = FakePipeline()
        setter, callback = adapter_callback("lora", [0.5, 0.8])
        setter(pipe, 0.5)
        callback(pipe, 0, None, {})
        self.assertEqual(pipe.calls, [(["style"], [0.5]), (["style"], [0.8])])


class ReproducibilityTests(unittest.TestCase):
    def test_noise_persisted_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            a, path = latent_for(123, 17, 512, d)
            b, other = latent_for(123, 17, 512, d)
            self.assertEqual(a.shape, (1, 4, 64, 64))
            np.testing.assert_array_equal(a, b)
            self.assertEqual(path, other)
            c, _ = latent_for(123, 42, 512, d)
            self.assertFalse(np.array_equal(a, c))

    def test_tampering_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.txt"
            p.write_text("before")
            rows = [{"path": str(p), "sha256": sha256(p)}]
            verify_files(rows)
            p.write_text("after")
            with self.assertRaises(RuntimeError): verify_files(rows)

    def test_dry_run_no_model_and_no_output_images(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            im = d / "fixture.png"
            Image.new("RGB", (512, 512), (100, 100, 100)).save(im)
            manifest = d / "manifest.json"
            write_json(manifest, {"images": [{"id": 1, "split": "dev", "path": str(im),
                "sha256": sha256(im), "prompt": "SYNTHETIC TEST FIXTURE, NOT A BASELINE"}]})
            cfg = json.loads((root / "configs/baseline.json").read_text())
            cfg.update(photos_manifest=str(manifest), style_reference=str(im))
            config = d / "config.json"
            write_json(config, cfg)
            env = {**os.environ, "PYTHONPATH": str(root)}
            result = subprocess.run([sys.executable, "-m", "src.run_baseline", "--config", str(config),
                                     "--dry-run"], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            run = json.loads((d / "outputs/dry_run.json").read_text())
            self.assertEqual(run["status"], "VALIDATED_WITHOUT_GENERATION")
            self.assertEqual(len(run["noise"]), 3)
            self.assertEqual(list((d / "outputs").glob("*.png")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
