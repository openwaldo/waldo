# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pytorch_fixed_batch as fixed


class FixedBatchContractTest(unittest.TestCase):
    def test_matrix_holds_global_batch_across_world_sizes(self):
        cells = fixed.matrix([1, 2, 4])
        self.assertEqual(len(cells), 10)
        self.assertEqual(len(cells), len(set(cells)))
        for case in ("dense", "uneven"):
            for world in (1, 2, 4):
                self.assertIn((case, world, 1), cells)
                self.assertIn((case, world, 4 // world), cells)
        for _, world, accumulation in cells:
            self.assertEqual(8 % (world * accumulation), 0)

    def test_invalid_world_size_and_duplicate_are_rejected(self):
        for worlds in ([], [1, 1], [3], [1, 8]):
            with self.subTest(worlds=worlds), self.assertRaises(ValueError):
                fixed.matrix(worlds)

    def test_cuda_requires_explicit_authorization(self):
        with patch.dict(os.environ, {}, clear=True):
            fixed.require_cuda_authorization("cpu")
            with self.assertRaisesRegex(ValueError, "allocation"):
                fixed.require_cuda_authorization("cuda")

    def test_fixture_rejects_tampering_missing_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "initial.safetensors"
            fixture = root / "fixture.json"
            weights.write_bytes(b"weights")
            fixture.write_text("{}")
            fixed.write_json(root / "MANIFEST.json", {"schema": 1, "files": {
                p.name: {"bytes": p.stat().st_size, "sha256": fixed.digest(p)}
                for p in (weights, fixture)}})
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                fixed.verify_fixture(root)
            weights.unlink()
            with self.assertRaises(FileNotFoundError):
                fixed.verify_fixture(root)
            weights.symlink_to(fixture)
            with self.assertRaisesRegex(ValueError, "integrity"):
                fixed.verify_fixture(root)

    def test_evidence_inventory_rejects_changes_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.json"
            payload.write_text("original")
            (root / "SHA256SUMS").write_text(f"{fixed.digest(payload)}  payload.json\n")
            fixed.verify_inventory(root)
            (root / "unlisted.json").write_text("extra")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                fixed.verify_inventory(root)
            (root / "unlisted.json").unlink()
            payload.write_text("changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                fixed.verify_inventory(root)

    def test_evidence_inventory_rejects_traversal_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.json"
            payload.write_text("original")
            line = f"{fixed.digest(payload)}  payload.json\n"
            for inventory in (line + line, "0000  ../outside\n"):
                (root / "SHA256SUMS").write_text(inventory)
                with self.assertRaisesRegex(ValueError, "invalid or duplicated"):
                    fixed.verify_inventory(root)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU PyTorch runtime")
class TensorComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import pytorch_conformance
        fixed.torch, fixed.oracle = torch, pytorch_conformance
        cls.torch = torch

    def test_changed_gradient_cannot_pass(self):
        expected = self.torch.tensor([0.02, -0.03])
        with self.assertRaises(AssertionError):
            fixed.compare_tensor("gradient", expected / 2, expected, fixed.TOLERANCES["gradients"], {})

    def test_nonfinite_and_wrong_shape_are_rejected(self):
        expected = self.torch.tensor([1.0])
        for actual in (self.torch.tensor([float("nan")]), self.torch.tensor([float("inf")]),
                       self.torch.tensor([[1.0]]), expected.double()):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                fixed.compare_tensor("weights", actual, expected, fixed.TOLERANCES["weights"], {})

    def test_exact_match_records_zero_error(self):
        value = self.torch.tensor([1.0, 0.0])
        errors = {}
        fixed.compare_tensor("weights", value, value, fixed.TOLERANCES["weights"], errors)
        self.assertEqual(errors, {"weights": 0.0})


if __name__ == "__main__":
    unittest.main()
