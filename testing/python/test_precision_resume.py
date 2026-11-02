# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytorch_precision_resume as qualification


class CheckpointReceiptTest(unittest.TestCase):
    def test_native_payload_tampering_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoints/step-00000004'
            checkpoint.mkdir(parents=True)
            files = {}
            for name in ('model.safetensors', 'runtime.pt', 'state.json'):
                path = checkpoint / name
                path.write_bytes(b'original')
                files[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': qualification.base.digest(path)}
            qualification.base.write_json(root / 'checkpoint-ready.json', {'files': files, 'item': {'step': 4}})
            qualification.checkpoint_receipt(root)
            (checkpoint / 'runtime.pt').write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                qualification.checkpoint_receipt(root)

    def test_unexpected_checkpoint_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            qualification.base.write_json(Path(directory) / 'checkpoint-ready.json', {'files': {'../outside': {}}, 'item': {'step': 4}})
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                qualification.checkpoint_receipt(directory)


class NegativeMatrixTest(unittest.TestCase):
    def test_completed_mismatch_continues_and_verifies_as_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(device='cpu', strategy='data-parallel', mode='resume',
                                   fixture=directory, output=str(Path(directory) / 'result'),
                                   world_sizes=[1], timeout_seconds=30, gpu_hour_budget=1)
            (Path(directory) / 'MANIFEST.json').write_text('{}')
            errors = [AssertionError('deliberate numerical mismatch'), 10]
            def launched(args, directory, *positional, **keywords):
                Path(directory).mkdir(parents=True)
                return {'status': 'pass'}
            with patch.object(qualification.base, 'verify_fixture', return_value={'steps': 8}), \
                 patch.object(qualification, 'runtime'), \
                 patch.object(qualification, 'torch', SimpleNamespace(__version__='test', version=SimpleNamespace(cuda=None)), create=True), \
                 patch.object(qualification, 'launch_worker', side_effect=launched) as launch, \
                 patch.object(qualification, 'check_resume', side_effect=errors):
                qualification.run(args)
                self.assertEqual(launch.call_count, 6)
            path = Path(args.output) / 'summary.json'
            summary = json.loads(path.read_text())
            self.assertEqual(summary['status'], 'fail')
            self.assertFalse(summary['cuda_qualified'])
            self.assertEqual([c['status'] for c in summary['cases']], ['fail', 'pass'])
            with patch.object(qualification.base, 'verify_fixture', return_value={'steps': 8}), \
                 patch.object(qualification, 'runtime'), \
                 patch.object(qualification, 'check_resume', side_effect=errors):
                qualification.verify(args)
            summary['status'] = 'pass'
            path.write_text(json.dumps(summary))
            qualification.seal(args.output)
            with patch.object(qualification.base, 'verify_fixture', return_value={'steps': 8}), \
                 patch.object(qualification, 'runtime'), \
                 patch.object(qualification, 'check_resume', side_effect=errors), \
                 self.assertRaisesRegex(AssertionError, 'aggregate verdict'):
                qualification.verify(args)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires optional CPU PyTorch')
class ExactResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import pytorch_conformance
        qualification.torch = torch
        qualification.base.torch = torch
        qualification.base.oracle = pytorch_conformance
        cls.torch = torch

    def test_resume_rejects_drift_below_bf16_tolerance(self):
        left = self.torch.tensor([0.0])
        right = self.torch.tensor([1e-8])
        with self.assertRaisesRegex(AssertionError, 'bitwise'):
            qualification.exact('weights', left, right, [0])

    def test_rng_change_rejected(self):
        left = self.torch.tensor([1, 2, 3], dtype=self.torch.uint8)
        right = self.torch.tensor([1, 2, 4], dtype=self.torch.uint8)
        with self.assertRaises(AssertionError):
            qualification.exact('rng', left, right, [0])

    def test_missing_tensor_and_reordered_samples_rejected(self):
        for left, right in (({'w': self.torch.ones(1)}, {}), ([1, 2], [2, 1])):
            with self.assertRaises(AssertionError):
                qualification.exact('state', left, right, [0])

    def test_large_gradient_error_fails_bf16_policy(self):
        expected = self.torch.tensor([0.1, -0.1])
        with self.assertRaises(AssertionError):
            qualification.base.compare_tensor('gradient', expected / 2, expected,
                                               qualification.BF16_TOLERANCES['gradients'], {})


if __name__ == '__main__':
    unittest.main()
