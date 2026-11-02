# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
import base64
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HELPER = Path(__file__).resolve().parents[2] / 'internal/model/pythonenv/environment.py'
spec = importlib.util.spec_from_file_location('waldo_pythonenv', HELPER)
env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env)


def make_wheel(extra=None):
    files = {
        'sample/__init__.py': b'def main():\n    print("working")\n',
        'sample-1.0.dist-info/METADATA': b'Metadata-Version: 2.1\nName: sample\nVersion: 1.0\n',
        'sample-1.0.dist-info/WHEEL': b'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        'sample-1.0.dist-info/entry_points.txt': b'[console_scripts]\nsample-cli = sample:main\n',
    }
    files.update(extra or {})
    record = io.StringIO()
    writer = csv.writer(record)
    for name, data in files.items():
        writer.writerow([name, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(), len(data)])
    writer.writerow(['sample-1.0.dist-info/RECORD', '', ''])
    files['sample-1.0.dist-info/RECORD'] = record.getvalue().encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)
    return output.getvalue()


class Response(io.BytesIO):
    def geturl(self):
        return 'https://example.invalid/sample-1.0-py3-none-any.whl'


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix='waldo-env-test-'))
        self.addCleanup(env.cleanup, self.temp)
        self.wheel = make_wheel()
        self.lock = self.temp / 'pylock.toml'
        self.lock.write_text(self.lock_text())
        self.request = dict(name='cpu', lock=str(self.lock), cache=str(self.temp / 'cache'), python_version='.'.join(map(str, sys.version_info[:3])))
        self.download = patch.object(env.urllib.request, 'urlopen', side_effect=lambda *a, **k: Response(self.wheel)).start()
        self.addCleanup(patch.stopall)

    def lock_text(self):
        return f'''lock-version = "1.0"
created-by = "waldo-test"
[[packages]]
name = "sample"
version = "1.0"
[[packages.wheels]]
url = "https://example.invalid/sample-1.0-py3-none-any.whl"
hashes = {{sha256 = "{hashlib.sha256(self.wheel).hexdigest()}"}}
'''

    def create(self):
        result = env.ensure(self.request)
        root = Path(result['run']['cache_path'])
        return result, root, env.site_path(root)

    def writable(self, root):
        for base, dirs, files in env.os.walk(root):
            Path(base).chmod(0o755)
            for name in files:
                p = Path(base, name)
                if not p.is_symlink():
                    p.chmod(p.stat().st_mode | 0o200)

    def test_matching_cache_reverify_without_install(self):
        result, root, site = self.create()
        self.assertEqual(result['identity']['lock_sha256'], env.digest(self.lock))
        self.assertEqual(result['run']['installer'], 'pip')
        self.assertFalse((site / 'pip').exists())
        completed = subprocess.run([str(root / 'bin/sample-cli')], capture_output=True, text=True, check=True)
        self.assertEqual(completed.stdout.strip(), 'working')
        self.download.reset_mock()
        with patch.object(env, 'run', side_effect=AssertionError('reinstalled')):
            self.assertEqual(env.ensure(self.request), result)
            self.assertEqual(env.ensure(self.request, 'verify'), result)
        self.download.assert_not_called()
        self.assertEqual(len(list((self.temp / 'cache').iterdir())), 1)

    def test_mutations_fail(self):
        mutations = {
            'extra file': lambda root, site: (site / 'extra.py').write_text('x'),
            'extra bin file': lambda root, site: (root / 'bin/extra').write_text('x'),
            'modified file': lambda root, site: (site / 'sample/__init__.py').write_text('x'),
            'missing package': lambda root, site: shutil.rmtree(site / 'sample-1.0.dist-info'),
            'extra package': lambda root, site: shutil.copytree(site / 'sample-1.0.dist-info', site / 'extra-1.0.dist-info'),
            'version mismatch': lambda root, site: (site / 'sample-1.0.dist-info/METADATA').write_text('Name: sample\nVersion: 2.0\n'),
            'system packages': lambda root, site: (root / 'pyvenv.cfg').write_text('include-system-site-packages = true\n'),
            'local install': lambda root, site: (site / 'sample-1.0.dist-info/direct_url.json').write_text('{"url":"file:///tmp/x"}'),
            'unattested pth': lambda root, site: (site / 'extra.pth').write_text('import os\n'),
            'RECORD tampering': lambda root, site: (site / 'sample-1.0.dist-info/RECORD').write_text(''),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                self.request['cache'] = str(self.temp / label)
                result, root, site = self.create()
                self.writable(root)
                mutate(root, site)
                with self.assertRaises((ValueError, FileNotFoundError)) as error:
                    env.ensure(self.request)
                self.assertTrue(str(error.exception))

    def test_concurrent_creators_publish_one_entry(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: env.ensure(self.request), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(list((self.temp / 'cache').iterdir())), 1)

    def test_extra_package(self):
        result, root, site = self.create()
        self.writable(root)
        extra = site / 'extra-1.0.dist-info'
        shutil.copytree(site / 'sample-1.0.dist-info', extra)
        (extra / 'METADATA').write_text('Name: extra\nVersion: 1.0\n')
        with self.assertRaisesRegex(ValueError, 'extra package: extra'):
            env.ensure(self.request)

    def test_interrupted_publication(self):
        with patch.object(env, 'freeze', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                env.ensure(self.request)
        self.assertEqual(list((self.temp / 'cache').iterdir()), [])

    def test_rewritten_record_cannot_hide_modified_file(self):
        result, root, site = self.create()
        self.writable(root)
        target = site / 'sample/__init__.py'
        target.write_text('changed = True\n')
        record = site / 'sample-1.0.dist-info/RECORD'
        rows = list(csv.reader(io.StringIO(record.read_text())))
        for row in rows:
            if row[0] == 'sample/__init__.py':
                row[1] = 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(env.digest(target))).rstrip(b'=').decode()
                row[2] = str(target.stat().st_size)
        with record.open('w', newline='') as f:
            csv.writer(f).writerows(rows)
        with self.assertRaisesRegex(ValueError, 'installation receipt'):
            env.ensure(self.request)

    def test_interpreter_link_tampering(self):
        result, root, site = self.create()
        self.writable(root)
        target = root / 'bin/python'
        target.unlink()
        target.symlink_to('/bin/false')
        with self.assertRaisesRegex(ValueError, 'modified venv baseline'):
            env.ensure(self.request)

    def test_wheel_cannot_replace_interpreter(self):
        self.wheel = make_wheel({'sample-1.0.dist-info/entry_points.txt': b'[console_scripts]\npython = sample:main\n'})
        self.lock.write_text(self.lock_text())
        with self.assertRaisesRegex(ValueError, 'unsafe wheel entry point'):
            env.ensure(self.request)

    def test_shared_wheel_data_is_relocated_and_attested(self):
        self.wheel = make_wheel({'sample-1.0.data/data/share/man/man1/sample.1': b'man page'})
        self.lock.write_text(self.lock_text())
        result, root, _ = self.create()
        page = root / 'share/man/man1/sample.1'
        self.assertEqual(page.read_bytes(), b'man page')
        env.ensure(self.request, 'verify')
        self.writable(root)
        page.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'RECORD mismatch'):
            env.ensure(self.request, 'verify')

    def test_wheel_data_cannot_target_venv_or_escape(self):
        for member in ('data/bin/python', 'data/pyvenv.cfg', 'data/share/../../bin/python', 'headers/a.h'):
            with self.subTest(member=member):
                wheel = self.temp / 'unsafe.whl'
                wheel.write_bytes(make_wheel({'sample-1.0.data/' + member: b'bad'}))
                with self.assertRaises(ValueError):
                    env.check_wheel(wheel, {})

    def test_locked_pth_is_not_executed_by_verifier(self):
        marker = self.temp / 'executed'
        self.wheel = make_wheel({'sample.pth': f'import pathlib; pathlib.Path({str(marker)!r}).touch()\n'.encode()})
        self.lock.write_text(self.lock_text())
        self.create()
        env.ensure(self.request, 'verify')
        self.assertFalse(marker.exists())

    def test_wrong_interpreter(self):
        self.request['python_version'] = '3.0.0'
        with self.assertRaisesRegex(ValueError, 'wrong interpreter'):
            env.ensure(self.request)
        self.download.assert_not_called()

    def test_interrupted_creation(self):
        with patch.object(env, 'run', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                env.ensure(self.request)
        self.assertEqual(list((self.temp / 'cache').iterdir()), [])

    def test_bad_hash(self):
        self.wheel += b'tampered'
        with self.assertRaisesRegex(ValueError, 'wheel hash mismatch'):
            env.ensure(self.request)
        self.assertEqual(list((self.temp / 'cache').iterdir()), [])

    def test_markers_select_platform(self):
        data = self.lock_text() + '''
[[packages]]
name = "not-for-this-python"
version = "1.0"
marker = 'python_version < "2"'
'''
        self.assertEqual(list(env.parse_lock(data.encode())), ['sample'])
        data = data.replace('python_version < "2"', 'python_version >= "2"')
        with self.assertRaisesRegex(ValueError, 'no compatible locked wheel'):
            env.parse_lock(data.encode())

    def test_no_matching_environment(self):
        data = 'environments = [\'python_version < "2"\']\n' + self.lock_text()
        with self.assertRaisesRegex(ValueError, 'environments do not match'):
            env.parse_lock(data.encode())

    def test_reject_source_and_optional_groups(self):
        for data in [self.lock_text().replace('version = "1.0"\n[[packages.wheels]]', 'version = "1.0"\ndirectory = {path="."}\n[[packages.wheels]]'),
                     'default-groups = ["main"]\n' + self.lock_text()]:
            with self.assertRaises(ValueError):
                env.parse_lock(data.encode())

    def test_added_pth_never_executes_during_verification(self):
        result, root, site = self.create()
        self.writable(root)
        marker = self.temp / 'executed'
        (site / 'attack.pth').write_text(f'import pathlib; pathlib.Path({str(marker)!r}).touch()\n')
        with self.assertRaisesRegex(ValueError, 'unattested extra file'):
            env.ensure(self.request)
        self.assertFalse(marker.exists())

    def test_record_path_escape(self):
        result, root, site = self.create()
        self.writable(root)
        with (site / 'sample-1.0.dist-info/RECORD').open('a') as f:
            f.write('../../../../outside,,\n')
        with self.assertRaisesRegex(ValueError, 'escaping RECORD'):
            env.ensure(self.request)

    def test_named_environment_shares_cache(self):
        result, root, site = self.create()
        self.request['name'] = 'tokenizer'
        other = env.ensure(self.request)
        self.assertEqual(result['run'], other['run'])
        self.assertEqual(other['identity']['name'], 'tokenizer')


if __name__ == '__main__':
    unittest.main()
