# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Trusted environment helper, embedded in WALDO. Never import target packages."""
import base64
import csv
import configparser
import email.parser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
import urllib.request
import venv
import zipfile

# pip is host tooling, never implicitly added to the target environment.
from pip import __version__ as pip_version
from pip._vendor.packaging.markers import Marker, default_environment
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
from pip._vendor.packaging.version import Version

RECEIPT = "waldo-environment.json"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message):
    raise ValueError(message)


def marker_matches(value):
    # PoC consumes single-use locks. Fail closed for optional group selection.
    if re.search(r"\b(extras?|dependency_groups)\b", value):
        fail("extras/dependency-group markers are not supported; export a single-use lock")
    return Marker(value).evaluate(default_environment())


def parse_lock(data):
    lock = tomllib.loads(data.decode("utf-8"))
    if lock.get("lock-version") != "1.0" or not lock.get("created-by"):
        fail("pylock requires lock-version 1.0 and created-by")
    if not isinstance(lock.get("packages"), list):
        fail("pylock requires a packages array")
    if any(lock.get(k) for k in ("extras", "dependency-groups", "default-groups")):
        fail("export a single-use pylock without extras or dependency groups")
    version = ".".join(map(str, sys.version_info[:3]))
    if version not in SpecifierSet(lock.get("requires-python", "")):
        fail("lock requires-python does not match interpreter")
    if lock.get("environments") and not any(marker_matches(m) for m in lock["environments"]):
        fail("lock environments do not match this platform")
    tags = list(sys_tags())
    ranks = {tag: i for i, tag in enumerate(tags)}
    selected = {}
    for package in lock["packages"]:
        if not marker_matches(package.get("marker", 'python_version >= "0"')):
            continue
        name = package.get("name", "")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            fail("package name must be normalized")
        if name in selected:
            fail(f"multiple matching lock entries for {name}")
        if any(k in package for k in ("vcs", "directory", "archive")):
            fail(f"{name}: local, editable and source installs are forbidden")
        if not package.get("version"):
            fail(f"{name}: exact version required")
        expected_version = Version(package["version"])
        if version not in SpecifierSet(package.get("requires-python", "")):
            fail(f"{name}: requires-python does not match interpreter")
        candidates = []
        for wheel in package.get("wheels", []):
            url = urllib.parse.urlsplit(wheel.get("url", ""))
            if "path" in wheel or url.scheme != "https" or not url.netloc or url.username or url.password or url.fragment:
                fail(f"{name}: wheel requires an HTTPS URL (no local installs)")
            filename = wheel.get("name") or urllib.parse.unquote(url.path.rsplit("/", 1)[-1])
            if Path(filename).name != filename or "\\" in filename:
                fail(f"{name}: unsafe wheel filename")
            wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(filename)
            if wheel_name != name or wheel_version != expected_version:
                fail(f"{name}: wheel filename does not match locked name/version")
            sha = wheel.get("hashes", {}).get("sha256", "")
            if not re.fullmatch(r"[a-f0-9]{64}", sha):
                fail(f"{name}: wheel requires a SHA-256 hash")
            compatible = wheel_tags.intersection(ranks)
            if compatible:
                candidates.append((min(ranks[t] for t in compatible), filename, wheel))
        if not candidates:
            fail(f"{name}: no compatible locked wheel; sdists and builds are forbidden")
        _, filename, wheel = min(candidates, key=lambda item: (item[0], item[1]))
        selected[name] = {"version": package["version"], "filename": filename, "wheel": wheel}
    return selected


def inventory(root):
    """Attest all venv files, including bin, cfg and RECORDs; never follow links."""
    result = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(base, name)
            rel = path.relative_to(root).as_posix()
            if rel == RECEIPT:
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                result[rel] = {"link": os.readlink(path)}
            elif stat.S_ISREG(info.st_mode):
                if path.suffix == ".pyc" and path.parent.name == "__pycache__":
                    continue
                result[rel] = {"sha256": digest(path), "executable": bool(info.st_mode & 0o111)}
            elif not stat.S_ISDIR(info.st_mode):
                fail(f"unsupported environment file: {rel}")
    return result


def site_path(root):
    return root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def safe_record_path(root, site, value):
    if not value or "\\" in value or PurePosixPath(value).is_absolute():
        fail(f"unsafe RECORD path: {value}")
    path = Path(os.path.abspath(site / value))
    if not path.is_relative_to(root):
        fail(f"escaping RECORD path: {value}")
    current = path
    while current != root:
        if current.is_symlink():
            fail(f"symlink in RECORD path: {value}")
        current = current.parent
    if not path.is_file():
        fail(f"missing installed file: {value}")
    return path


def verify_packages(root, expected, baseline):
    cfg = {}
    for line in (root / "pyvenv.cfg").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            cfg[key.strip().lower()] = value.strip().lower()
    if cfg.get("include-system-site-packages") != "false":
        fail("pyvenv.cfg must have include-system-site-packages = false")
    site = site_path(root)
    installed, owned = {}, set()
    for dist in sorted(site.glob("*.dist-info")):
        if dist.is_symlink():
            fail(f"symlink distribution: {dist.name}")
        metadata = email.parser.Parser().parsestr((dist / "METADATA").read_text())
        name = canonicalize_name(metadata.get("Name", ""))
        version = metadata.get("Version", "")
        if name in installed:
            fail(f"duplicate installed package: {name}")
        installed[name] = version
        if name not in expected:
            fail(f"extra package: {name}")
        if Version(version) != Version(expected[name]["version"]):
            fail(f"version mismatch for {name}: {version} != {expected[name]['version']}")
        direct = dist / "direct_url.json"
        if direct.exists():
            origin = json.loads(direct.read_text())
            if "dir_info" in origin or "vcs_info" in origin or urllib.parse.urlsplit(origin.get("url", "")).scheme != "https":
                fail(f"local or editable install: {name}")
        record = dist / "RECORD"
        rows = list(csv.reader(io.StringIO(record.read_text())))
        self_seen = False
        for row in rows:
            if len(row) != 3:
                fail(f"invalid RECORD for {name}")
            value, hash_value, size = row
            path = safe_record_path(root, site, value)
            rel = path.relative_to(root).as_posix()
            if rel in owned or rel in baseline:
                fail(f"overlapping installed file: {rel}")
            owned.add(rel)
            if path == record:
                if hash_value or size:
                    fail(f"RECORD must not hash itself: {name}")
                self_seen = True
                continue
            if path.suffix == ".pyc" and path.parent.name == "__pycache__":
                continue
            algorithm, sep, encoded = hash_value.partition("=")
            if not sep or algorithm not in ("sha256", "sha384", "sha512"):
                fail(f"missing or unsupported RECORD hash: {value}")
            h = hashlib.new(algorithm)
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            actual = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()
            if actual != encoded or str(path.stat().st_size) != size:
                fail(f"modified installed file (RECORD mismatch): {value}")
        if not self_seen:
            fail(f"RECORD is not self-listed: {name}")
    missing = sorted(set(expected) - set(installed))
    if missing:
        fail("missing package: " + ", ".join(missing))
    current = inventory(root)
    for rel, evidence in baseline.items():
        if current.get(rel) != evidence:
            fail(f"modified venv baseline file: {rel}")
    extra = set(current) - owned - set(baseline)
    if extra:
        fail("unattested extra file: " + sorted(extra)[0])
    return current


def download(wheel, destination):
    with urllib.request.urlopen(wheel["url"], timeout=60) as response:
        if urllib.parse.urlsplit(response.geturl()).scheme != "https":
            fail("wheel download redirected away from HTTPS")
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    if digest(destination) != wheel["hashes"]["sha256"]:
        fail(f"wheel hash mismatch: {destination.name}")
    if "size" in wheel and destination.stat().st_size != wheel["size"]:
        fail(f"wheel size mismatch: {destination.name}")


def check_wheel(path, baseline):
    with zipfile.ZipFile(path) as archive:
        seen = set()
        for entry in archive.infolist():
            name = entry.filename
            parts = PurePosixPath(name).parts
            if name in seen or not parts or name.startswith("/") or ".." in parts or "\\" in name or stat.S_ISLNK(entry.external_attr >> 16):
                fail(f"unsafe wheel member: {name}")
            seen.add(name)
            shared_data = len(parts) >= 3 and parts[1:3] == ("data", "share")
            if parts[0].endswith(".data") and not shared_data and (len(parts) < 2 or parts[1] not in ("purelib", "platlib", "scripts")):
                fail(f"unsupported wheel data scheme: {name}")
            if name.endswith(".dist-info/entry_points.txt"):
                config = configparser.ConfigParser(interpolation=None)
                config.optionxform = str
                config.read_string(archive.read(entry).decode("utf-8"))
                for group in ("console_scripts", "gui_scripts"):
                    for script in config[group] if group in config else []:
                        if not re.fullmatch(r"[A-Za-z0-9_.-]+", script) or script in (".", "..") or "bin/" + script in baseline:
                            fail(f"unsafe wheel entry point: {script}")
            if parts[0].endswith(".data") and shared_data:
                if "/".join(parts[2:]) in baseline:
                    fail(f"wheel overwrites venv baseline: {name}")
            # pip maps .data/scripts into bin. Protect the venv's interpreter
            # and activation files before pip can overwrite them.
            if len(parts) > 2 and parts[0].endswith(".data") and parts[1] == "scripts":
                if "bin/" + "/".join(parts[2:]) in baseline:
                    fail(f"wheel overwrites venv baseline: {name}")


def run(args):
    completed = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env={"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/"), "LANG": "C.UTF-8", "PIP_CONFIG_FILE": os.devnull})
    if completed.returncode:
        fail(f"environment command failed ({args[0]}): {completed.stderr[-8192:]}")


def relocate(root, final):
    # venv/pip embed absolute paths. Rewrite only these paths and update the
    # corresponding RECORD entries; all other installed hashes remain intact.
    changed = set()
    old, new = os.fsencode(root), os.fsencode(final)
    for path in list((root / "bin").iterdir()) + [root / "pyvenv.cfg"]:
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if old in data:
            if b"\x00" in data:
                fail(f"cannot relocate binary script: {path.name}")
            path.write_bytes(data.replace(old, new))
            changed.add(path)
    site = site_path(root)
    for record in site.glob("*.dist-info/RECORD"):
        rows = list(csv.reader(io.StringIO(record.read_text())))
        for row in rows:
            path = safe_record_path(root, site, row[0])
            if path in changed:
                row[1] = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(digest(path))).rstrip(b"=").decode()
                row[2] = str(path.stat().st_size)
        with record.open("w", newline="") as f:
            csv.writer(f).writerows(rows)


def freeze(root):
    for base, dirs, files in os.walk(root):
        for name in files:
            path = Path(base, name)
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        Path(base).chmod(0o555)


def cleanup(root):
    if not root.exists():
        return
    for base, dirs, files in os.walk(root):
        Path(base).chmod(0o700)
    shutil.rmtree(root)


def verify(root, identity, expected):
    if root.is_symlink() or not root.is_dir():
        fail("environment cache entry missing or symlinked")
    receipt_path = root / RECEIPT
    if receipt_path.is_symlink():
        fail("symlinked environment receipt")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("kind") != "waldo-python-environment" or receipt.get("schema") != 1:
        fail("unsupported environment receipt")
    # Different manifest names may share the same physical environment.
    facts = {k: v for k, v in identity.items() if k != "name"}
    if receipt["identity"] != facts:
        fail("environment identity mismatch")
    current = verify_packages(root, expected, receipt["baseline"])
    if current != receipt["files"]:
        fail("environment files differ from verified installation receipt")
    # Never start the target interpreter until its paths and bytes are checked.
    # -S prevents execution of installed .pth and sitecustomize code.
    python = root / "bin/python"
    if digest(python.resolve(strict=True)) != identity["interpreter_sha256"]:
        fail("wrong interpreter: digest mismatch")
    completed = subprocess.run([str(python), "-I", "-S", "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"], capture_output=True, text=True, check=True)
    if completed.stdout.strip() != identity["interpreter_version"]:
        fail("wrong interpreter: version mismatch")
    return {"identity": identity, "run": {"cache_path": str(root), "installer": "pip", "installer_version": receipt["installer_version"]}}


def ensure(request, action="ensure"):
    if os.name != "posix":
        fail("python-lock PoC requires a POSIX host")
    version = ".".join(map(str, sys.version_info[:3]))
    if version != request["python_version"]:
        fail(f"wrong interpreter: expected {request['python_version']}, got {version}")
    if Version(pip_version) < Version("22.3"):
        fail("python-lock requires host pip >=22.3")
    lock_path = Path(request["lock"])
    if not re.fullmatch(r"pylock(?:\.[^.]+)?\.toml", lock_path.name):
        fail("PEP 751 lock must be named pylock.toml or pylock.<name>.toml")
    data = lock_path.read_bytes()
    expected = parse_lock(data)
    identity = {"name": request["name"], "kind": "python-lock", "lock_sha256": hashlib.sha256(data).hexdigest(),
                "interpreter_version": version, "interpreter_sha256": digest(Path(sys.executable).resolve()), "platform_tag": str(next(sys_tags()))}
    key = hashlib.sha256(json.dumps([identity[k] for k in ("lock_sha256", "interpreter_sha256", "platform_tag")], separators=(",", ":")).encode()).hexdigest()
    cache = Path(request["cache"]).resolve()
    root = cache / key
    if root.exists() or root.is_symlink() or action == "verify":
        return verify(root, identity, expected)
    cache.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".python-env-", dir=cache))
    env = Path(tempfile.mkdtemp(prefix=".python-venv-", dir=cache))
    try:
        venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(env)
        baseline = inventory(env)
        wheelhouse = temp / "wheels"
        wheelhouse.mkdir()
        requirements = []
        for name, package in sorted(expected.items()):
            wheel = package["wheel"]
            destination = wheelhouse / package["filename"]
            download(wheel, destination)
            check_wheel(destination, baseline)
            requirements.append(f"{name}=={package['version']} --hash=sha256:{wheel['hashes']['sha256']}")
        if requirements:
            requirements_path = temp / "requirements.txt"
            requirements_path.write_text("\n".join(requirements) + "\n")
            run([sys.executable, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check", "--python", str(env / "bin/python"),
                 "install", "--no-index", "--no-cache-dir", "--no-compile", "--no-deps", "--only-binary=:all:", "--require-hashes",
                 "--find-links", str(wheelhouse), "-r", str(requirements_path)])
        verify_packages(env, expected, baseline)
        relocate(env, root)
        current = inventory(env)
        baseline = {key: current[key] for key in baseline}
        files = verify_packages(env, expected, baseline)
        receipt = {"kind": "waldo-python-environment", "schema": 1, "identity": {k: v for k, v in identity.items() if k != "name"},
                   "installer_version": pip_version, "baseline": baseline, "files": files}
        (env / RECEIPT).write_text(json.dumps(receipt, sort_keys=True) + "\n")
        verify(env, identity, expected)
        freeze(env)
        try:
            env.rename(root)
        except OSError:
            if not root.exists():
                raise
            # Another creator won. Never replace or repair its entry.
        return verify(root, identity, expected)
    finally:
        cleanup(temp)
        cleanup(env)


if __name__ == "__main__":
    try:
        print(json.dumps(ensure(json.load(sys.stdin), sys.argv[1])))
    except (Exception, KeyboardInterrupt) as error:
        print(f"python-lock: {error}", file=sys.stderr)
        sys.exit(1)
