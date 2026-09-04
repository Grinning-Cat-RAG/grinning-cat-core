#!/usr/bin/env python3
"""
Write `.requirements_hash` files next to each plugin's `pyproject.toml`.

The hash mirrors what `_install_requirements` computes at runtime: a digest
of the plugin's pyproject.toml plus the root project's uv.lock. Run this
right after `install_plugin_requirements.py` at image build time so a freshly
built image takes the fast path on first activation, instead of redundantly
recompiling and reinstalling what the build already did.

Usage:
    python3 scripts/hash_plugin_requirements.py
"""
import hashlib
import os
import glob
import sys
import tomllib

PLUGIN_GLOBS = [
    "cat/core_plugins/*/pyproject.toml",
    "cat/plugins/*/pyproject.toml",
]
ROOT_LOCK = "uv.lock"

def hash_files(paths):
    h = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


hashed = 0
for pattern in PLUGIN_GLOBS:
    for pyproject_file in sorted(glob.glob(pattern)):
        with open(pyproject_file, "rb") as f:
            data = tomllib.load(f)
        if not data.get("project", {}).get("dependencies"):
            continue

        inputs = [pyproject_file]
        if os.path.exists(ROOT_LOCK):
            inputs.append(ROOT_LOCK)
        digest = hash_files(inputs)

        hash_file = os.path.join(os.path.dirname(pyproject_file), ".requirements_hash")
        with open(hash_file, "w") as f:
            f.write(digest)
        print(f"  {pyproject_file} -> {digest[:12]}...", file=sys.stderr)
        hashed += 1

if hashed == 0:
    print("  (no plugin pyproject.toml with dependencies found)", file=sys.stderr)
