#!/usr/bin/env python3
"""
Write `.requirements_hash` files next to each plugin's `requirements.txt`.

The hash is the SHA-256 digest of the requirements file content.
At runtime `_install_requirements` compares this hash to skip redundant
`uv pip install` calls across uvicorn workers.

Usage:
    python3 scripts/hash_plugin_requirements.py
"""
import hashlib
import os
import glob
import sys

PLUGIN_GLOBS = [
    "cat/core_plugins/*/requirements.txt",
    "cat/plugins/*/requirements.txt",
]

hashed = 0
for pattern in PLUGIN_GLOBS:
    for req in glob.glob(pattern):
        h = hashlib.sha256(open(req, "rb").read()).hexdigest()
        hash_file = os.path.join(os.path.dirname(req), ".requirements_hash")
        with open(hash_file, "w") as f:
            f.write(h)
        print(f"  {req} -> {h[:12]}...", file=sys.stderr)
        hashed += 1

if hashed == 0:
    print("  (no plugin requirements.txt found)", file=sys.stderr)