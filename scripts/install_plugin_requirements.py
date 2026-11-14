#!/usr/bin/env python3
"""
Install every plugin's declared dependencies from its `pyproject.toml`
(`requirements.txt` is not supported: plugins install exclusively through
pyproject.toml + uv).

All the plugin dependencies are resolved in ONE `uv.lock`, together with the
root project's dependencies and constrained against the root project's own
`uv.lock`. A single resolution is required: resolving each plugin on its own,
constrained by the plugins compiled before it, is order-dependent and fails
when two plugins need different versions of the same library, even though a
common set exists. The merged resolution finds that common set.

The dependencies are then installed strictly from that lock into the active
virtual environment, without removing packages unrelated to the plugins
(`--inexact`).

Used at image build time (Dockerfile) and by `make install`; the runtime
reinstall mirrors this logic only for newly installed plugins.

Usage:
    python3 scripts/install_plugin_requirements.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import tomllib

PLUGIN_GLOBS = [
    "cat/core_plugins/*/pyproject.toml",
    "cat/plugins/*/pyproject.toml",
]
ROOT_PYPROJECT = "pyproject.toml"
ROOT_LOCK = "uv.lock"


def root_lock_constraints():
    if not os.path.exists(ROOT_LOCK):
        return []
    with open(ROOT_LOCK, "rb") as f:
        data = tomllib.load(f)
    return [
        f"{pkg['name']}=={pkg['version']}"
        for pkg in data.get("package", [])
        if "version" in pkg
    ]


def plugin_dependencies():
    deps = []
    for pattern in PLUGIN_GLOBS:
        for path in sorted(glob.glob(pattern)):
            with open(path, "rb") as f:
                data = tomllib.load(f)
            deps.extend(data.get("project", {}).get("dependencies", []))
    # stable de-duplication: the same requirement declared twice is one constraint
    return list(dict.fromkeys(deps))


def root_requires_python():
    if not os.path.exists(ROOT_PYPROJECT):
        return ">=3.11,<3.14"
    with open(ROOT_PYPROJECT, "rb") as f:
        return tomllib.load(f).get("project", {}).get("requires-python", ">=3.11,<3.14")


def merged_pyproject(dependencies, constraints, requires_python):
    platform_marker = json.dumps(f"sys_platform == '{sys.platform}'")
    lines = [
        "[project]",
        'name = "unified-plugin-dependencies"',
        'version = "0.0.0"',
        f"requires-python = {json.dumps(requires_python)}",
        "dependencies = [",
    ]
    lines += [f"    {json.dumps(dep)}," for dep in dependencies]
    lines += [
        "]",
        "",
        "[tool.uv]",
        # Resolve only the current platform: a universal lock also spans platforms the
        # image never runs on (e.g. win32), where unrelated plugins' dependencies can
        # be unsatisfiable and abort the whole install.
        f"environments = [{platform_marker}]",
        "constraint-dependencies = [",
    ]
    lines += [f"    {json.dumps(c)}," for c in constraints]
    lines += ["]", ""]
    return "\n".join(lines)


def main():
    dependencies = plugin_dependencies()
    if not dependencies:
        print("  (no plugin pyproject.toml with dependencies found)", file=sys.stderr)
        return

    constraints = root_lock_constraints()
    text = merged_pyproject(dependencies, constraints, root_requires_python())
    env = {**os.environ, "VIRTUAL_ENV": os.path.join(os.getcwd(), ".venv")}

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "pyproject.toml"), "w") as f:
            f.write(text)

        print(
            f"Resolving {len(dependencies)} plugin dependencies in one lock "
            f"({len(constraints)} core constraints)",
            file=sys.stderr,
        )
        subprocess.check_call(["uv", "lock", "--project", tmp_dir, "--no-cache"], env=env)
        subprocess.check_call(
            [
                "uv", "sync", "--project", tmp_dir,
                "--no-install-project", "--active", "--frozen", "--inexact",
                "--link-mode=copy", "--no-cache",
            ],
            env=env,
        )


if __name__ == "__main__":
    main()
