#!/usr/bin/env python3
"""
Install every core/user plugin's declared dependencies from its `pyproject.toml`
(`requirements.txt` is not supported: plugins install exclusively through
pyproject.toml + uv).

Each plugin's pyproject.toml is compiled into a fresh `uv.lock` — replacing any
existing one — constrained against the root project's own `uv.lock`, so uv
refuses to resolve a dangerous upgrade or downgrade of a system library the
core app depends on, and against the `uv.lock` of the plugins compiled before it
in this run, so it never replaces a library another plugin installed. Dependencies are then installed strictly from that lock
into the active virtual environment, without removing packages unrelated to
this plugin (`--inexact`).

Used at image build time (Dockerfile) and by `make install`; mirrors the
runtime install logic in `cat.looking_glass.mad_hatter.plugin.Plugin`.

Usage:
    python3 scripts/install_plugin_requirements.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib

PLUGIN_GLOBS = [
    "cat/core_plugins/*/pyproject.toml",
    "cat/plugins/*/pyproject.toml",
]
ROOT_LOCK = "uv.lock"
VENV_DIR = os.path.join(os.getcwd(), ".venv")


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


def lock_pins(lock_file: str, pinned: set) -> list:
    """Pins of the libraries in a plugin's uv.lock, except the ones in *pinned*
    (mirrors Plugin._other_plugins_constraints)."""
    with open(lock_file, "rb") as f:
        packages = tomllib.load(f).get("package", [])
    return [
        f"{pkg['name']}=={pkg['version']}"
        for pkg in packages
        if "version" in pkg
        and not {"virtual", "editable"} & set(pkg.get("source", {}))
        and pkg["name"] not in pinned
    ]


def compile_and_install(pyproject_file: str, constraints: list) -> None:
    plugin_dir = os.path.dirname(pyproject_file)
    with open(pyproject_file, "r") as f:
        pyproject_text = f.read()

    constraint_block = (
        "\n[tool.uv]\nconstraint-dependencies = [\n"
        + "".join(f'    "{c}",\n' for c in constraints)
        + "]\n"
    )
    env = {**os.environ, "VIRTUAL_ENV": VENV_DIR}

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "pyproject.toml"), "w") as f:
            f.write(pyproject_text + constraint_block)

        subprocess.check_call(["uv", "lock", "--project", tmp_dir, "--no-cache"], env=env)
        subprocess.check_call(
            [
                "uv", "sync", "--project", tmp_dir,
                "--no-install-project", "--active", "--frozen", "--inexact",
                "--link-mode=copy", "--no-cache",
            ],
            env=env,
        )

        # Persist the compiled lock next to the plugin's pyproject.toml,
        # replacing any existing one
        shutil.copyfile(os.path.join(tmp_dir, "uv.lock"), os.path.join(plugin_dir, "uv.lock"))


constraints = root_lock_constraints()
found = 0
for pattern in PLUGIN_GLOBS:
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            data = tomllib.load(f)
        if not data.get("project", {}).get("dependencies"):
            continue
        print(f"Compiling and installing dependencies for {path}", file=sys.stderr)
        compile_and_install(path, constraints)
        # the next plugins can never replace what this one installed (only the locks of this run: a stale uv.lock
        # left in the folder may come from another root uv.lock)
        constraints = constraints + lock_pins(
            os.path.join(os.path.dirname(path), "uv.lock"), {c.split("==")[0] for c in constraints},
        )
        found += 1

if found == 0:
    print("  (no plugin pyproject.toml with dependencies found)", file=sys.stderr)
