"""How a plugin's dependencies are compiled and installed (no uv run: its commands are recorded).

- The plugins share one virtual environment: a plugin is compiled against the root uv.lock AND the uv.lock of the other
  installed plugins, so it can never replace the version of a library another plugin installed.
- After an installation only the plugin's own bytecode is cleaned, never the whole application (virtual environment,
  data).
"""
import os
import sys
from pathlib import Path

from cat.looking_glass.mad_hatter import plugin as plugin_module
from cat.looking_glass.mad_hatter.plugin import Plugin


def lock(path: Path, *packages: str, raw: str | None = None) -> None:
    """A uv.lock with the given `name==version` packages (and the plugin project itself, a virtual source)."""
    path.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (path / "uv.lock").write_text(raw)
        return
    entries = [f'[[package]]\nname = "{path.name}-project"\nversion = "0"\nsource = {{ virtual = "." }}\n']
    for p in packages:
        name, version = p.split("==")
        entries.append(f'[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ registry = "https://pypi.org/simple" }}\n')
    (path / "uv.lock").write_text("version = 1\n\n" + "\n".join(entries))


def installed(folder: Path, root: Path) -> None:
    """The plugin in *folder* was installed in this environment: its record matches its pyproject.toml and the root
    uv.lock, as the core writes it after an installation."""
    (folder / "pyproject.toml").write_text(f'[project]\nname = "{folder.name}"\nversion = "0.1.0"\ndependencies = ["x"]\n')
    inputs = [str(folder / "pyproject.toml")] + ([str(root / "uv.lock")] if (root / "uv.lock").exists() else [])
    (folder / ".requirements_hash").write_text(Plugin._hash_files(inputs))


def a_plugin(folder: Path, dependencies: list[str] | None = None) -> Plugin:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "hooks.py").write_text("")
    if dependencies is not None:
        deps = ", ".join(f'"{d}"' for d in dependencies)
        (folder / "pyproject.toml").write_text(f'[project]\nname = "{folder.name}"\nversion = "0.1.0"\ndependencies = [{deps}]\n')
    return Plugin(str(folder))


def layout(tmp_path, monkeypatch):
    root, core, plugins = tmp_path / "app", tmp_path / "app" / "core_plugins", tmp_path / "app" / "plugins"
    root.mkdir()
    monkeypatch.setattr(plugin_module, "get_project_path", lambda: str(root))
    monkeypatch.setattr(plugin_module, "get_core_plugins_path", lambda: str(core))
    monkeypatch.setattr(plugin_module, "get_plugins_path", lambda: str(plugins))
    return root, core, plugins


def test_other_plugins_constraints(tmp_path, monkeypatch):
    root, core, plugins = layout(tmp_path, monkeypatch)
    lock(root, "numpy==2.0.0")
    me = a_plugin(plugins / "me")
    lock(core / "a", "numpy==2.0.0", "pyfiglet==1.0.2")
    lock(plugins / "b", "pyfiglet==0.8", "six==1.16.0")  # pyfiglet: the first lock wins
    lock(plugins / "me", "tinydb==4.8.2")  # its own lock: about to be replaced
    lock(plugins / "c", raw="not toml [")  # unreadable: ignored
    lock(plugins / "shipped", "openpyxl==3.0.0")  # shipped in an archive, never installed here: ignored
    lock(plugins / "stale", "lxml==4.0.0")  # installed against an older root uv.lock: ignored
    for folder in (core / "a", plugins / "b", plugins / "me", plugins / "c", plugins / "stale"):
        installed(folder, root)
    (plugins / "stale" / ".requirements_hash").write_text("an older hash")

    assert me._other_plugins_constraints(pinned={"numpy"}) == ["pyfiglet==1.0.2", "six==1.16.0"]


def test_a_plugin_is_compiled_against_the_root_and_the_other_plugins(tmp_path, monkeypatch):
    root, core, plugins = layout(tmp_path, monkeypatch)
    lock(root, "numpy==2.0.0")
    lock(plugins / "other", "pyfiglet==1.0.2")
    installed(plugins / "other", root)
    me = a_plugin(plugins / "me", ["pyfiglet<1.0"])
    (plugins / "me" / "__pycache__").mkdir()
    (root / "venv_package" / "__pycache__").mkdir(parents=True)  # e.g. the shared virtual environment

    compiled = {}

    def run(cmd, env):
        project = Path(cmd[cmd.index("--project") + 1])
        if cmd[1] == "lock":
            compiled["pyproject"] = (project / "pyproject.toml").read_text()
            (project / "uv.lock").write_text("version = 1\n")
        return 0

    monkeypatch.setattr(Plugin, "_run_streamed", staticmethod(run))
    me._install_requirements_sync(str(plugins / "me" / "pyproject.toml"), "hash")

    assert '"numpy==2.0.0"' in compiled["pyproject"]  # the root project's pins
    assert '"pyfiglet==1.0.2"' in compiled["pyproject"]  # the other plugin's: pyfiglet<1.0 cannot replace it
    assert (plugins / "me" / ".requirements_hash").read_text() == "hash"
    assert not (plugins / "me" / "__pycache__").exists()  # the plugin's own bytecode is cleaned
    assert (root / "venv_package" / "__pycache__").exists()  # nothing else


def test_a_failed_uv_command_logs_its_output(monkeypatch):
    errors = []
    monkeypatch.setattr(plugin_module.log, "error", lambda msg: errors.append(msg))
    code = Plugin._run_streamed(
        [sys.executable, "-c", "print('No solution found: pyfiglet==1.0.2 conflicts'); raise SystemExit(3)"], dict(os.environ),
    )
    assert code == 3
    assert len(errors) == 1 and "pyfiglet==1.0.2 conflicts" in errors[0]


def test_a_successful_command_logs_no_error(monkeypatch):
    errors = []
    monkeypatch.setattr(plugin_module.log, "error", lambda msg: errors.append(msg))
    assert Plugin._run_streamed([sys.executable, "-c", "print('ok')"], dict(os.environ)) == 0
    assert errors == []
