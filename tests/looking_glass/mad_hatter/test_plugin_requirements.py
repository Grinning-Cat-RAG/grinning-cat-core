"""How a plugin's dependencies are compiled and installed (no uv run: its commands are recorded).

- The plugins share one virtual environment: a plugin is compiled against the root uv.lock AND the uv.lock of the other
  installed plugins, so it can never replace the version of a library another plugin installed.
- After an installation only the plugin's own bytecode is cleaned, never the whole application (virtual environment,
  data).
- A plugin that ships its own uv.lock is installed strictly from it and the lock is never regenerated: a lock that does
  not match the pyproject.toml, or that would replace a version pinned by the root or by another plugin, is refused.
- Dependency groups (dev included) are neither resolved nor installed.
"""
import os
import sys
from pathlib import Path

import pytest
import tomli

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
    uv.lock, as the core writes it after an installation (its uv.lock, if any, compiled by the core)."""
    (folder / "pyproject.toml").write_text(f'[project]\nname = "{folder.name}"\nversion = "0.1.0"\ndependencies = ["x"]\n')
    if (folder / "uv.lock").exists():
        (folder / ".uv.lock.origin").write_text(f"generated:{Plugin._hash_files([str(folder / 'uv.lock')])}")
    (folder / ".requirements_hash").write_text(Plugin._hash_files(Plugin._requirements_hash_inputs(str(folder))))


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
    assert (plugins / "me" / ".uv.lock.origin").read_text().startswith("generated:")
    assert Plugin._shipped_lock(str(plugins / "me")) is None  # the lock the core compiled is not the plugin's own


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


# --------------------------------------------------------------------------------------------------------------------
# A plugin shipping its own uv.lock
# --------------------------------------------------------------------------------------------------------------------

PYPROJECT = """[project]
name = "me"
version = "0.1.0"
dependencies = [{deps}]

[dependency-groups]
dev = ["pytest==8.4.2"]
"""


def shipped_lock(folder: Path, deps: dict[str, str], extra_packages: str = "", skip: tuple[str, ...] = ()) -> None:
    """The plugin in *folder* ships a pyproject.toml and the uv.lock uv would write for it: *deps* are its dependencies
    (`name: version`, declared and locked as `==version`) and its dev group pins pytest 8.4.2. *extra_packages* is
    appended verbatim (more [[package]] entries); the packages in *skip* get no plain entry (*extra_packages* has one)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "hooks.py").write_text("")
    (folder / "pyproject.toml").write_text(PYPROJECT.format(deps=", ".join(f'"{n}=={v}"' for n, v in deps.items())))
    edges = "".join(f'    {{ name = "{n}" }},\n' for n in deps)
    requires = "".join(f'    {{ name = "{n}", specifier = "=={v}" }},\n' for n, v in deps.items())
    packages = "".join(
        f'[[package]]\nname = "{n}"\nversion = "{v}"\nsource = {{ registry = "https://pypi.org/simple" }}\n\n'
        for n, v in {**deps, "pytest": "8.4.2"}.items() if n not in skip
    )
    (folder / "uv.lock").write_text(
        f'version = 1\n\n[[package]]\nname = "me"\nversion = "0.1.0"\nsource = {{ virtual = "." }}\n'
        f'dependencies = [\n{edges}]\n\n[package.dev-dependencies]\ndev = [\n    {{ name = "pytest" }},\n]\n\n'
        f"[package.metadata]\nrequires-dist = [\n{requires}]\n\n"
        f'[package.metadata.requires-dev]\ndev = [{{ name = "pytest", specifier = "==8.4.2" }}]\n\n'
        + packages + extra_packages
    )


def recorded_commands(monkeypatch) -> list[list[str]]:
    commands = []

    def run(cmd, env):
        commands.append(cmd)
        if cmd[1] == "lock":
            (Path(cmd[cmd.index("--project") + 1]) / "uv.lock").write_text("version = 1\n")
        return 0

    monkeypatch.setattr(Plugin, "_run_streamed", staticmethod(run))
    return commands


def install(me: Plugin) -> None:
    pyproject = os.path.join(me.path, "pyproject.toml")
    me._install_requirements_sync(pyproject, Plugin._hash_files(Plugin._requirements_hash_inputs(me.path)))


def capture_errors(monkeypatch) -> list[str]:
    errors = []
    monkeypatch.setattr(plugin_module.log, "error", lambda msg: errors.append(msg))
    return errors


def test_a_shipped_lock_is_installed_as_is_and_never_regenerated(tmp_path, monkeypatch):
    root, core, plugins = layout(tmp_path, monkeypatch)
    lock(root, "numpy==2.0.0", "pytest==9.0.2")  # pytest: only the plugin's dev group pins another version
    shipped_lock(plugins / "me", {"numpy": "2.0.0", "pyfiglet": "1.0.2"})
    before = (plugins / "me" / "uv.lock").read_text()
    commands = recorded_commands(monkeypatch)
    me = Plugin(str(plugins / "me"))

    install(me)

    assert [c[:2] for c in commands] == [["uv", "sync"]]  # no `uv lock`
    assert {"--frozen", "--no-default-groups", "--no-install-project", "--inexact"} <= set(commands[0])
    assert (plugins / "me" / "uv.lock").read_text() == before
    assert (plugins / "me" / ".uv.lock.origin").read_text() == "shipped"
    assert Plugin._is_installed_here(str(plugins / "me"))


def test_a_shipped_lock_changing_a_pinned_version_is_refused(tmp_path, monkeypatch):
    root, core, plugins = layout(tmp_path, monkeypatch)
    lock(root, "numpy==2.0.0")
    lock(plugins / "other", "pyfiglet==1.0.2")
    installed(plugins / "other", root)
    shipped_lock(plugins / "me", {"numpy": "2.1.0", "pyfiglet": "0.8"})
    before = (plugins / "me" / "uv.lock").read_text()
    errors = capture_errors(monkeypatch)
    commands = recorded_commands(monkeypatch)

    with pytest.raises(Exception, match="Error while installing plugin me requirements"):
        install(Plugin(str(plugins / "me")))

    assert commands == []  # nothing installed, nothing regenerated
    assert (plugins / "me" / "uv.lock").read_text() == before
    assert "numpy==2.1.0, but numpy==2.0.0 is pinned" in errors[0]
    assert "pyfiglet==0.8, but pyfiglet==1.0.2 is pinned" in errors[0]
    assert not (plugins / "me" / ".requirements_hash").exists()


def test_a_shipped_lock_out_of_date_with_the_pyproject_is_refused(tmp_path, monkeypatch):
    root, core, plugins = layout(tmp_path, monkeypatch)
    shipped_lock(plugins / "me", {"numpy": "2.0.0"})
    pyproject = plugins / "me" / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('"numpy==2.0.0"', '"numpy==2.0.0", "six==1.16.0"'))
    errors = capture_errors(monkeypatch)
    commands = recorded_commands(monkeypatch)

    with pytest.raises(Exception):
        install(Plugin(str(plugins / "me")))

    assert commands == []
    assert "declared in pyproject.toml, not locked: six==1.16.0" in errors[0]


def test_markers_are_compared_by_their_outcome_not_their_text(tmp_path, monkeypatch):
    """uv writes `python_version` markers as `python_full_version`: that is not a stale lock."""
    root, core, plugins = layout(tmp_path, monkeypatch)
    shipped_lock(plugins / "me", {"numpy": "2.0.0"})
    pyproject = plugins / "me" / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('"numpy==2.0.0"', "\"numpy==2.0.0; python_version >= '3.8'\""))
    me_lock = plugins / "me" / "uv.lock"
    me_lock.write_text(me_lock.read_text().replace(
        '{ name = "numpy", specifier = "==2.0.0" }',
        '{ name = "numpy", marker = "python_full_version >= \'3.8\'", specifier = "==2.0.0" }',
    ))
    commands = recorded_commands(monkeypatch)

    install(Plugin(str(plugins / "me")))

    assert [c[:2] for c in commands] == [["uv", "sync"]]


def test_only_what_the_lock_installs_here_is_checked(tmp_path, monkeypatch):
    """A package behind a marker that does not hold here, and the dev group, are not installed: they cannot conflict."""
    root, core, plugins = layout(tmp_path, monkeypatch)
    lock(root, "colorama==0.4.6", "pytest==9.0.2")
    pyfiglet_with_edge = (
        '[[package]]\nname = "pyfiglet"\nversion = "1.0.2"\nsource = { registry = "https://pypi.org/simple" }\n'
        "dependencies = [{ name = \"colorama\", marker = \"sys_platform == 'nowhere'\" }]\n\n"
        '[[package]]\nname = "colorama"\nversion = "0.3.0"\nsource = { registry = "https://pypi.org/simple" }\n'
    )
    shipped_lock(plugins / "me", {"pyfiglet": "1.0.2"}, extra_packages=pyfiglet_with_edge, skip=("pyfiglet",))
    commands = recorded_commands(monkeypatch)

    install(Plugin(str(plugins / "me")))

    assert [c[:2] for c in commands] == [["uv", "sync"]]
    lock_data = tomli.loads((plugins / "me" / "uv.lock").read_text())
    assert Plugin._locked_runtime_packages(lock_data) == {"pyfiglet": {"1.0.2"}}


def test_transitive_dependencies_and_extras_are_checked(tmp_path, monkeypatch):
    root, core, plugins = layout(tmp_path, monkeypatch)
    lock(root, "colorama==0.4.6", "six==1.16.0")
    pyfiglet_with_extra = (
        '[[package]]\nname = "pyfiglet"\nversion = "1.0.2"\nsource = { registry = "https://pypi.org/simple" }\n'
        'dependencies = [{ name = "colorama" }]\n\n'
        '[package.optional-dependencies]\nfancy = [{ name = "six" }]\n\n'
        '[[package]]\nname = "colorama"\nversion = "0.3.0"\nsource = { registry = "https://pypi.org/simple" }\n\n'
        '[[package]]\nname = "six"\nversion = "1.0.0"\nsource = { registry = "https://pypi.org/simple" }\n'
    )
    shipped_lock(plugins / "me", {"pyfiglet": "1.0.2"}, extra_packages=pyfiglet_with_extra, skip=("pyfiglet",))
    me_lock = plugins / "me" / "uv.lock"
    me_lock.write_text(me_lock.read_text().replace('    { name = "pyfiglet" },', '    { name = "pyfiglet", extra = ["fancy"] },'))
    errors = capture_errors(monkeypatch)
    recorded_commands(monkeypatch)

    with pytest.raises(Exception):
        install(Plugin(str(plugins / "me")))

    assert "colorama==0.3.0, but colorama==0.4.6 is pinned" in errors[0]
    assert "six==1.0.0, but six==1.16.0 is pinned" in errors[0]


def test_which_uv_lock_is_the_plugins_own(tmp_path):
    folder = tmp_path / "me"
    folder.mkdir()
    assert Plugin._shipped_lock(str(folder)) is None  # no lock at all
    lock(folder, "numpy==2.0.0")
    assert Plugin._shipped_lock(str(folder)) == str(folder / "uv.lock")  # never installed here: shipped
    (folder / ".requirements_hash").write_text("x")
    assert Plugin._shipped_lock(str(folder)) is None  # installed by a core older than the record: it compiled the lock
    (folder / ".uv.lock.origin").write_text(f"generated:{Plugin._hash_files([str(folder / 'uv.lock')])}")
    assert Plugin._shipped_lock(str(folder)) is None  # compiled by the core
    lock(folder, "numpy==2.1.0")
    assert Plugin._shipped_lock(str(folder)) == str(folder / "uv.lock")  # replaced since: a new version of the plugin
    (folder / ".uv.lock.origin").write_text("shipped")
    assert Plugin._shipped_lock(str(folder)) == str(folder / "uv.lock")


def test_a_change_to_the_shipped_lock_installs_again(tmp_path, monkeypatch):
    root, core, plugins = layout(tmp_path, monkeypatch)
    shipped_lock(plugins / "me", {"numpy": "2.0.0"})
    recorded_commands(monkeypatch)
    install(Plugin(str(plugins / "me")))
    assert Plugin._is_installed_here(str(plugins / "me"))

    me_lock = plugins / "me" / "uv.lock"
    me_lock.write_text(me_lock.read_text() + "\n# a new version of the lock\n")

    assert not Plugin._is_installed_here(str(plugins / "me"))


def test_dependency_groups_are_not_resolved_when_compiling(tmp_path, monkeypatch):
    root, core, plugins = layout(tmp_path, monkeypatch)
    lock(root, "pytest==9.0.2")
    me = a_plugin(plugins / "me", ["pyfiglet"])
    pyproject = plugins / "me" / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text()
        + '\n[dependency-groups]\ndev = ["pytest==8.4.2"]\n\n[tool.uv]\nconstraint-dependencies = ["six<2"]\n'
    )
    compiled = {}

    def run(cmd, env):
        project = Path(cmd[cmd.index("--project") + 1])
        if cmd[1] == "lock":
            compiled["pyproject"] = tomli.loads((project / "pyproject.toml").read_text())
            (project / "uv.lock").write_text("version = 1\n")
        else:
            compiled["sync"] = cmd
        return 0

    monkeypatch.setattr(Plugin, "_run_streamed", staticmethod(run))
    install(me)

    assert "dependency-groups" not in compiled["pyproject"]
    # the plugin's own [tool.uv] is kept, and the core's pins are added to its constraints
    constraints = compiled["pyproject"]["tool"]["uv"]["constraint-dependencies"]
    assert constraints[0] == "six<2" and "pytest==9.0.2" in constraints
    assert "--no-default-groups" in compiled["sync"]


def test_a_plugin_installed_by_an_older_core_is_left_as_it_is(tmp_path, monkeypatch):
    """An older core always replaced the plugin's uv.lock with the one it compiled and hashed only the pyproject.toml and
    the root uv.lock: such a plugin is still installed (no reinstall), and its uv.lock is still the core's, so a new root
    uv.lock compiles it again instead of refusing it."""
    root, core, plugins = layout(tmp_path, monkeypatch)
    lock(root, "numpy==2.0.0")
    me = a_plugin(plugins / "me", ["numpy"])
    lock(plugins / "me", "numpy==2.0.0")
    old_hash = Plugin._hash_files([str(plugins / "me" / "pyproject.toml"), str(root / "uv.lock")])
    (plugins / "me" / ".requirements_hash").write_text(old_hash)

    assert Plugin._is_installed_here(str(plugins / "me"))
    assert Plugin._shipped_lock(str(plugins / "me")) is None

    lock(root, "numpy==2.1.0")
    commands = recorded_commands(monkeypatch)
    install(me)
    assert [c[:2] for c in commands] == [["uv", "lock"], ["uv", "sync"]]
