"""Plugin filesystem safety: concurrent installs/uninstalls must not leave a partial target, and the staging/trash
folders of the atomic install must never be discovered as plugins.

The filesystem helpers are exercised in real child processes (no Redis needed). The lock wiring of
``MadHatter.install_plugin``/``uninstall_plugin`` is tested separately with a fake context manager, so this module
runs without a Redis instance.
"""
import multiprocessing
import os

import pytest

from cat.services.atomic_fs import atomic_replace_dir, trash_then_delete


def _install_worker(tmp_root: str, payload: str, queue) -> None:
    try:
        tmp_dir = os.path.join(tmp_root, f".tmp-{payload}")
        os.makedirs(tmp_dir, exist_ok=True)
        with open(os.path.join(tmp_dir, "plugin.py"), "w") as f:
            f.write(payload)
        atomic_replace_dir(tmp_dir, os.path.join(tmp_root, "my_plugin"), os.path.join(tmp_root, ".trash"))
        queue.put(("ok", payload))
    except BaseException as e:  # noqa: BLE001 - the test asserts that nothing escapes
        queue.put(("error", repr(e)))


def _uninstall_worker(tmp_root: str, queue) -> None:
    try:
        trash_then_delete(os.path.join(tmp_root, "my_plugin"), os.path.join(tmp_root, ".trash"))
        queue.put(("ok", None))
    except BaseException as e:  # noqa: BLE001 - the test asserts that nothing escapes
        queue.put(("error", repr(e)))


def _run_two(target, args: tuple):
    methods = multiprocessing.get_all_start_methods()
    ctx = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=target, args=(*args, queue)) for _ in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(60)

    results = []
    for _ in procs:
        try:
            results.append(queue.get(timeout=10))
        except Exception:  # noqa: BLE001 - a dead child never writes to the queue
            results.append(("missing", None))

    return procs, results


def test_two_concurrent_installs_leave_one_complete_plugin(tmp_path):
    procs, results = _run_two(_install_worker, (str(tmp_path), "a"))

    assert [p.exitcode for p in procs] == [0, 0]
    assert [r[0] for r in results] == ["ok", "ok"]

    target = tmp_path / "my_plugin"
    assert target.is_dir()
    # the winner's payload is complete: never a half-written folder
    assert (target / "plugin.py").read_text() in {"a", "b"}
    assert sorted(p.name for p in target.iterdir()) == ["plugin.py"]
    # no temporary extraction folder is left behind
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")] == []


def test_two_concurrent_uninstalls_do_not_raise(tmp_path):
    target = tmp_path / "my_plugin"
    target.mkdir()
    (target / "plugin.py").write_text("x")

    procs, results = _run_two(_uninstall_worker, (str(tmp_path),))

    assert [p.exitcode for p in procs] == [0, 0]
    assert [r[0] for r in results] == ["ok", "ok"]
    assert not target.exists()


def test_an_install_and_an_uninstall_do_not_leave_a_partial_target(tmp_path):
    target = tmp_path / "my_plugin"
    target.mkdir()
    (target / "plugin.py").write_text("old")

    methods = multiprocessing.get_all_start_methods()
    ctx = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_install_worker, args=(str(tmp_path), "new", queue)),
        ctx.Process(target=_uninstall_worker, args=(str(tmp_path), queue)),
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(60)

    assert [p.exitcode for p in procs] == [0, 0]
    assert [queue.get(timeout=10)[0] for _ in procs] == ["ok", "ok"]
    # either the new version is fully there, or the folder is gone: never a partial one
    if target.exists():
        assert (target / "plugin.py").read_text() == "new"


def test_discovery_ignores_the_staging_and_trash_folders(tmp_path, monkeypatch):
    """``.tmp``/``.trash`` live inside the plugins folder: they must never become plugin ids."""
    from cat import utils
    from cat.looking_glass.mad_hatter.mad_hatter import MadHatter

    plugins = tmp_path / "plugins"
    core_plugins = tmp_path / "core_plugins"
    for name in ("real_plugin", ".tmp", ".trash", "__pycache__", "lost+found", ".git"):
        (plugins / name).mkdir(parents=True)
    (core_plugins / "base_plugin").mkdir(parents=True)

    monkeypatch.setattr(utils, "get_plugins_path", lambda: str(plugins))
    monkeypatch.setattr(utils, "get_core_plugins_path", lambda: str(core_plugins))

    mh = MadHatter("agent")

    assert mh.load_active_plugins_ids_from_folders() == ["base_plugin", "real_plugin"]
    assert mh.plugin_exists(".tmp") is False
    assert mh.plugin_exists(".trash") is False
    # the skip list is the authoritative exclusion: keep it explicit even if the listing mechanism changes
    assert {".tmp", ".trash", ".git"} <= set(mh._skip_folders)


class _FakeLock:
    """Records the lock keys and behaves like ``crud.distributed_lock`` (an async context manager)."""

    def __init__(self, keys: list):
        self._keys = keys
        self.held = 0

    def __call__(self, key_pattern: str, timeout: float = 10.0, blocking_timeout: float = 15.0):
        self._keys.append((key_pattern, timeout, blocking_timeout))
        return self

    async def __aenter__(self):
        self.held += 1
        return self

    async def __aexit__(self, *exc_info):
        self.held -= 1
        return False


async def test_install_plugin_takes_the_plugin_fs_lock(tmp_path, monkeypatch):
    from cat.looking_glass.mad_hatter import mad_hatter as mh_module

    keys = []
    lock = _FakeLock(keys)
    monkeypatch.setattr(mh_module, "distributed_lock", lock)

    class FakeExtractor:
        def __init__(self, path):
            self.id = "my_plugin"

        def extract(self, to):
            return os.path.join(to, "my_plugin")

    monkeypatch.setattr(mh_module, "PluginExtractor", FakeExtractor)

    mh = mh_module.MadHatter("agent")
    installed = []

    async def fake_install_extracted(plugin_id, persist=True):
        # the activation touches the database: it must not run while the filesystem lock is held
        assert lock.held == 0, "activation must run outside the plugin-fs lock"
        installed.append(plugin_id)
        return plugin_id

    async def fake_load_plugin(plugin_id, with_deactivation=True):
        # `load_plugin` may run `uv sync` for minutes: it must not run while the filesystem lock is held
        assert lock.held == 0, "load_plugin must run outside the plugin-fs lock"
        return mh_module.LoadedPlugin()

    monkeypatch.setattr(mh, "install_extracted_plugin", fake_install_extracted)
    monkeypatch.setattr(mh, "load_plugin", fake_load_plugin)

    assert await mh.install_plugin("pkg.zip") == "my_plugin"
    assert keys == [("plugin-fs:my_plugin", 300, 60)]
    assert installed == ["my_plugin"]


async def test_install_plugin_cleans_up_missing_deps_under_the_lock(tmp_path, monkeypatch):
    from cat.looking_glass.mad_hatter import mad_hatter as mh_module

    keys = []
    lock = _FakeLock(keys)
    monkeypatch.setattr(mh_module, "distributed_lock", lock)

    plugins = tmp_path / "plugins"
    plugin_dir = plugins / "my_plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text("x")

    class FakeExtractor:
        def __init__(self, path):
            self.id = "my_plugin"

        def extract(self, to):
            return str(plugin_dir)

    monkeypatch.setattr(mh_module, "PluginExtractor", FakeExtractor)
    monkeypatch.setattr(mh_module.utils, "get_plugins_path", lambda: str(plugins))

    mh = mh_module.MadHatter("agent")

    async def fake_load_plugin(plugin_id, with_deactivation=True):
        assert lock.held == 0, "load_plugin must run outside the plugin-fs lock"
        return mh_module.LoadedPlugin(missing_dependencies=["nope"])

    monkeypatch.setattr(mh, "load_plugin", fake_load_plugin)

    with pytest.raises(Exception, match="missing dependencies"):
        await mh.install_plugin("pkg.zip")

    # one acquisition for the extraction, one for the cleanup
    assert keys == [("plugin-fs:my_plugin", 300, 60), ("plugin-fs:my_plugin", 300, 60)]
    assert not plugin_dir.exists()


async def test_uninstall_plugin_takes_the_plugin_fs_lock(tmp_path, monkeypatch):
    from cat.looking_glass.mad_hatter import mad_hatter as mh_module

    keys = []
    monkeypatch.setattr(mh_module, "distributed_lock", _FakeLock(keys))
    monkeypatch.setattr(mh_module.MadHatter, "get_core_plugins_ids", property(lambda self: []))

    plugin_dir = tmp_path / "plugins" / "my_plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text("x")

    destroyed = []

    async def fake_destroy(plugin_id):
        destroyed.append(plugin_id)

    monkeypatch.setattr(mh_module.crud_plugins, "destroy_plugin", fake_destroy)

    mh = mh_module.MadHatter("agent")
    monkeypatch.setattr(mh, "plugin_exists", lambda plugin_id: True)
    monkeypatch.setattr(mh, "_get_plugins_depending_on", lambda plugin_id: [])
    monkeypatch.setattr(mh, "_get_plugin_folder_path", lambda plugin_id: str(plugin_dir))

    await mh.uninstall_plugin("my_plugin")

    assert keys == [("plugin-fs:my_plugin", 60, 30)]
    assert not plugin_dir.exists()
    assert destroyed == ["my_plugin"]


async def test_uninstall_of_an_already_missing_plugin_only_forgets_it(tmp_path, monkeypatch):
    from cat.looking_glass.mad_hatter import mad_hatter as mh_module

    keys = []
    monkeypatch.setattr(mh_module, "distributed_lock", _FakeLock(keys))
    monkeypatch.setattr(mh_module.MadHatter, "get_core_plugins_ids", property(lambda self: []))

    mh = mh_module.MadHatter("agent")
    forgotten = []

    async def fake_forget(plugin_id):
        forgotten.append(plugin_id)

    monkeypatch.setattr(mh, "plugin_exists", lambda plugin_id: False)
    monkeypatch.setattr(mh, "_forget_plugin", fake_forget)

    await mh.uninstall_plugin("my_plugin")

    assert keys == [("plugin-fs:my_plugin", 60, 30)]
    assert forgotten == ["my_plugin"]
