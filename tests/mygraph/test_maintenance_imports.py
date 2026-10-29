"""MyGRAPH maintenance CLI against the extracted EffING plugin.

EffING moved to ``cat.core_plugins.cat_efficient_ingestion`` (``reembed`` and
``registry`` modules). ``maintenance_agent.py`` lives in the GraphRAG plugin
folder and is imported by the Cat plugin loader, so it must stay import-safe:
its EffING imports are lazy and guarded.

These tests load the real ``maintenance_agent.py`` from the GraphRAG plugin
checkout (sibling ``plugins/`` dir by default, overridable with
``MYGRAPH_MAINTENANCE_AGENT``) and assert:

1. with EffING importable the module imports and the symbols it needs resolve
   through the new module paths (and are the ones actually used by an op);
2. with EffING blocked the module still imports, the CLI ``--help`` works, and
   an op needing EffING aborts with a clear message and exit code 2 instead of
   raising a traceback.
"""

import asyncio
import importlib
import importlib.abc
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

EFFING = "cat.core_plugins.cat_efficient_ingestion"


def _maintenance_path() -> Path:
    override = os.environ.get("MYGRAPH_MAINTENANCE_AGENT")
    if override:
        return Path(override)
    core_root = Path(__file__).resolve().parents[2]
    return core_root.parent / "plugins" / "cat-graphrag.my" / "maintenance_agent.py"


def _load_maintenance() -> ModuleType:
    path = _maintenance_path()
    assert path.is_file(), f"maintenance_agent.py not found at {path}"
    spec = importlib.util.spec_from_file_location("mygraph_maintenance_agent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BlockEffing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == EFFING or fullname.startswith(EFFING + "."):
            raise ImportError(f"{fullname} blocked for test")
        return None


class _FakeCCat:
    agent_key = "agent-test"

    async def get_stored_sources_with_metadata(self):
        from cat.services.memory.models import VectorMemoryType

        return {VectorMemoryType("declarative"): []}


def test_module_imports_and_uses_new_effing_paths(monkeypatch):
    import cat.services.memory.models  # noqa: F401  (used by _op_reembed)

    calls: list[tuple[object, ...]] = []

    fake_reembed = ModuleType(f"{EFFING}.reembed")
    fake_registry = ModuleType(f"{EFFING}.registry")

    async def reembed_sources(ccat, memory_type, sources):
        calls.append((ccat, memory_type, sources))

    async def delete_status(*args, **kwargs):
        calls.append(("delete_status", args, kwargs))

    async def list_statuses(agent_id):
        calls.append(("list_statuses", agent_id))
        return []

    setattr(fake_reembed, "reembed_sources", reembed_sources)
    setattr(fake_registry, "delete_status", delete_status)
    setattr(fake_registry, "list_statuses", list_statuses)
    monkeypatch.setitem(sys.modules, f"{EFFING}.reembed", fake_reembed)
    monkeypatch.setitem(sys.modules, f"{EFFING}.registry", fake_registry)

    module = _load_maintenance()

    assert asyncio.run(module._op_reembed(_FakeCCat(), None, "declarative")) is True
    assert calls[0] == ("list_statuses", "agent-test")
    assert calls[-1][2] == []  # reembed_sources(ccat, VectorMemoryType, sources)

    source = _maintenance_path().read_text()
    assert "cat.core_plugins.efficient_ingestion" not in source
    assert "cat.core_plugins.ingestion_status" not in source
    assert EFFING in source


def test_real_effing_symbols_resolve_from_new_paths():
    reembed = importlib.import_module(f"{EFFING}.reembed")
    registry = importlib.import_module(f"{EFFING}.registry")
    assert callable(reembed.reembed_sources)
    assert callable(registry.delete_status)
    assert callable(registry.list_statuses)


@pytest.mark.parametrize("op_name", ["_op_reembed", "_op_reingest", "_verify_agent"])
def test_missing_effing_aborts_without_traceback(monkeypatch, capsys, op_name):
    module = _load_maintenance()  # import first: module itself stays import-safe

    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == EFFING or name.startswith(EFFING + ".")
    }
    blocker = _BlockEffing()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(SystemExit) as excinfo:
            if op_name == "_verify_agent":
                asyncio.run(module._verify_agent(_FakeCCat(), None, "declarative"))
            else:
                asyncio.run(getattr(module, op_name)(_FakeCCat(), None, "declarative"))
        assert excinfo.value.code == 2
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)

    err = capsys.readouterr().err
    assert "ABORTED" in err
    assert EFFING in err
    assert "Traceback" not in err


def test_help_works_with_effing_blocked(tmp_path):
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        textwrap.dedent(
            f"""
            import sys

            class _BlockEffing:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == {EFFING!r} or fullname.startswith({EFFING!r} + "."):
                        raise ImportError(f"{{fullname}} blocked for test")
                    return None

            sys.meta_path.insert(0, _BlockEffing())
            """
        )
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, str(_maintenance_path()), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "Traceback" not in result.stderr
