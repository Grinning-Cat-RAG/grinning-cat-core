"""Dependency-metadata alignment with upstream CAT (plan Todo 6).

Upstream CAT resolves plugin-to-plugin dependencies from ``plugin.json``
``dependencies`` — exact folder-id strings matched against the plugin folders
(``Plugin.missing_dependencies``) — and installs pip dependencies from
``pyproject.toml`` ``[project].dependencies`` via uv (``requirements.txt`` is
not supported). This test pins both sides of that contract for EffING:

- the declared dependency string is the real folder id of the multimodal
  plugin (``cat_multimodal_ingestion``): the hyphenated repo name
  (``cat-multimodal-ingestion``) is NOT a folder id and would be reported
  missing, silently preventing the plugin from loading;
- the plugin loads on CAT with the dependency resolved, and a genuinely
  missing dependency is reported loudly (never silently skipped).

Runs against upstream CAT: the plugin folder is resolved via ``cat.utils``
when the file is copied into the host's ``tests/`` (the EffING-vs-CAT
harness), or by walking up to the folder holding ``plugin.json`` when run
from the plugin repo itself.
"""

import json
import tomllib
from pathlib import Path

import pytest

from cat.looking_glass.mad_hatter.plugin import Plugin

PLUGIN_ID = "cat_efficient_ingestion"
MMODING_ID = "cat_multimodal_ingestion"


def _real_plugins_path() -> Path:
    """The real ``cat/core_plugins`` folder, immune to the test conftest's mock.

    The canonical plugin path: the host loads ``cat/core_plugins/`` in
    preference to ``cat/plugins/`` (``mad_hatter.py``), so the harness must
    resolve EffING and its MmodING dependency there.
    """
    from cat import utils

    return Path(utils.__file__).resolve().parent / "core_plugins"


def _plugin_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent, here.parent.parent):
        if (candidate / "plugin.json").is_file():
            return candidate
    return _real_plugins_path() / PLUGIN_ID


def _manifest() -> dict:
    return json.loads((_plugin_dir() / "plugin.json").read_text())


def test_plugin_json_dependency_matches_folder_id():
    """The declared dependency is the folder id of the multimodal plugin.

    Upstream matches dependencies against plugin folder ids exactly; the
    hyphenated repo name (``cat-multimodal-ingestion``) is not a folder id and
    would be reported missing, blocking the load.
    """
    assert _manifest()["dependencies"] == [MMODING_ID]
    assert (_real_plugins_path() / MMODING_ID).is_dir(), (
        f"multimodal plugin folder {MMODING_ID!r} not found in the plugins path"
    )


def test_pyproject_declares_project_dependencies():
    """CAT installs plugin pip deps from ``pyproject.toml [project].dependencies`` (uv)."""
    data = tomllib.loads((_plugin_dir() / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert deps, "pyproject.toml must declare [project].dependencies for CAT's uv-based install"


def test_loads_on_cat_with_dependency_resolved():
    """The plugin loads on CAT: no missing dependencies when the multimodal plugin is present."""
    plugin = Plugin(str(_plugin_dir()))
    assert plugin.missing_dependencies([PLUGIN_ID, MMODING_ID]) == []


def test_missing_dependency_is_reported_not_silent():
    """Failure mode: a missing dependency is reported clearly, never silently skipped."""
    plugin = Plugin(str(_plugin_dir()))
    assert plugin.missing_dependencies([PLUGIN_ID]) == [MMODING_ID]


@pytest.mark.asyncio
async def test_plugin_manager_loads_effing_with_dependency_resolved(lizard, monkeypatch):
    """End-to-end: the CAT plugin manager loads EffING with its dependency resolved.

    The test conftest redirects ``get_plugins_path`` to a mock folder; restore
    the real one for this test so the manager resolves the installed plugins.
    """
    from cat import utils as cat_utils

    monkeypatch.setattr(cat_utils, "get_plugins_path", lambda: str(_real_plugins_path()))

    loaded = await lizard.plugin_manager.load_plugin(PLUGIN_ID)
    assert loaded.plugin is not None, (
        f"plugin failed to load; missing dependencies: {loaded.missing_dependencies}"
    )
    assert loaded.missing_dependencies == []