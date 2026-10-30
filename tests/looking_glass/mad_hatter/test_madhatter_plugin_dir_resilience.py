"""Resilience of plugin discovery against inaccessible core_plugins entries.

Reproducing an unreadable entry on every OS/user is tricky: on POSIX a
``chmod 000`` directory denies access to a normal user, but it does NOT deny
access to root (CAP_DAC_OVERRIDE bypasses the permission check). To have a
deterministic OSError regardless of the uid running the test, we create a
self-referential symlink (``bad_plugin -> bad_plugin``): resolving it raises
``OSError`` (ELOOP, "Too many levels of symbolic links") for root and non-root
alike. This mirrors the production failure (a symlink whose target cannot be
resolved/read) without depending on permission semantics.
"""
from unittest.mock import patch

import pytest

import cat.utils as utils
from cat.looking_glass.mad_hatter import mad_hatter as mad_hatter_module
from cat.looking_glass.mad_hatter.mad_hatter import MadHatter


def _make_unreadable_symlink(folder, name: str) -> str:
    target = folder / name
    target.symlink_to(name)  # self-loop -> OSError on stat/resolve
    return str(target)


def test_get_core_plugins_ids_skips_unreadable_entry(tmp_path, monkeypatch):
    core_plugins = tmp_path / "core_plugins"
    core_plugins.mkdir()
    (core_plugins / "good_plugin").mkdir()

    bad_path = _make_unreadable_symlink(core_plugins, "bad_plugin")

    monkeypatch.setattr(utils, "get_core_plugins_path", lambda: str(core_plugins))

    mad_hatter = MadHatter("test_agent_key")

    with patch.object(mad_hatter_module.log, "warning") as warning:
        ids = mad_hatter.get_core_plugins_ids

    assert ids == ["good_plugin"]
    assert "bad_plugin" not in ids
    assert warning.call_count == 1
    assert bad_path in warning.call_args.args[0]


@pytest.mark.asyncio
async def test_load_plugin_bad_id_logs_and_returns(tmp_path, monkeypatch):
    core_plugins = tmp_path / "core_plugins"
    core_plugins.mkdir()
    (core_plugins / "good_plugin").mkdir()
    plugins_dir = tmp_path / "plugins"

    monkeypatch.setattr(utils, "get_core_plugins_path", lambda: str(core_plugins))
    monkeypatch.setattr(utils, "get_plugins_path", lambda: str(plugins_dir))

    mad_hatter = MadHatter("test_agent_key")

    with patch.object(mad_hatter_module.log, "error") as error:
        result = await mad_hatter.load_plugin("bad_plugin", with_deactivation=False)

    assert result.plugin is None
    error.assert_called_once()
    message = error.call_args.args[0]
    assert "bad_plugin" in message
    assert str(plugins_dir / "bad_plugin") in message
