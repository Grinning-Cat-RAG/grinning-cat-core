"""A plugin toggled on a system level by one POD reaches the other PODs (March Hare).

Every POD keeps its system-level plugins in memory (BillTheLizard), loaded at startup; the agents read theirs from the
database whenever a Cheshire Cat is created. Here the lizard of the test plays "this POD", and "another POD" is
simulated by writing the system list of the active plugins into the database, as the other POD's toggle does, then
delivering its event.

Invariant (checked once the events are delivered): the plugins active in the memory of the POD are the ones active in
the database; an active plugin has its hooks and its endpoints mounted exactly once, an inactive one has neither.
"""
import json
import os
import shutil
from unittest import mock

from hypothesis import HealthCheck, given, settings, strategies as st

from cat.core_plugins.march_hare import march_hare as mh
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.db.models import Setting

PLUGIN = "mock_plugin"


async def stored_active() -> list:
    return (await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, "active_plugins"))["value"]


async def toggled_by_another_pod(active: bool) -> None:
    """What the toggle of another POD leaves in the database."""
    plugins = [p for p in await stored_active() if p != PLUGIN] + ([PLUGIN] if active else [])
    await crud_settings.upsert_setting_by_name(DEFAULT_SYSTEM_KEY, Setting(name="active_plugins", value=plugins))


async def deliver(lizard, source_pod: str = "another-pod") -> None:
    await mh._handle_plugin_event(lizard, {
        "event_type": mh.MarchHareConfig.events["PLUGIN_SYSTEM_TOGGLE"],
        "source_pod": source_pod,
        "payload": json.dumps({"plugin_id": PLUGIN}),
    })


def has_hooks(lizard) -> bool:
    return any(h.plugin_id == PLUGIN for hooks in lizard.plugin_manager.hooks.values() for h in hooks)


def mounted(lizard, endpoints) -> list[int]:
    """How many times each endpoint of the plugin is mounted on the API of the POD."""
    return [
        sum(1 for r in lizard.fastapi_app.routes if getattr(r, "path", None) == e.name and r.methods == e.methods)
        for e in endpoints
    ]


async def assert_aligned(lizard, endpoints) -> None:
    active = PLUGIN in await stored_active()
    assert set(lizard.plugin_manager.active_plugins) == set(await stored_active())
    assert (PLUGIN in lizard.plugin_manager.plugins) == active
    assert has_hooks(lizard) == active
    assert mounted(lizard, endpoints) == [1 if active else 0] * len(endpoints)


async def test_a_system_toggle_is_published(lizard, plugin_manager):
    with mock.patch.object(mh._march_hare, "notify_event", mock.AsyncMock()) as notify:
        await lizard.toggle_plugin(PLUGIN)
    notify.assert_awaited_once_with(
        event_type=mh.MarchHareConfig.events["PLUGIN_SYSTEM_TOGGLE"],
        payload={"plugin_id": PLUGIN},
        stream_name=mh.MarchHareConfig.streams["PLUGIN_EVENTS"],
    )


async def test_deactivated_by_another_pod(lizard, plugin_manager):
    endpoints = list(lizard.plugin_manager.plugins[PLUGIN].endpoints)
    assert endpoints and mounted(lizard, endpoints) == [1] * len(endpoints)

    await toggled_by_another_pod(active=False)
    stored = await stored_active()
    await deliver(lizard)

    await assert_aligned(lizard, endpoints)
    assert PLUGIN not in lizard.plugin_manager.active_plugins
    assert await stored_active() == stored  # nothing written back


async def test_activated_by_another_pod(lizard, plugin_manager):
    endpoints = list(lizard.plugin_manager.plugins[PLUGIN].endpoints)
    await lizard.toggle_plugin(PLUGIN)  # off here, as if this POD started with the plugin off
    assert not has_hooks(lizard)

    await toggled_by_another_pod(active=True)
    await deliver(lizard)

    await assert_aligned(lizard, endpoints)
    assert PLUGIN in lizard.plugin_manager.active_plugins


async def test_an_event_of_the_pod_itself_is_ignored(lizard, plugin_manager):
    await toggled_by_another_pod(active=False)
    with mock.patch.object(lizard, "sync_system_plugins", mock.AsyncMock()) as sync:
        await deliver(lizard, source_pod=mh._march_hare.pod_id)
    sync.assert_not_awaited()


async def test_duplicated_events_change_nothing(lizard, plugin_manager):
    endpoints = list(lizard.plugin_manager.plugins[PLUGIN].endpoints)
    await toggled_by_another_pod(active=False)
    await deliver(lizard)
    await toggled_by_another_pod(active=True)
    for _ in range(3):
        await deliver(lizard)
    await assert_aligned(lizard, endpoints)


# ------------------------------------------------------------------ property-based, with fault injection

operation = st.one_of(
    st.tuples(st.just("another_pod_toggles"), st.booleans()),
    # an event delivered 0 (lost), 1 or 2 (duplicated) times; or the database failing while the POD aligns
    st.tuples(st.just("deliver"), st.integers(min_value=0, max_value=2), st.booleans()),
    st.tuples(st.just("this_pod_toggles")),
)


@settings(
    max_examples=25, deadline=None, database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(operations=st.lists(operation, min_size=1, max_size=12))
async def test_the_pods_converge_on_the_database(lizard, plugin_manager, operations):
    # every example starts from the plugin active everywhere
    await toggled_by_another_pod(active=True)
    await deliver(lizard)
    endpoints = list(lizard.plugin_manager.plugins[PLUGIN].endpoints)
    await assert_aligned(lizard, endpoints)

    for op in operations:
        if op[0] == "another_pod_toggles":
            await toggled_by_another_pod(active=op[1])
        elif op[0] == "this_pod_toggles":
            with mock.patch.object(mh._march_hare, "notify_event", mock.AsyncMock()):
                await lizard.toggle_plugin(PLUGIN)
            await assert_aligned(lizard, endpoints)
        else:
            _, times, db_fails = op
            if db_fails:
                failing = mock.AsyncMock(side_effect=ConnectionError("redis down"))
                with mock.patch.object(lizard.plugin_manager, "load_active_plugins_ids_from_db", failing):
                    await deliver(lizard)  # the error is logged, the POD stays as it was
                continue
            for _ in range(times):
                await deliver(lizard)
            if times:
                await assert_aligned(lizard, endpoints)

    # whatever was lost or failed, the next event aligns the POD
    await deliver(lizard)
    await assert_aligned(lizard, endpoints)



# ------------------------------------------------------------------ uninstallation by another POD

async def test_uninstalled_by_another_pod(lizard, plugin_manager):
    plugin = lizard.plugin_manager.plugins[PLUGIN]
    endpoints = list(plugin.endpoints)
    assert has_hooks(lizard) and mounted(lizard, endpoints) == [1] * len(endpoints)

    # what the other POD leaves: no folder on the shared volume, no plugin among the active ones in the database
    shutil.rmtree(plugin.path)
    await toggled_by_another_pod(active=False)
    stored = await stored_active()

    await mh._handle_plugin_event(lizard, {
        "event_type": mh.MarchHareConfig.events["PLUGIN_UNINSTALLATION"],
        "source_pod": "another-pod",
        "payload": json.dumps({"plugin_id": PLUGIN}),
    })

    assert PLUGIN not in lizard.plugin_manager.active_plugins
    assert PLUGIN not in lizard.plugin_manager.plugins
    assert not has_hooks(lizard)
    assert mounted(lizard, endpoints) == [0] * len(endpoints)
    assert await stored_active() == stored  # nothing written back


async def test_a_toggle_event_also_forgets_a_plugin_uninstalled_elsewhere(lizard, plugin_manager):
    """The uninstallation event was lost: the next system toggle event aligns the POD anyway."""
    plugin = lizard.plugin_manager.plugins[PLUGIN]
    endpoints = list(plugin.endpoints)
    shutil.rmtree(plugin.path)
    await toggled_by_another_pod(active=False)

    await deliver(lizard)

    assert PLUGIN not in lizard.plugin_manager.active_plugins
    assert PLUGIN not in lizard.plugin_manager.plugins
    assert not has_hooks(lizard)
    assert mounted(lizard, endpoints) == [0] * len(endpoints)


# ------------------------------------------------------------------ startup

async def test_a_starting_pod_aligns_once_with_the_database(lizard):
    """The events published while a POD starts are not handled: it aligns once, when ready."""
    with mock.patch.object(lizard, "sync_system_plugins", mock.AsyncMock()) as sync:
        await mh.after_lizard_bootstrap.function(lizard)
    sync.assert_awaited_once()


async def test_a_failed_alignment_does_not_stop_the_startup(lizard):
    failing = mock.AsyncMock(side_effect=ConnectionError("redis down"))
    with mock.patch.object(lizard, "sync_system_plugins", failing):
        await mh.after_lizard_bootstrap.function(lizard)  # logged, not raised
    failing.assert_awaited_once()


async def test_an_inactive_plugin_can_be_uninstalled(lizard, plugin_manager):
    path = lizard.plugin_manager.plugins[PLUGIN].path
    await lizard.toggle_plugin(PLUGIN)  # off on a system level: no longer among the loaded plugins
    assert PLUGIN not in lizard.plugin_manager.plugins

    await lizard.uninstall_plugin(PLUGIN)

    assert not os.path.exists(path)


async def test_installed_by_another_pod_is_not_written_back(lizard):
    with mock.patch.object(lizard.plugin_manager, "install_extracted_plugin", mock.AsyncMock()) as install, \
            mock.patch.object(lizard, "activate_plugin_endpoints") as endpoints:
        await mh._handle_plugin_event(lizard, {
            "event_type": mh.MarchHareConfig.events["PLUGIN_INSTALLATION"],
            "source_pod": "another-pod",
            "payload": json.dumps({"plugin_id": PLUGIN}),
        })
    install.assert_awaited_once_with(PLUGIN, persist=False)  # the other POD already stored the active plugins
    endpoints.assert_called_once_with(PLUGIN)


async def test_a_plugin_failing_to_deactivate_keeps_its_endpoints(lizard, plugin_manager):
    endpoints = list(lizard.plugin_manager.plugins[PLUGIN].endpoints)
    await toggled_by_another_pod(active=False)
    failing = mock.AsyncMock(side_effect=RuntimeError("boom"))
    with mock.patch.object(lizard.plugin_manager, "deactivate_plugin", failing):
        await deliver(lizard)  # logged
    assert PLUGIN in lizard.plugin_manager.active_plugins  # still running here...
    assert mounted(lizard, endpoints) == [1] * len(endpoints)  # ...and still reachable: consistent

    await deliver(lizard)  # the next event aligns it
    await assert_aligned(lizard, endpoints)


async def test_a_plugin_failing_to_activate_gets_no_endpoints(lizard, plugin_manager):
    endpoints = list(lizard.plugin_manager.plugins[PLUGIN].endpoints)
    await lizard.toggle_plugin(PLUGIN)
    await toggled_by_another_pod(active=True)
    failing = mock.AsyncMock(side_effect=RuntimeError("boom"))
    with mock.patch.object(lizard.plugin_manager, "activate_plugin", failing):
        await deliver(lizard)  # logged
    assert PLUGIN not in lizard.plugin_manager.active_plugins
    assert mounted(lizard, endpoints) == [0] * len(endpoints)


async def test_a_core_plugin_is_never_uninstalled(lizard):
    await lizard.plugin_manager.uninstall_plugin("base_plugin")
    assert lizard.plugin_manager.plugin_exists("base_plugin")
    assert "base_plugin" in lizard.plugin_manager.active_plugins


async def test_a_plugin_uninstalled_elsewhere_is_forgotten_even_if_its_deactivation_fails(lizard, plugin_manager):
    plugin = lizard.plugin_manager.plugins[PLUGIN]
    shutil.rmtree(plugin.path)
    with mock.patch.object(type(plugin), "deactivate", mock.AsyncMock(side_effect=RuntimeError("boom"))):
        await lizard.plugin_manager.uninstall_plugin(PLUGIN)  # logged
    assert PLUGIN not in lizard.plugin_manager.plugins
    assert PLUGIN not in lizard.plugin_manager.active_plugins
    assert not has_hooks(lizard)


async def test_forgetting_a_plugin_listed_active_but_never_loaded(lizard, plugin_manager):
    """E.g. a plugin that failed to load at startup: it stays among the active ones, not among the loaded ones."""
    plugin = lizard.plugin_manager.plugins.pop(PLUGIN)
    shutil.rmtree(plugin.path)
    await lizard.plugin_manager.uninstall_plugin(PLUGIN)
    assert PLUGIN not in lizard.plugin_manager.active_plugins


async def test_forgetting_a_plugin_loaded_but_not_listed_active(lizard, plugin_manager):
    lizard.plugin_manager.active_plugins.remove(PLUGIN)
    shutil.rmtree(lizard.plugin_manager.plugins[PLUGIN].path)
    await lizard.plugin_manager.uninstall_plugin(PLUGIN)
    assert PLUGIN not in lizard.plugin_manager.plugins
