import pytest
from unittest.mock import MagicMock
from cat.looking_glass.mad_hatter.mad_hatter import MadHatter


@pytest.mark.asyncio
async def test_hooks_execution_order_is_deterministic(mock_plugins_env, test_agent_key: str):
    """
    Verify that plugin hooks with equal priorities execute in a deterministic order
    across repeated discoveries, avoiding issues caused by hash-randomized sets.
    """
    mad_hatter = MadHatter(test_agent_key)
    await mad_hatter.discover_plugins()

    hook_name = "before_cat_sends_message"
    assert hook_name in mad_hatter.hooks, f"Expected hook '{hook_name}' to be registered"

    initial_order = [h.plugin_id for h in mad_hatter.hooks[hook_name]]

    # Re-discover plugins multiple times to verify consistency across boots
    for _ in range(5):
        await mad_hatter.discover_plugins()
        current_order = [h.plugin_id for h in mad_hatter.hooks[hook_name]]
        assert current_order == initial_order, (
            "Hook execution order changed across discoveries. Deduplication must be deterministic."
        )


def test_topological_sort_plugins_ordering(test_agent_key: str):
    """
    Verify that `_topological_sort_plugins` correctly orders dependent plugins
    after their required dependencies.
    """
    mad_hatter = MadHatter(test_agent_key)

    # Setup mock plugins where plugin_c -> plugin_b -> plugin_a
    plugin_a = MagicMock()
    plugin_a.manifest.dependencies = []

    plugin_b = MagicMock()
    plugin_b.manifest.dependencies = ["plugin_a"]

    plugin_c = MagicMock()
    plugin_c.manifest.dependencies = ["plugin_b"]

    mad_hatter.plugins = {
        "plugin_c": plugin_c,
        "plugin_b": plugin_b,
        "plugin_a": plugin_a,
    }

    # Pass plugin IDs in arbitrary/reversed order
    input_ids = ["plugin_c", "plugin_b", "plugin_a"]
    sorted_ids = mad_hatter._topological_sort_plugins(input_ids)

    # Verify dependency resolution order (A must precede B, B must precede C)
    assert sorted_ids.index("plugin_a") < sorted_ids.index("plugin_b")
    assert sorted_ids.index("plugin_b") < sorted_ids.index("plugin_c")


def test_topological_sort_circular_dependency_fallback(test_agent_key: str):
    """
    Verify that circular dependencies do not trigger unhandled exceptions
    and safely fall back to the original input order.
    """
    mad_hatter = MadHatter(test_agent_key)

    # Create a cyclic dependency: plugin_a <-> plugin_b
    plugin_a = MagicMock()
    plugin_a.manifest.dependencies = ["plugin_b"]

    plugin_b = MagicMock()
    plugin_b.manifest.dependencies = ["plugin_a"]

    mad_hatter.plugins = {
        "plugin_a": plugin_a,
        "plugin_b": plugin_b,
    }

    input_ids = ["plugin_a", "plugin_b"]
    sorted_ids = mad_hatter._topological_sort_plugins(input_ids)

    # Verify fallback to original order on cycle detection
    assert sorted_ids == input_ids


@pytest.mark.asyncio
async def test_topological_sorting_with_filesystem_env(mock_plugins_env, test_agent_key: str):
    """
    Integration test verifying topological sorting against real plugin files
    created dynamically on disk via the mock_plugins_env fixture.
    """
    mad_hatter = MadHatter(test_agent_key)

    plugin_ids = ["plugin_c", "plugin_b", "plugin_a"]
    for p_id in plugin_ids:
        loaded = await mad_hatter.load_plugin(p_id, with_deactivation=False)
        if loaded.plugin:
            mad_hatter.plugins[p_id] = loaded.plugin

    sorted_ids = mad_hatter._topological_sort_plugins(plugin_ids)

    assert sorted_ids.index("plugin_a") < sorted_ids.index("plugin_b")
    assert sorted_ids.index("plugin_b") < sorted_ids.index("plugin_c")


@pytest.mark.asyncio
async def test_why_hook_executes_before_conversation_history(mock_plugins_env, test_agent_key: str):
    """
    Specific integration test ensuring that the 'why' plugin hook executes
    before the 'conversation_history' hook when sending messages.
    """
    # Create why and conversation_history plugins on the mock filesystem
    custom_dir = mock_plugins_env["custom_plugins_dir"]
    mock_plugins_env["create_plugin"](custom_dir, "why", priority=2)
    mock_plugins_env["create_plugin"](custom_dir, "conversation_history", priority=1)

    mad_hatter = MadHatter(test_agent_key)
    await mad_hatter.discover_plugins()

    hooks = mad_hatter.hooks.get("before_cat_sends_message", [])

    why_index = next((i for i, h in enumerate(hooks) if h.plugin_id == "why"), None)
    history_index = next((i for i, h in enumerate(hooks) if h.plugin_id == "conversation_history"), None)

    assert why_index is not None, "Hook from 'why' plugin was not registered"
    assert history_index is not None, "Hook from 'conversation_history' plugin was not registered"

    assert why_index < history_index, (
        f"The 'why' hook (priority {hooks[why_index].priority}) must execute BEFORE "
        f"the 'conversation_history' hook (priority {hooks[history_index].priority})."
    )
