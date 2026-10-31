"""System-only EffING with agent-level phase providers.

The efficient-ingestion engine is a SYSTEM-level factory service (like the
embedder): ``cat_efficient_ingestion`` is system-only, so its OWN phase hooks
(``ingestion_phase_pending`` / ``_run`` / ``_settings_marker``) are dispatched
on BillTheLizard's plugin manager. The phase PROVIDERS (MyGRAPH, CATALOG) stay
agent-level plugins, so EffING must collect ``ingestion_phase_specs`` from BOTH
managers and run each phase on the manager that owns it.

These tests lock the three failure modes of a single-manager dispatch:

(a) an agent WITHOUT ``cat_efficient_ingestion`` still runs the built-in
    phases (they belong to the system manager);
(b) an agent WITH MyGRAPH still registers and runs ``graph_spacy_extraction``
    and ``graph_llm_extraction`` (agent-level ownership is preserved);
(c) a normal agent emits no ``Phase hook ... not present in any active
    plugin`` warning.
"""

import inspect
from types import SimpleNamespace

from cat.core_plugins.cat_efficient_ingestion import hooks, phases
from cat.core_plugins.cat_efficient_ingestion.phases import PhaseSpec


class _Manager:
    """Minimal plugin manager honouring the accumulator/threading contract."""

    def __init__(self, name, hooks=None, active_plugins=None, context="cat"):
        self.name = name
        self.hooks = hooks or {}
        self.active_plugins = list(active_plugins or [])
        # the real managers expose this property: "lizard" on the system
        # manager, "cat" on an agent manager. It decides the caller kwarg.
        self.context_execute_hook = context
        self.calls = []
        # phases the run hook actually executed on this manager
        self.run_calls: list[str] = []

    async def execute_hook(self, hook_name, *args, caller=None):
        self.calls.append((hook_name, args))
        functions = self.hooks.get(hook_name)
        if not functions:
            raise Exception(f"Hook {hook_name} not present in any plugin")
        value = args[0] if args else None
        for function in functions:
            result = function(value, *args[1:], **{self.context_execute_hook: caller})
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                value = result
        return value


def _hook_function(hook):
    """The plain function behind a ``CatHook`` (the hook itself in fakes)."""
    return getattr(hook, "function", hook)


def _effing_system_manager():
    """The system manager: EffING's own built-in hooks only."""
    run_calls = []

    def system_run(phase, source, completed_phases, cat=None, lizard=None):
        run_calls.append(phase)
        return {"status": "done", "ran_by": "system"}

    manager = _Manager(
        "system",
        context="lizard",
        hooks={
            "ingestion_phase_specs": [_hook_function(hooks.ingestion_phase_specs)],
            "ingestion_phase_pending": [_hook_function(phases.ingestion_phase_pending)],
            "ingestion_phase_run": [system_run],
            "ingestion_phase_settings_marker": [
                _hook_function(hooks.ingestion_phase_settings_marker)
            ],
            "before_ingestion_status_completed": [
                _hook_function(hooks.before_ingestion_status_completed)
            ],
        },
    )
    manager.run_calls = run_calls
    return manager


def _mygraph_agent_manager():
    """An agent manager carrying MyGRAPH's two phase registrations."""

    def specs(accumulator, cat):
        accumulator.append(
            PhaseSpec("graph_spacy_extraction", None, ("parsing_chunking", "embedding"))
        )
        accumulator.append(
            PhaseSpec("graph_llm_extraction", None, ("parsing_chunking", "embedding"))
        )
        return accumulator

    def marker(phase, cat):
        if phase in ("graph_spacy_extraction", "graph_llm_extraction"):
            return f"marker::{phase}"
        return None

    run_calls = []

    def run(phase, source, completed_phases, cat):
        if phase in ("graph_spacy_extraction", "graph_llm_extraction"):
            run_calls.append(phase)
            return {"status": "done", "ran_by": "agent"}
        raise AssertionError(f"agent manager must not run built-in phase {phase}")

    def gate(source, cat):
        return None

    manager = _Manager(
        "agent",
        active_plugins=["cat_graphrag.my", "cat_alog"],
        hooks={
            "ingestion_phase_specs": [specs],
            "ingestion_phase_settings_marker": [marker],
            "ingestion_phase_run": [run],
            "before_ingestion_status_completed": [gate],
        },
    )
    manager.run_calls = run_calls
    return manager


def _cat(system_manager, agent_manager):
    return SimpleNamespace(
        agent_key="agent_test",
        lizard=SimpleNamespace(plugin_manager=system_manager),
        plugin_manager=agent_manager,
    )


async def _no_settings(*args, **kwargs):
    return None


async def test_agent_without_effing_still_runs_builtin_phases(monkeypatch):
    """(a) The built-in phases live on the system manager, not the agent."""
    monkeypatch.setattr(phases.crud_settings, "get_settings_by_category", _no_settings)

    system_manager = _effing_system_manager()
    # the agent's active_plugins predates EffING: no phase hook on it at all
    agent_manager = _Manager("agent", active_plugins=["cat_graphrag.my"])
    cat = _cat(system_manager, agent_manager)

    assert "cat_efficient_ingestion" not in agent_manager.active_plugins
    assert await phases.merged_phases(cat, cat) == phases.PHASES

    pending = await hooks.execute_phase_accumulator(
        cat, "ingestion_phase_pending", [], "doc.pdf", [], default=list
    )
    entries = pending if isinstance(pending, list) else []
    assert sorted(entry["phase"] for entry in entries) == [
        "embedding",
        "parsing_chunking",
    ]

    # built-ins are owned by the system manager
    assert await phases.phase_owner(cat, "parsing_chunking") is system_manager
    assert await phases.phase_owner(cat, "embedding") is system_manager

    # the run dispatch reaches the system manager, never the agent one
    result = await phases.execute_phase_on_owner(
        cat, "ingestion_phase_run", "embedding", "embedding", "doc.pdf", []
    )
    assert result == {"status": "done", "ran_by": "system"}
    assert system_manager.run_calls == ["embedding"]
    assert agent_manager.calls == []


async def test_agent_with_mygraph_registers_and_runs_graph_phases():
    """(b) Agent-level MyGRAPH phases are collected from both managers and run
    on the manager that owns them."""
    system_manager = _effing_system_manager()
    agent_manager = _mygraph_agent_manager()
    cat = _cat(system_manager, agent_manager)

    merged = await phases.merged_phases(cat, cat)
    assert "graph_spacy_extraction" in merged
    assert "graph_llm_extraction" in merged
    # built-ins still win and stay intact
    assert merged["parsing_chunking"] == phases.PHASES["parsing_chunking"]

    for phase in ("graph_spacy_extraction", "graph_llm_extraction"):
        assert await phases.phase_owner(cat, phase) is agent_manager
        result = await phases.execute_phase_on_owner(
            cat, "ingestion_phase_run", phase, phase, "doc.pdf", []
        )
        assert result == {"status": "done", "ran_by": "agent"}

    run_calls = [call for call in agent_manager.run_calls]
    assert run_calls == ["graph_spacy_extraction", "graph_llm_extraction"]
    # the system manager never receives an external phase it does not own
    assert system_manager.run_calls == []


async def test_normal_agent_emits_no_missing_phase_hook_warning(monkeypatch):
    """(c) A normal agent (system EffING + agent MyGRAPH) never logs the
    "Phase hook ... not present in any active plugin" warning."""
    monkeypatch.setattr(phases.crud_settings, "get_settings_by_category", _no_settings)

    warnings = []
    monkeypatch.setattr(hooks.log, "warning", warnings.append)

    system_manager = _effing_system_manager()
    agent_manager = _mygraph_agent_manager()
    cat = _cat(system_manager, agent_manager)

    await hooks.execute_phase_accumulator(
        cat, "ingestion_phase_pending", [], "doc.pdf", [], default=list
    )
    await hooks.execute_phase_hook_on_all(
        cat, "before_ingestion_status_completed", "doc.pdf"
    )

    assert not [
        message
        for message in warnings
        if "not present in any active plugin" in message
    ]


async def test_agent_level_providers_also_loaded_on_system_are_skipped(monkeypatch):
    """(d) A cat-only provider that is ALSO legacy-active on the system manager
    is skipped there instead of raising ``unexpected keyword argument 'lizard'``.

    On the deployed instance MyGRAPH/CATALOG are in ``system:agent``'s
    active_plugins too. Dispatching their hooks through the system manager would
    feed them ``lizard=`` (they declare ``cat``), logging an error on every
    probe. The compat dispatcher must skip them there and use the agent manager.
    """
    monkeypatch.setattr(phases.crud_settings, "get_settings_by_category", _no_settings)
    errors = []
    monkeypatch.setattr(hooks.log, "error", errors.append)

    system_manager = _effing_system_manager()
    agent_manager = _mygraph_agent_manager()
    # legacy dual-active: the same cat-only provider hooks sit on the system
    # manager as well
    for hook_name in (
        "ingestion_phase_specs",
        "ingestion_phase_settings_marker",
        "ingestion_phase_run",
        "before_ingestion_status_completed",
    ):
        system_manager.hooks[hook_name].extend(agent_manager.hooks[hook_name])
    cat = _cat(system_manager, agent_manager)

    merged = await phases.merged_phases(cat, cat)
    assert "graph_spacy_extraction" in merged
    assert "graph_llm_extraction" in merged

    # built-ins use the system manager (EffING), external phases the agent one
    assert await phases.phase_owner(cat, "embedding") is system_manager
    assert await phases.phase_owner(cat, "graph_spacy_extraction") is agent_manager

    await hooks.execute_phase_accumulator(
        cat, "ingestion_phase_pending", [], "doc.pdf", [], default=list
    )
    await hooks.execute_phase_hook_on_all(
        cat, "before_ingestion_status_completed", "doc.pdf"
    )
    result = await phases.execute_phase_on_owner(
        cat, "ingestion_phase_run", "graph_spacy_extraction",
        "graph_spacy_extraction", "doc.pdf", [],
    )
    assert result == {"status": "done", "ran_by": "agent"}

    assert not [
        message for message in errors if "unexpected keyword argument" in message
    ]
