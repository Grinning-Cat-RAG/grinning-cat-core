"""The phases of an ingestion: the API the plugins implement to add their own phases, and the two built-in ones.

A source goes through a graph of phases, each one depending on others (``depends_on``). The RabbitHole runs a phase
when it is stale: never completed for the current content, its *tag* changed, or a phase it depends on ran again.
The tag of a phase is the SHA-256 of its id and of its material settings (``settings``), folded with the tags of its
dependencies: a change of the chunker makes ``parsing_chunking`` stale, and so ``embedding`` too; a change of the
embedder makes only ``embedding`` stale, which reuses the stored chunks.

Built-in phases:

- ``parsing_chunking``: parses the source, collects the images, splits the documents in chunks and stores the chunks
  (see ``cat.rabbit_hole.chunks``);
- ``embedding``: embeds the stored chunks (and the images, through the plugins) and replaces the points of the source.

A plugin adds phases by returning ``IngestionPhase`` subclasses from the ``rabbithole_ingestion_phases`` hook: the
RabbitHole instantiates them without arguments, for every run (a phase keeps no state of its own between runs).
"""
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Tuple

from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.utils import SUFFIX_TO_CRYPT

PHASE_PARSING_CHUNKING = "parsing_chunking"
PHASE_EMBEDDING = "embedding"


@dataclass(frozen=True)
class IngestionSource:
    """A source of an agent: in its knowledge base (``chat_id=None``) or in a chat."""

    agent_id: str
    chat_id: str | None
    name: str
    type: Literal["file", "url"] = "file"


@dataclass(frozen=True)
class PhaseResult:
    """The outcome of a phase.

    - ``done``: completed (``processed``/``expected``/``warnings`` are kept in the diary);
    - ``not_ready``: retriable, not runnable yet: the RabbitHole retries after ``retry_after`` seconds, within the budget
      of the phase (``IngestionPhase.not_ready_budget_seconds`` or ``CAT_INGESTION_NOT_READY_MAX_SECONDS``);
    - ``skipped``: nothing to do for this source (its dependents are skipped too);
    - ``error``: failed (its dependents fail too). An exception raised by the phase is an error.
    """

    status: Literal["done", "not_ready", "skipped", "error"]
    reason: str | None = None
    retry_after: float | None = None
    processed: int | None = None
    expected: int | None = None
    warnings: List[Dict] | None = field(default=None)


class IngestionPhase(ABC):
    """A phase of the ingestion of a source. ``id``: lowercase letters, digits and underscores, starting with a letter.

    A plugin phase is instantiated without arguments for every run: what it needs comes from ``cat`` and ``source``."""

    id: str = ""
    label: str = ""
    depends_on: Tuple[str, ...] = ()
    not_ready_budget_seconds: float | None = None

    async def settings(self, cat) -> Any:
        """The material settings of the phase: when they change, the phase (and its dependents) run again."""
        return None

    async def applies_to(self, source: IngestionSource, cat) -> Tuple[bool, str | None]:
        """Whether the phase applies to the source; when it does not, the phase is ``skipped`` with the reason."""
        return True, None

    @abstractmethod
    async def process(self, source: IngestionSource, cat) -> PhaseResult:
        """Run the phase on the source. ``cat`` is the CheshireCat of the agent.

        The phase may run again on the same source: after a crash of the worker, a new upload of the source, or a
        change of its settings. It replaces what it produced for the source before, never adds to it."""

    @abstractmethod
    async def delete(self, source: IngestionSource, cat) -> None:
        """Remove what the phase produced for the source. Idempotent: it may run again after an interruption."""

    @property
    def display_label(self) -> str:
        """The label shown to the users (the id when the phase has none)."""
        return self.label or self.id.replace("_", " ").capitalize()


class PhaseGraphError(ValueError):
    """The phases do not form a graph: a cycle, or a dependency on a phase that does not exist."""


def execution_order(phases: Iterable[IngestionPhase]) -> List[IngestionPhase]:
    """The phases in dependency order (each one after the phases it depends on), stable on the given order."""
    phases = list(phases)
    by_id = {phase.id: phase for phase in phases}
    for phase in phases:
        missing = [dep for dep in phase.depends_on if dep not in by_id]
        if missing:
            raise PhaseGraphError(f"The phase {phase.id!r} depends on unknown phases {missing}")

    ordered: List[IngestionPhase] = []
    visiting: set = set()

    def visit(phase: IngestionPhase, path: List[str]):
        if phase in ordered:
            return
        if phase.id in visiting:
            raise PhaseGraphError(f"The phases form a cycle: {' -> '.join(path + [phase.id])}")
        visiting.add(phase.id)
        for dep in phase.depends_on:
            visit(by_id[dep], path + [phase.id])
        visiting.discard(phase.id)
        ordered.append(phase)

    for phase in phases:
        visit(phase, [])
    return ordered


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


async def compute_tags(ordered: List[IngestionPhase], cat) -> Dict[str, str]:
    """The tag of every phase (``ordered`` as returned by ``execution_order``)."""
    tags: Dict[str, str] = {}
    for phase in ordered:
        own = hashlib.sha256(f"{phase.id}:{_canonical(await phase.settings(cat))}".encode()).hexdigest()
        folded = "|".join(f"{dep}={tags[dep]}" for dep in sorted(phase.depends_on))
        tags[phase.id] = hashlib.sha256(f"{own}|{folded}".encode()).hexdigest()
    return tags


async def service_settings(key_id: str, category: str) -> Dict:
    """The material settings of a configured service: its name and its values, without the credentials (they are
    encrypted again on every save, so they would look changed)."""
    setting = await crud_settings.get_settings_by_category(key_id, category)
    if not setting:
        return {"name": None}
    value = setting.get("value") or {}
    return {
        "name": setting.get("name"),
        "value": {k: v for k, v in value.items() if not any(k.endswith(suffix) for suffix in SUFFIX_TO_CRYPT)},
    }


class ParsingChunkingPhase(IngestionPhase):
    id = PHASE_PARSING_CHUNKING
    label = "Parsing and chunking"

    def __init__(self, rabbit_hole):
        self.rabbit_hole = rabbit_hole

    async def settings(self, cat) -> Any:
        return await service_settings(cat.agent_key, "chunker")

    async def process(self, source: IngestionSource, cat) -> PhaseResult:
        return await self.rabbit_hole.run_parsing_chunking(source, cat)

    async def delete(self, source: IngestionSource, cat) -> None:
        await self.rabbit_hole.delete_parsing_chunking(source, cat)


class EmbeddingPhase(IngestionPhase):
    id = PHASE_EMBEDDING
    label = "Computing embeddings"
    depends_on = (PHASE_PARSING_CHUNKING,)

    def __init__(self, rabbit_hole):
        self.rabbit_hole = rabbit_hole

    async def settings(self, cat) -> Any:
        return await service_settings(DEFAULT_SYSTEM_KEY, "embedder")

    async def process(self, source: IngestionSource, cat) -> PhaseResult:
        return await self.rabbit_hole.run_embedding(source, cat)

    async def delete(self, source: IngestionSource, cat) -> None:
        await self.rabbit_hole.delete_embedding(source, cat)
