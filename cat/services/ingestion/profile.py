"""The ingestion profile bundle.

One :class:`IngestionProfile` groups the five replaceable parts of the ingestion
for a single configuration: the pipeline, the state store, the recovery, the
locks and the tag resolver. The configuration class builds it
(:meth:`BaseIngestionConfiguration.profile`) and the engine carries it, so a
call site resolves the parts from ONE place instead of importing core
implementations directly.

Instances are per engine, built from the configuration class: never a module
singleton. Import-safe: importing this module performs no Redis, filesystem or
network access.
"""

from dataclasses import dataclass

from cat.services.ingestion.locks import IngestionLockManager
from cat.services.ingestion.pipeline import IngestionPipeline
from cat.services.ingestion.recovery import IngestionRecovery
from cat.services.ingestion.state import IngestionStateStore
from cat.services.ingestion.tags import IngestionTagResolver


@dataclass(frozen=True)
class IngestionProfile:
    """The five ingestion parts, instantiated for one configuration."""

    pipeline: IngestionPipeline
    state_store: IngestionStateStore
    recovery: IngestionRecovery
    locks: IngestionLockManager
    tags: IngestionTagResolver
