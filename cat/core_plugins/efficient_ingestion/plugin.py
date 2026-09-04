"""Efficient ingestion — replaceable ingestion engine.

Registers ``EfficientIngestionConfiguration`` through the
``factory_allowed_ingestions`` hook: the core resolves the engine through the
ServiceFactory (``ingestion`` category) when the embedder changes and, when
this plugin is present, prefers our efficient implementation.

The plugin is system-level: factory entries and the engine selection live in
the global ``system:agent`` store under the ``ingestion`` category (see
``configs.py`` and the plugin's settings endpoints).

It also owns the token-budget chunk split (moved from the core
``RabbitHole._split_oversized``, a MyCAT-only addition): the
``finalize_oversized_chunks`` and ``before_rabbithole_stores_documents`` hooks
resolve the active embedder and split oversized chunks into budget-compliant
sub-chunks. With this plugin disabled the core returns to upstream parity (no
token-budget split).
"""

from typing import List

from langchain_core.documents import Document

from cat import hook
from cat.core_plugins.efficient_ingestion.configs import EfficientIngestionConfiguration
from cat.core_plugins.efficient_ingestion.split import split_oversized


@hook(priority=0)
def factory_allowed_ingestions(allowed, lizard):
    """Register the efficient ingestion engine as a factory option."""
    return list(allowed) + [EfficientIngestionConfiguration]


@hook(priority=0)
async def finalize_oversized_chunks(docs: List[Document], cat) -> List[Document]:
    """Split oversized chunks so none exceeds the active embedder's input limit.

    Fired by the core RabbitHole after chunking/splitting: resolves the active
    embedder and applies the token-budget split (pure fold, metadata carried
    forward). No-op-compatible when the embedder has no ``max_input_tokens``.
    """
    embedder = await cat.lizard.embedder()
    return split_oversized(docs, embedder)


@hook(priority=100)
async def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    """Apply the token-budget split on the store path of the core engine.

    Fired by ``RabbitHole.store_documents`` before the embed/store loop (e.g.
    also covering documents added by other post-chunking hooks): split oversized
    chunks so the 1:1 embed/store pairing is never broken.
    """
    embedder = await cat.lizard.embedder()
    return split_oversized(docs, embedder)
