"""Shared reconcile (GC) logic for the ``ingestion_status`` core plugin.

Both the reconcile endpoint and the startup GC sweep use this to
purge status entries whose source is absent from the canonical lists:

- files: the file must still exist on disk (via the file manager);
- urls: the URL must still be derivable from the vector store (web points);
- chat-scoped entries: the conversation must still exist.

The helper accepts an optional ``ccat`` so a caller that already holds a
CheshireCat (e.g. a route handler) avoids a redundant re-fetch.
"""
import os
from typing import Any, Dict, List, Optional

from cat.core_plugins.ingestion_status.registry import delete_status, list_statuses
from cat.db.cruds import conversations as crud_conversations
from cat.looking_glass.bill_the_lizard import BillTheLizard
from cat.log import log
from cat.services.memory.models import VectorMemoryType


async def reconcile_agent(
    agent_id: str,
    chat_id: Optional[str] = None,
    ccat: Any = None,
) -> List[Dict]:
    """Purge status entries whose source is absent from the canonical lists.

    Carve-outs — never purged by this canonical-source reconcile:
    - in-flight entries (uploaded/downloading/downloaded/processing): they are
      legitimately not yet in the file manager / web points (these populate as
      the pipeline advances), so purging on first read would hide the queue
      from the dashboard;
    - error entries: a failed upload never lands in the canonical lists, so an
      error would otherwise vanish immediately; the teacher dismisses it via
      DELETE /ingestion/status or re-uploads.

    Only terminal COMPLETED entries whose source has genuinely vanished (file
    removed, URL with no web points, conversation gone) are purged.

    Args:
        agent_id: The agent (chatbot) id.
        chat_id: Restrict to one conversation scope when given.
        ccat: An optional CheshireCat instance; when omitted it is fetched via
            ``BillTheLizard``.

    Returns:
        The list of purged status docs.
    """
    if ccat is None:
        ccat = await BillTheLizard().get_cheshire_cat(agent_id)
    if ccat is None:
        return []

    entries = await list_statuses(agent_id, chat_id=chat_id)
    never_purge = {"uploaded", "downloading", "downloaded", "processing", "error"}
    purged: List[Dict] = []
    for entry in entries:
        scope = entry.get("scope")
        source = entry.get("source")
        type_ = entry.get("type")
        if not source:
            continue

        # in-flight / error carve-out: never purge via the canonical reconcile
        if entry.get("status") in never_purge:
            continue

        # chat-scoped entries: the conversation must still exist
        if scope != "agent":
            if not await crud_conversations.get_user_id_from_conversation_keys(agent_id, str(scope)):
                await delete_status(agent_id, str(scope), source)
                purged.append(entry)
                continue

        # file entries: the file must still exist on disk
        if type_ == "file":
            path = agent_id
            if scope != "agent":
                path = os.path.join(path, str(scope))
            try:
                names = {f.name for f in ccat.file_manager.list_files(path)}
            except Exception as e:
                log.error(f"Ingestion reconcile: failed to list files for {path}: {e}")
                continue
            if source not in names:
                await delete_status(agent_id, str(scope), source)
                purged.append(entry)
            continue

        # url entries: the URL must still be derivable from the vector store
        if type_ == "url":
            collection = str(
                VectorMemoryType.DECLARATIVE if scope == "agent" else VectorMemoryType.EPISODIC
            )
            try:
                points, _ = await ccat.vector_memory_handler.get_all_tenant_points_from_web(collection)
            except Exception as e:
                log.error(f"Ingestion reconcile: failed to read web points for {agent_id}: {e}")
                continue
            urls = {(p.payload or {}).get("metadata", {}).get("source") for p in points}
            if source not in urls:
                await delete_status(agent_id, str(scope), source)
                purged.append(entry)

    return purged