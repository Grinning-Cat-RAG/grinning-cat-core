import pytest

from cat.db import crud
from cat.db.database import get_async_db
from tests.utils import agent_id


async def test_destroy_purges_ingestion_status_keys(cheshire_cat):
    # seed ingestion-status registry keys for the agent under test
    agent_key = cheshire_cat.agent_key
    status_key = f"agents:{agent_key}:ingestion:agent:{'a' * 64}"
    await crud.store(status_key, {"status": "uploaded", "source": "doc.pdf"})

    # seed a key for another agent that must survive the purge
    other_agent_key = "agent_other"
    other_status_key = f"agents:{other_agent_key}:ingestion:agent:{'b' * 64}"
    await crud.store(other_status_key, {"status": "completed", "source": "other.pdf"})

    # sanity: both keys exist before destroy
    assert await crud.read(status_key) is not None
    assert await crud.read(other_status_key) is not None

    await cheshire_cat.destroy()

    # the destroyed agent's ingestion keys are gone (purged by the
    # ingestion_status plugin via the after_cheshire_cat_destroy hook)
    assert await crud.read(status_key) is None

    # other agents' keys remain untouched
    assert await crud.read(other_status_key) is not None

    # no stray keys remain under the destroyed agent's ingestion namespace
    db = get_async_db()
    remaining = [k async for k in db.scan_iter(f"agents:{agent_key}:ingestion:*")]
    assert remaining == []