import asyncio
import io

from cat.core_plugins.ingestion_status.registry import get_status, set_status
from cat.services.memory.models import VectorMemoryType

from tests.utils import agent_id, send_file, get_memory_contents


async def check_file_deleted(secure_client, secure_client_headers, collection: VectorMemoryType, ch_id = None):
    # set file_manager_name = "LocalFileManagerConfig"
    file_manager_name = "LocalFileManagerConfig"
    response = await secure_client.put(
        f"/file_manager/settings/{file_manager_name}", headers=secure_client_headers, json={},
    )
    assert response.status_code == 200

    content_type = "application/pdf"
    response, file_path = await send_file(
        "sample.pdf", content_type, secure_client, secure_client_headers, ch_id=ch_id
    )
    assert response.status_code == 200

    headers = secure_client_headers
    if ch_id:
        headers |= {"X-Chat-ID": ch_id}

    # seed an ingestion-status row for the file: deleting the file must also
    # delete its status row, so a stale processing/completed row cannot linger
    scope = str(ch_id) if ch_id else "agent"
    await set_status(agent_id, scope, "sample.pdf", type_="file", status="completed")

    # check memory contents
    memories = await get_memory_contents(secure_client, headers, collection)
    assert len(memories) > 0

    # check that the file exists in the list of files
    res = await secure_client.request("GET", "/file_manager/", headers=headers)
    assert res.status_code == 200
    json = res.json()
    files = json["files"]
    assert len(files) == 1
    assert any(f["name"] == "sample.pdf" for f in files)

    # delete first document
    res = await secure_client.request("DELETE", "/file_manager/files/sample.pdf", headers=headers)
    # check memory contents
    assert res.status_code == 200
    json = res.json()
    assert isinstance(json["deleted"], bool)
    memories = await get_memory_contents(secure_client, headers, collection)
    assert len(memories) == 0

    # the ingestion-status row was removed along with the file
    assert await get_status(agent_id, scope, "sample.pdf") is None

    # check that the file does not exist anymore in the list of files
    res = await secure_client.request("GET", "/file_manager/", headers=headers)
    assert res.status_code == 200
    json = res.json()
    files = json["files"]
    assert len(files) == 0


async def check_files_deleted(secure_client, secure_client_headers, collection: VectorMemoryType, ch_id = None):
    # set file_manager_name = "LocalFileManagerConfig"
    file_manager_name = "LocalFileManagerConfig"
    response = await secure_client.put(f"/file_manager/settings/{file_manager_name}", headers=secure_client_headers, json={})
    assert response.status_code == 200

    content_type = "application/pdf"
    response, file_path = await send_file("sample.pdf", content_type, secure_client, secure_client_headers, ch_id=ch_id)
    assert response.status_code == 200

    # check memory contents
    headers = secure_client_headers
    if ch_id:
        headers |= {"X-Chat-ID": ch_id}

    # upload another document
    with open(file_path, "rb") as f:
        files = {"file": ("sample2.pdf", f, content_type)}
        response = await secure_client.post("/rabbithole/", files=files, headers=headers)
        assert response.status_code == 200

    # check memory contents
    memories = await get_memory_contents(secure_client, headers, collection)
    assert len(memories) > 0

    # check that the files exist in the list of files
    res = await secure_client.request("GET", "/file_manager/", headers=headers)
    assert res.status_code == 200
    json = res.json()
    files = json["files"]
    assert len(files) == 2
    assert any(f["name"] == "sample.pdf" for f in files)
    assert any(f["name"] == "sample2.pdf" for f in files)

    # delete all documents
    res = await secure_client.request("DELETE", "/file_manager/files", headers=headers)
    # check memory contents
    assert res.status_code == 200
    json = res.json()
    assert isinstance(json["deleted"], bool)
    memories = await get_memory_contents(secure_client, headers, collection)
    assert len(memories) == 0

    # check that the files do not exist anymore in the list of files
    res = await secure_client.request("GET", "/file_manager/", headers=headers)
    assert res.status_code == 200
    json = res.json()
    files = json["files"]
    assert len(files) == 0


async def test_file_deleted(secure_client, secure_client_headers, cheshire_cat):
    await check_file_deleted(secure_client, secure_client_headers, VectorMemoryType.DECLARATIVE)


async def test_file_chat_deleted(secure_client, secure_client_headers, stray_no_memory, cheshire_cat):
    await check_file_deleted(secure_client, secure_client_headers, VectorMemoryType.EPISODIC, ch_id=stray_no_memory.id)


async def test_files_deleted(secure_client, secure_client_headers, cheshire_cat):
    await check_files_deleted(secure_client, secure_client_headers, VectorMemoryType.DECLARATIVE)


async def test_files_chat_deleted(secure_client, secure_client_headers, stray_no_memory, cheshire_cat):
    await check_files_deleted(secure_client, secure_client_headers, VectorMemoryType.EPISODIC, ch_id=stray_no_memory.id)


async def test_file_saved_before_processing_completes(
    secure_client, secure_client_headers, cheshire_cat, monkeypatch
):
    """The uploaded file bytes must be on disk BEFORE processing completes.

    `RabbitHole.ingest_file` saves the file right after parsing (before
    `store_documents`), so a container restart can resume reading the bytes
    from disk even if embedding is interrupted. This test stubs
    `store_documents` to be slow and asserts the file is already present for
    the agent while ingestion is still in progress.
    """
    # set file_manager_name = "LocalFileManagerConfig"
    file_manager_name = "LocalFileManagerConfig"
    response = await secure_client.put(
        f"/file_manager/settings/{file_manager_name}", headers=secure_client_headers, json={},
    )
    assert response.status_code == 200

    # refresh the fixture's file manager so it matches the persisted setting
    cheshire_cat.file_manager = await cheshire_cat.service_provider.get_file_manager(
        cheshire_cat.agent_key, cheshire_cat.plugin_manager
    )

    # read the sample file bytes
    file_path = "tests/mocks/sample.pdf"
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # stub store_documents to be slow, so we can observe the file on disk
    # while ingestion is still in progress (before processing completes)
    async def slow_store_documents(*args, **kwargs):
        await asyncio.sleep(1)
        return []

    rabbit_hole = cheshire_cat.rabbit_hole
    monkeypatch.setattr(rabbit_hole, "store_documents", slow_store_documents)

    # start ingestion in the background
    task = asyncio.create_task(
        rabbit_hole.ingest_file(
            cat=cheshire_cat,
            file=io.BytesIO(file_bytes),
            metadata={},
            filename="sample.pdf",
            content_type="application/pdf",
        )
    )

    # wait for the early save to happen (it runs right after _file_to_docs,
    # before store_documents)
    for _ in range(50):
        files = cheshire_cat.file_manager.list_files(cheshire_cat.agent_key)
        if any(f.name == "sample.pdf" for f in files):
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("sample.pdf was not saved to disk during processing")

    # ingestion must still be in progress (store_documents is sleeping)
    assert not task.done(), "ingestion should still be in progress"

    # the bytes on disk must match the uploaded file
    content = cheshire_cat.file_manager.read_file("sample.pdf", cheshire_cat.agent_key)
    assert content == file_bytes

    # let ingestion finish
    await task
