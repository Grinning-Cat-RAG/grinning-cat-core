import os
import types

from cat.routes.file_manager import delete_file


async def test_delete_file_fires_image_cascade_hook(monkeypatch):
    """Deleting a source file must fire ``before_file_manager_file_delete`` so the
    multimodal_ingestion plugin cascade-removes its extracted-image files.

    The core only wires the hook BEFORE the memory points are deleted; the actual
    image-file cascade lives in the plugin (covered in test_multimodal_ingestion.py).
    """
    removed_files = []
    deleted_points = []
    hook_calls = []

    class FakeVectorMemoryHandler:
        async def delete_tenant_points(self, collection_name, metadata):
            deleted_points.append((collection_name, metadata))

    handler = FakeVectorMemoryHandler()

    class FakeFileManager:
        def remove_file(self, file_path):
            removed_files.append(file_path)
            return True

    class FakePluginManager:
        async def execute_hook(self, name, *args, caller=None):
            hook_calls.append((name, args))

    fake_cheshire_cat = types.SimpleNamespace(
        agent_key="agent",
        vector_memory_handler=handler,
        file_manager=FakeFileManager(),
        plugin_manager=FakePluginManager(),
    )
    fake_info = types.SimpleNamespace(cheshire_cat=fake_cheshire_cat, stray_cat=None)

    # fixed (path, collection_id, metadata) so no real storage/DB is touched
    monkeypatch.setattr(
        "cat.routes.file_manager.get_from_info",
        lambda info: ("path", "declarative", {}),
    )

    res = await delete_file("test.txt", info=fake_info)

    assert res.deleted is True
    # the source file is removed
    assert removed_files == [os.path.join("path", "test.txt")]
    # the image-cascade hook fires BEFORE the points are deleted, with the
    # source name and the agent scope ("agent" since the path has no chat part)
    assert hook_calls == [
        ("before_file_manager_file_delete", ("test.txt", "agent")),
        ("after_file_manager_file_deleted", ("test.txt", "agent")),
    ]
    # the points are deleted after the cascade hook
    assert deleted_points == [("declarative", {"source": "test.txt"})]
