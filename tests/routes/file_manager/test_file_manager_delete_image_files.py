import os
import types

from cat.routes.file_manager import delete_file


async def test_delete_file_fires_image_cascade_hook(monkeypatch):
    """Deleting a source file must fire ``before_file_manager_file_delete`` so a
    plugin can cascade-remove its extracted-image files.

    The core only wires the hook BEFORE the memory points are deleted; the actual
    image-file cascade lives in the plugin that answers the hook.
    """
    removed_files = []
    deleted_points = []
    hook_calls = []

    class FakeRabbitHole:
        async def delete_source(self, cat, name, chat_id=None, wait=None, before=None):
            result = await before()  # holding the lock of the source
            deleted_points.append((name, chat_id))
            return result

    class FakeFileManager:
        def remove_file(self, file_path):
            removed_files.append(file_path)
            return True

    class FakePluginManager:
        async def execute_hook(self, name, *args, caller=None):
            hook_calls.append((name, args))

    fake_cheshire_cat = types.SimpleNamespace(
        agent_key="agent",
        file_manager=FakeFileManager(),
        plugin_manager=FakePluginManager(),
    )
    fake_info = types.SimpleNamespace(
        cheshire_cat=fake_cheshire_cat, stray_cat=None, lizard=types.SimpleNamespace(rabbit_hole=FakeRabbitHole()),
    )

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
    # the source (its points among what its phases produced) is removed from the memory after the cascade hook
    assert deleted_points == [("test.txt", None)]
