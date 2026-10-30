"""Host-side fixtures for the CATALOG integration tests.

The plugins under test are not installed in the host, so the tests need them on
the host's core-plugins path to let the REAL plugin manager discover and import
them. A temporary symlink is created for each missing plugin before the app
fixtures boot and removed afterwards (see AGENTS.md plugin conventions: no test
may leave state behind). Links that already exist in the environment are left
untouched — only the ones created here are removed.

``cat_alog`` provides the catalogue card; ``cat_efficient_ingestion`` (with its
``cat_multimodal_ingestion`` dependency) provides the phase machine used by the
efficient mode. ``test_dual_mode.py`` additionally exposes MyGRAPH under
``cat/plugins/cat_graphrag`` for the ``NEXT`` chunk edges.
"""

import os
import pytest

CORE_PLUGINS_DIR = os.path.join("cat", "core_plugins")

#: plugin folder id -> source folder in the worktree
PLUGIN_LINKS = {
    "cat_alog": "/root/git/plugins/cat_alog",
    "cat_efficient_ingestion": "/root/git/plugins/cat-efficient-ingestion",
    "cat_multimodal_ingestion": "/root/git/plugins/cat-multimodal-ingestion",
}

#: links created by this test session; must be gone at the end (cleanup receipt)
CREATED_LINKS: list[str] = []


def created_links() -> list[str]:
    return list(CREATED_LINKS)


@pytest.fixture
def catalog_plugin_symlinks():
    for plugin_id, target in PLUGIN_LINKS.items():
        link = os.path.join(CORE_PLUGINS_DIR, plugin_id)
        if not os.path.lexists(link):
            os.symlink(target, link)
            CREATED_LINKS.append(link)
    try:
        yield
    finally:
        for link in created_links():
            if os.path.lexists(link):
                os.remove(link)
            if link in CREATED_LINKS:
                CREATED_LINKS.remove(link)
