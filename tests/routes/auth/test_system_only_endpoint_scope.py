"""System-only plugins serve every agent, like the embedder does.

Their endpoints are instance-level services: a request carrying an agent id
must reach the handler (which scopes the response to that agent), while the
guard keeps rejecting agent-level plugin endpoints that are not active for
the picked agent.
"""
from tests.utils import api_key, just_installed_plugin


async def test_system_only_endpoint_reachable_with_agent(
    secure_client, secure_client_headers, cheshire_cat, lizard
):
    # ``cat_efficient_ingestion`` is system-only: its endpoint lives on the system plugin manager only.
    assert lizard.is_custom_endpoint("/ingestion/status", method="GET")
    assert lizard.is_system_custom_endpoint("/ingestion/status", method="GET")
    assert not cheshire_cat.has_custom_endpoint("/ingestion/status", method="GET")

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)

    assert response.status_code == 200
    assert response.json() == []


async def test_system_only_endpoint_reachable_in_system_scope(secure_client, cheshire_cat):
    response = await secure_client.get("/ingestion/status", headers={"Authorization": f"Bearer {api_key}"})

    assert response.status_code == 200
    assert response.json() == []


async def test_agent_level_endpoint_inactive_for_agent_still_returns_404(
    secure_client, secure_client_headers, lizard, cheshire_cat
):
    # activate an agent-level plugin for ``agent_id`` so its endpoint is registered on the app
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)

    # another agent exists but does not carry the plugin
    other_agent_id = "agent_without_mock_plugin"
    await lizard.create_cheshire_cat(other_agent_id)
    other_cat = await lizard.get_cheshire_cat(other_agent_id)

    assert lizard.is_custom_endpoint("/tests/crud", method="GET")
    assert not lizard.is_system_custom_endpoint("/tests/crud", method="GET")
    assert not other_cat.has_custom_endpoint("/tests/crud", method="GET")

    response = await secure_client.get(
        "/tests/crud", headers={**secure_client_headers, "X-Agent-ID": other_agent_id},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Not Found"
