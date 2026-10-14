from copy import deepcopy

from typing_extensions import get_args

from cat.core_plugins.webhooks.webhooks import WEBHOOK_EVENT
import cat.core_plugins.webhooks.crud as crud_webhook

from tests.utils import agent_id


async def test_webhooks_events(secure_client, secure_client_headers, cheshire_cat):
    events = get_args(WEBHOOK_EVENT)

    res = await secure_client.get("/webhooks/events", headers=secure_client_headers)
    assert res.status_code == 200

    json_response = res.json()
    assert len(json_response) == len(events)

    for event in events:
        assert event in json_response


async def test_webhooks_create_no_auth(client, cheshire_cat):
    payload = {
        "url": "https://example.com",
        "event": "knowledge_source_loaded",
        "secret": "secret",
    }

    res = await client.post("/webhooks/", json=payload)
    assert res.status_code == 401

    res_db = await crud_webhook.get_webhooks(agent_id=agent_id, event=payload["event"])
    assert res_db is None


async def test_webhooks_create_wrong_payloads(secure_client, secure_client_headers, cheshire_cat):
    payload_no_secret = {
        "url": "https://example.com",
        "event": "plugin_installed",
    }
    res = await secure_client.post("/webhooks/", json=payload_no_secret, headers=secure_client_headers)
    assert res.status_code == 422

    payload_wrong_event = {
        "url": "https://example.com",
        "event": "wrong_event",
        "secret": "secret",
    }
    res = await secure_client.post("/webhooks/", json=payload_wrong_event, headers=secure_client_headers)
    assert res.status_code == 422

    payload_null_header_key = {
        "url": "https://example.com",
        "event": "plugin_installed",
        "secret": "secret",
        "header_key": None,
    }
    res = await secure_client.post("/webhooks/", json=payload_null_header_key, headers=secure_client_headers)
    assert res.status_code == 422


async def test_webhooks_create(secure_client, secure_client_headers, cheshire_cat):
    payload = {
        "url": "https://example.com",
        "event": "knowledge_source_loaded",
        "secret": "secret",
    }
    res = await secure_client.post("/webhooks/", json=payload, headers=secure_client_headers)
    assert res.status_code == 200

    another_payload = {
        "url": "https://example1.com",
        "event": "knowledge_source_loaded",
        "secret": "secret",
    }
    res = await secure_client.post("/webhooks/", json=another_payload, headers=secure_client_headers)
    assert res.status_code == 200

    res_db = await crud_webhook.get_webhooks(agent_id=agent_id, event=payload["event"])
    registered_urls = [item["url"] for item in res_db]
    assert res_db is not None
    assert len(res_db) == 2
    assert payload["url"] in registered_urls
    assert another_payload["url"] in registered_urls


async def test_webhooks_create_once(secure_client, secure_client_headers, cheshire_cat):
    payload = {
        "url": "https://example.com",
        "event": "knowledge_source_loaded",
        "secret": "secret",
    }
    res = await secure_client.post("/webhooks/", json=payload, headers=secure_client_headers)
    assert res.status_code == 200

    res = await secure_client.post("/webhooks/", json=payload, headers=secure_client_headers)
    assert res.status_code == 200

    res_db = await crud_webhook.get_webhooks(agent_id=agent_id, event=payload["event"])
    assert res_db is not None
    assert len(res_db) == 1


async def test_webhooks_delete(secure_client, secure_client_headers, cheshire_cat):
    await test_webhooks_create(secure_client, secure_client_headers, cheshire_cat)

    payload = {
        "url": "https://example.com",
        "event": "knowledge_source_loaded",
        "secret": "secret",
    }

    res = await secure_client.request(
        "DELETE",
        "/webhooks/",
        headers=secure_client_headers,
        json=payload,
    )
    assert res.status_code == 200


async def test_manage_a_webhook_with_missing_header(secure_client, secure_client_headers, cheshire_cat):
    payload = {
        "url": "https://example.com",
        "event": "knowledge_source_loaded",
        "secret": "secret",
    }

    expected_error_msg = "Cannot register / unregister the webhook without a CheshireCat instance"

    headers = deepcopy(secure_client_headers)
    del headers["X-Agent-ID"]
    res = await secure_client.post("/webhooks/", json=payload, headers=headers)
    json_response = res.json()
    assert res.status_code == 500
    assert json_response["detail"] == expected_error_msg

    await test_webhooks_create_once(secure_client, secure_client_headers, cheshire_cat)

    res = await secure_client.request("DELETE", "/webhooks/", json=payload, headers=headers)
    json_response = res.json()
    assert res.status_code == 500
    assert json_response["detail"] == expected_error_msg
