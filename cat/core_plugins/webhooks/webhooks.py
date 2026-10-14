import hashlib
import hmac
import json
from typing import List, Literal, get_args, Dict, Any
import asyncio
import httpx
from pydantic import BaseModel, Field

from cat import (
    hook,
    endpoint,
    AuthorizedInfo,
    check_permissions,
    AuthResource,
    AuthPermission,
    PointStruct,
    log,
    StrayCat,
)
import cat.core_plugins.webhooks.crud as crud_webhook
from cat.services.string_crypto import StringCrypto


WEBHOOK_EVENT = Literal[
    "knowledge_source_loaded",
    "plugin_installed",
    "plugin_uninstalled",
    "embedder_updated",
    "knowledge_source_files_transferred",
    "vector_memory_files_transferred",
]

WEBHOOK_CHESHIRECAT_EVENTS = [
    "knowledge_source_loaded",
    "knowledge_source_files_transferred",
    "vector_memory_files_transferred",
]

# Seconds a webhook has to answer: hooks await the notifications, so an unresponsive endpoint must not hold them
WEBHOOK_TIMEOUT_SECONDS = 10

crypto = StringCrypto()


class WebhookResponse(BaseModel):
    url: str
    event: WEBHOOK_EVENT
    # the header carrying the signature: without it the receiver could never verify a notification
    header_key: str = Field(default="X-CheshireCat-Signature")


class WebhookPayload(WebhookResponse):
    secret: str


async def trigger_webhook(webhook_data: WebhookPayload, payload: Dict[str, Any]):
    target_url = webhook_data.url
    custom_header_key = webhook_data.header_key

    secret = crypto.decrypt(webhook_data.secret)

    # the receiver verifies the signature against the bytes it gets: sign and send exactly the same body
    body = json.dumps(payload, separators=(',', ':')).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), msg=body, digestmod=hashlib.sha256).hexdigest()

    headers = {"Content-Type": "application/json", custom_header_key: signature}
    # redirects are followed as requests did: 307/308 resend the signed POST, while 301/302/303 turn it into a GET
    # without body, so a webhook URL answering with those never gets the notification
    async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS, follow_redirects=True) as client:
        await client.post(target_url, content=body, headers=headers)


async def trigger_webhooks(webhooks: List[Dict[str, Any]], payload: Dict[str, Any], what: str) -> None:
    """
    Notifies all the webhooks of an event concurrently, so that the caller waits at most one timeout. A failing or
    unresponsive webhook is logged and does not prevent the others from being notified.
    """
    async def notify(webhook: Dict[str, Any]) -> None:
        try:
            await trigger_webhook(WebhookPayload(**webhook), payload)
            log.info(f"Triggered the webhook '{webhook['url']}' {what}")
        except Exception as e:
            log.error(f"Failed to trigger the webhook '{webhook['url']}' {what}: {e}")

    await asyncio.gather(*[notify(webhook) for webhook in webhooks])


def parse_agent_key(info: AuthorizedInfo, webhook: WebhookPayload) -> str:
    if webhook.event not in WEBHOOK_CHESHIRECAT_EVENTS:
        return info.lizard.agent_key
    if not info.cheshire_cat:
        raise ValueError("Cannot register / unregister the webhook without a CheshireCat instance")
    return info.cheshire_cat.agent_key


async def notify_plugin_event_to_webhooks(plugin_id: str, webhooks: List[Dict[str, Any]], success: bool, what: str):
    payload = {"plugin_id": plugin_id, "success": success}
    await trigger_webhooks(webhooks, payload, f"on plugin {what}")


@endpoint.get("/events", tags=["Webhooks"], prefix="/webhooks", response_model=List[str])
async def get_available_events() -> List[str]:
    return list(get_args(WEBHOOK_EVENT))


@endpoint.post("/", tags=["Webhooks"], prefix="/webhooks", response_model=WebhookResponse)
async def register_webhook(
    webhook: WebhookPayload,
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.WRITE),
) -> WebhookResponse:
    global crypto

    agent_id = parse_agent_key(info, webhook)

    settings = webhook.model_dump(exclude={"event"}) | {"secret": crypto.encrypt(webhook.secret)}
    stored_webhook = await crud_webhook.set_webhook(agent_id, webhook.event, settings)
    return WebhookResponse(
        url=stored_webhook["url"],
        event=webhook.event,
        header_key=stored_webhook.get("header_key"),
    )


@endpoint.delete("/", tags=["Webhooks"], prefix="/webhooks")
async def delete_webhook(
    webhook: WebhookPayload,
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.DELETE),
) -> None:
    global crypto

    agent_id = parse_agent_key(info, webhook)

    secret = crypto.encrypt(webhook.secret)
    await crud_webhook.delete_webhook(agent_id, webhook.event, webhook.url, secret)


@hook(priority=0)
async def after_rabbithole_stored_documents(source, stored_points: List[PointStruct], cat) -> None:
    webhooks = await crud_webhook.get_webhooks(cat.agent_key, "knowledge_source_loaded")
    if webhooks is None:
        return

    payload = {
        "agent": cat.agent_key,
        "chat": cat.id if isinstance(cat, StrayCat) else None,
        "source": source,
        "points": [point.payload.get("metadata") for point in stored_points],
        "success": len(stored_points) > 0,
    }

    await trigger_webhooks(webhooks, payload, f"for the agent '{cat.agent_key}' on knowledge source loaded")


@hook(priority=0)
async def lizard_notify_plugin_installation(plugin_id: str, plugin_path: str, lizard) -> None:
    webhooks = await crud_webhook.get_webhooks(lizard.agent_key, "plugin_installed")
    if webhooks is None:
        return

    success = bool(plugin_id and lizard.plugin_manager.plugins.get(plugin_id))
    await notify_plugin_event_to_webhooks(plugin_id, webhooks, success, "installation")


@hook(priority=0)
async def lizard_notify_plugin_uninstallation(plugin_id: str, lizard) -> None:
    webhooks = await crud_webhook.get_webhooks(lizard.agent_key, "plugin_uninstalled")
    if webhooks is None:
        return

    success = lizard.plugin_manager.plugins.get(plugin_id) is None
    await notify_plugin_event_to_webhooks(plugin_id, webhooks, success, "uninstallation")


@hook(priority=0)
async def after_all_cheshire_cats_embedded(success: bool, lizard) -> None:
    webhooks = await crud_webhook.get_webhooks(lizard.agent_key, "embedder_updated")
    if webhooks is None:
        return

    payload = {"success": success}

    await trigger_webhooks(webhooks, payload, "on embedder updated")


@hook(priority=0)
async def after_file_manager_transfer_on_agent(success: bool, cat) -> None:
    webhooks = await crud_webhook.get_webhooks(cat.agent_key, "knowledge_source_files_transferred")
    if webhooks is None:
        return

    payload = {"agent": cat.agent_key, "success": success}

    await trigger_webhooks(
        webhooks, payload, f"for the agent '{cat.agent_key}' on knowledge source files transferred",
    )


@hook(priority=0)
async def after_vector_memory_transfer_on_agent(success: bool, cat) -> None:
    webhooks = await crud_webhook.get_webhooks(cat.agent_key, "vector_memory_files_transferred")
    if webhooks is None:
        return

    payload = {"agent": cat.agent_key, "success": success}

    await trigger_webhooks(
        webhooks, payload, f"for the agent '{cat.agent_key}' on vector memory points transferred",
    )
