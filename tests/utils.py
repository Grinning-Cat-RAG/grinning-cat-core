import shutil
import time
import uuid
import random
from typing import Dict
from urllib.parse import urlencode
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport
import httpx

from cat.auth.permissions import get_base_permissions, AuthResource, AuthPermission
from cat.db.cruds import users as crud_users
from cat.env import get_env
from cat.services.memory.models import VectorMemoryType

agent_id = "agent_test"
chat_id = "a1157e2d-ca3d-4f21-a4a5-b57a64dd01c9"
api_key = "meow_http"
jwt_secret = "meow_jwt"

new_user_password = "wandering_in_wonderland"
mock_plugin_path = "tests/mocks/mock_plugin/"

fake_timestamp = 1705855981


async def send_file(file_name, content_type, client, headers, payload = None, ch_id = None):
    file_path = f"tests/mocks/{file_name}"

    url = "/rabbithole/" if not ch_id else f"/rabbithole/{ch_id}"
    with open(file_path, "rb") as f:
        files = {"file": (file_name, f, content_type)}
        response = await client.post(url, files=files, data=payload, headers=headers)
    return response, file_path


# utility function to communicate with the cat via websocket
async def send_websocket_message(msg, client, token, query_params=None, ch_id=None):
    url = f"/ws/{agent_id}" if not ch_id else f"/ws/{agent_id}/{ch_id}"
    if query_params:
        url += "?" + urlencode(query_params)

    app = client._fastapi_test_app
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    async with httpx.AsyncClient(transport=ASGIWebSocketTransport(app=app)) as ws_client:
        async with aconnect_ws(f"http://server{url}", ws_client, headers=headers) as websocket:
            await websocket.send_json(msg)
            response = await websocket.receive_json()
            return response


# utility to send n messages via chat
async def send_n_websocket_messages(num_messages, client, image = None, ch_id = None, query_params = None):
    responses = []

    url = f"/ws/{agent_id}" if not ch_id else f"/ws/{agent_id}/{ch_id}"
    if query_params:
        url += "?" + urlencode(query_params)

    app = client._fastapi_test_app
    async with httpx.AsyncClient(transport=ASGIWebSocketTransport(app=app)) as ws_client:
        async with aconnect_ws(
                f"http://server{url}", ws_client, headers={"Authorization": f"Bearer {api_key}"},
        ) as websocket:
            for m in range(num_messages):
                message = {"text": f"Red Queen {m}"}
                if image:
                    message["image"] = image
                await websocket.send_json(message)
                response = await websocket.receive_json()
                responses.append(response)

    return responses


def key_in_json(key, json):
    return key in json.keys()


# create a plugin zip out of the mock plugin folder.
# - Used to test plugin upload.
# - zip can be created flat (plugin files in root dir) or nested (plugin files in zipped folder)
def create_mock_plugin_zip(flat: bool, plugin_id = "mock_plugin"):
    if flat:
        root_dir = f"tests/mocks/{plugin_id}"
        base_dir = "./"
    else:
        root_dir = "tests/mocks/"
        base_dir = plugin_id

    return shutil.make_archive(
        base_name=f"tests/mocks/{plugin_id}",
        format="zip",
        root_dir=root_dir,
        base_dir=base_dir,
    )


# utility to retrieve declarative memory contents
async def get_memory_contents(client, headers = None, collection: VectorMemoryType | None = VectorMemoryType.DECLARATIVE):
    headers = headers or {} | {"X-Agent-ID": agent_id}
    params = {"text": "Something"}
    response = await client.get("/memory/recall", params=params, headers=headers)
    assert response.status_code == 200
    json = response.json()
    memories = json["vectors"]["collections"][str(collection)]
    return memories


# utility to get collections and point count from `GET /memory/collections` in a simpler format
async def get_collections_names_and_point_count(client, headers = None):
    headers = headers or {} | {"X-Agent-ID": agent_id}
    response = await client.get("/memory/collections", headers=headers)
    json = response.json()
    assert response.status_code == 200
    collections_n_points = {c["name"]: c["vectors_count"] for c in json["collections"]}
    return collections_n_points


async def create_new_user(client, username = "Alice", headers = None, permissions = None, password = None):
    new_user = {
        "username": username,
        "password": password or new_user_password,
        "permissions": permissions or {str(AuthResource.CHAT): [str(AuthPermission.WRITE)]}
    }
    response = await client.post("/users/", json=new_user, headers=headers)
    assert response.status_code == 200
    return response.json()


def check_user_fields(u):
    for k in ["id", "username", "permissions"]:
        assert k in u.keys()

    assert isinstance(u["username"], str)
    assert isinstance(u["permissions"], dict)
    try:
        # Attempt to create a UUID object from the string to validate it
        uuid_obj = uuid.UUID(u["id"], version=4)
        assert str(uuid_obj) == u["id"]
    except ValueError:
        # If a ValueError is raised, the UUID string is invalid
        assert False, "Not a UUID"


async def get_fake_memory_export(client, embedder_name = "DumbEmbedder", dim = 2367):
    user = await create_new_user(
        client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )

    user = await crud_users.get_user_by_username(agent_id, user["username"])
    return {
        "embedder": embedder_name,
        "collections": {
            str(VectorMemoryType.DECLARATIVE): [
                {
                    "page_content": "test_memory",
                    "metadata": {"source": user["id"], "when": time.time()},
                    "id": str(uuid.uuid4()),
                    "vector": [random.random() for _ in range(dim)],
                }
            ]
        },
    }


async def get_client_admin_headers(client):
    creds = {
        "username": "admin",
        "password": get_env("CAT_ADMIN_DEFAULT_PASSWORD"),
    }

    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 200
    token = res.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


async def just_installed_plugin(client, headers, activate = False, plugin_id = "mock_plugin", expected_status_code = 200):
    # create zip file with a plugin
    zip_path = create_mock_plugin_zip(flat=True, plugin_id=plugin_id)
    zip_file_name = zip_path.split("/")[-1]  # mock_plugin.zip in tests/mocks folder

    # upload plugin via endpoint
    with open(zip_path, "rb") as f:
        response = await client.post(
            "/plugins/install/upload",
            files={"file": (zip_file_name, f, "application/zip")},
            headers=headers
        )

    # request was processed
    assert response.status_code == expected_status_code
    if expected_status_code == 200:
        assert response.json()["filename"] == zip_file_name

        if activate:
            # mock_plugin is installed but not enabled yet
            response = await client.put(f"/plugins/toggle/{plugin_id}", headers=headers)
            # request was processed
            assert response.status_code == 200

    return response


# utility to make http requests with some headers
async def http_message(client, message: Dict, headers = None):
    response = await client.post("/message", headers=headers, json=message)
    return response.status_code, response.json()


def agent_core_plugins(plugin_manager) -> list:
    """Core plugins an agent can see and manage.

    System (untoggling) plugins are always-on and off-limits at an agent level: they are
    hidden from `/plugins/` and `/plugins/settings`, and their per-plugin routes answer 404.
    They are still fully visible on the `/plugins/system/*` routes.
    """
    return [p for p in plugin_manager.get_core_plugins_ids if not plugin_manager.is_system_plugin(p)]
