import pytest
from pydantic import ValidationError

from cat.auth.permissions import get_full_permissions
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.env import get_env
from cat.routes.users import UserBase, UserUpdate
from cat.utils import sanitize_permissions

from tests.utils import (
    create_new_user,
    check_user_fields,
    get_client_admin_headers,
    new_user_password,
)


def test_validation_errors():
    with pytest.raises(ValidationError):
        UserBase(username="Alice", permissions={})

    admin = UserUpdate(username="Alice")
    assert isinstance(admin, UserUpdate)
    assert admin.username == "Alice"

    with pytest.raises(ValidationError):
        UserUpdate(username="Alice", permissions={"READ": []})
    with pytest.raises(ValidationError):
        UserUpdate(username="Alice", permissions={"CHESHIRE_CAT": ["WRITE", "WRONG"]})


async def test_create_admin(client):
    permissions = get_full_permissions()

    # create admin
    data = await create_new_user(client, headers=(await get_client_admin_headers(client)), permissions=permissions)

    # assertions on admin structure
    check_user_fields(data)

    assert data["username"] == "Alice"
    assert len(data["permissions"]) == len(permissions) - 1


async def test_cannot_create_duplicate_admin(client):
    permissions = get_full_permissions()

    # create admin
    data = await create_new_user(client, headers=await get_client_admin_headers(client), permissions=permissions)

    # create admin with the same username
    response = await client.post(
        "/users/", json={"username": data["username"], "password": "ecilA"}, headers=await get_client_admin_headers(client)
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Cannot duplicate user"


async def test_get_admins(client):
    permissions = get_full_permissions()

    # get list of admins
    response = await client.get("/users/", headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1

    # create admin
    await create_new_user(client, headers=await get_client_admin_headers(client), permissions=permissions)

    # get the updated list of admins
    response = await client.get("/users/", headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    # check admins integrity and values
    for idx, d in enumerate(data):
        check_user_fields(d)
        assert d["username"] in ["admin", "Alice"]
        assert len(d["permissions"]) == len(permissions) - 1


async def test_get_admin(client):
    permissions = get_full_permissions()

    # get unexisting admin
    response = await client.get("/users/wrong_admin_id", headers=await get_client_admin_headers(client))
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # create admin and obtain id
    admin_id = (await create_new_user(
        client, headers=await get_client_admin_headers(client), permissions=permissions,
    ))["id"]

    # get specific existing admin
    response = await client.get(f"/users/{admin_id}", headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()

    # check admin integrity and values
    check_user_fields(data)
    assert data["username"] == "Alice"
    assert len(data["permissions"]) == len(permissions) - 1


async def test_update_admin(client):
    permissions = get_full_permissions()

    # update unexisting admin
    response = await client.put(
        "/users/non_existent_id", json={"username": "Red Queen"}, headers=await get_client_admin_headers(client)
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # create admin and obtain id
    admin_id = (await create_new_user(
        client, headers=await get_client_admin_headers(client), permissions=permissions,
    ))["id"]

    # update unexisting attribute (bad request)
    updated_admin = {"username": "Alice", "something": 42}
    response = await client.put(f"/users/{admin_id}", json=updated_admin, headers=await get_client_admin_headers(client))
    assert response.status_code == 422

    response = await client.get(f"/users/{admin_id}", headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice"
    assert len(data["permissions"]) == len(permissions) - 1

    # update password
    updated_admin = {"username": data["username"], "password": "12345", "permissions": data["permissions"]}
    response = await client.put(f"/users/{admin_id}", json=updated_admin, headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice"
    assert len(data["permissions"]) == len(permissions) - 1
    assert "password" not in data # api will not send passwords around

    # change username
    updated_admin = {"username": "Alice2", "permissions": data["permissions"]}
    response = await client.put(f"/users/{admin_id}", json=updated_admin, headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice2"
    assert len(data["permissions"]) == len(permissions) - 1

    # change permissions
    updated_admin = {"username": data["username"], "permissions": {"EMBEDDER": ["READ"]}}
    response = await client.put(f"/users/{admin_id}", json=updated_admin, headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice2"
    assert data["permissions"] == {"EMBEDDER": ["READ"]}

    # change username and permissions
    updated_admin = {"username": "Alice3", "permissions": {"EMBEDDER": ["WRITE"]}}
    response = await client.put(f"/users/{admin_id}", json=updated_admin, headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice3"
    assert data["permissions"] == {"EMBEDDER": ["WRITE"]}

    # get list of admins
    response = await client.get("/users/", headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for d in data:
        check_user_fields(d)
        assert d["username"] in ["admin", "Alice3"]
        if d["username"] == "Alice3":
            assert d["permissions"] == {"EMBEDDER": ["WRITE"]}
        else:
            assert d["permissions"] == sanitize_permissions(get_full_permissions(), DEFAULT_SYSTEM_KEY)


async def test_delete_admin(client):
    permissions = get_full_permissions()

    # delete not existing admin
    response = await client.delete("/users/non_existent_id", headers=await get_client_admin_headers(client))
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # create admin and obtain id
    admin_id = (await create_new_user(
        client, headers=await get_client_admin_headers(client), permissions=permissions,
    ))["id"]

    # delete admin
    response = await client.delete(f"/users/{admin_id}", headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == admin_id

    # check that the admin is not in the db anymore
    response = await client.get(f"/users/{admin_id}", headers=await get_client_admin_headers(client))
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # check admin is no more in the list of admins
    response = await client.get("/users/", headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["username"] == "admin"


# note: using secure client (api key set both for http and ws)
async def test_no_access_if_api_keys_active(secure_client):
    # create admin (forbidden)
    response = await secure_client.post(
        "/users/",
        json={"username": "Alice", "password": new_user_password},
    )
    assert response.status_code == 401

    # read admins (forbidden)
    response = await secure_client.get("/users/")
    assert response.status_code == 401

    # edit admin (forbidden)
    response = await secure_client.put(
        "/users/non_existent_id", # it does not exist, but request should be blocked before the check
        json={"username": "Alice"},
    )
    assert response.status_code == 401

    # check the default list giving the correct CAT_API_KEY
    headers = {"Authorization": f"Bearer {get_env('CAT_API_KEY')}"}
    response = await secure_client.get("/users/", headers=headers)
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["username"] == "admin"
