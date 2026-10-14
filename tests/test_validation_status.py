"""
A request that does not match the endpoint's schema is answered with 422 (the status FastAPI documents in OpenAPI),
also when the error comes from a model validator raising ValueError (its context carries the exception object).
"""
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator


class _Body(BaseModel):
    permissions: dict

    @field_validator("permissions")
    @classmethod
    def check(cls, value):
        raise ValueError("Invalid resource: X")  # like cat.routes.routes_utils.validate_permissions


@pytest.fixture
def probe_client(monkeypatch):
    # create_app only checks that these are configured: without the lifespan, nothing is contacted
    monkeypatch.setenv("CAT_JWT_SECRET", "a-test-secret-of-at-least-32-characters")
    monkeypatch.setenv("CAT_REDIS_HOST", "localhost")
    from cat.startup import create_app  # builds the app at import time

    app = create_app()

    @app.post("/__validation_probe")
    async def probe(body: _Body):
        return {}

    return TestClient(app, raise_server_exceptions=False)


def test_a_missing_field_is_answered_with_422(probe_client):
    response = probe_client.post("/__validation_probe", json={})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "permissions"]


def test_a_validator_error_is_answered_with_422_and_a_readable_context(probe_client):
    response = probe_client.post("/__validation_probe", json={"permissions": {"X": ["write"]}})

    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["msg"] == "Value error, Invalid resource: X"
    assert detail["ctx"] == {"error": "Invalid resource: X"}
