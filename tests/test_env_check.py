import os
import pytest

from cat.env_check import EnvironmentConfigError, ensure_valid_env, validate_env
from cat.startup import create_app


@pytest.fixture
def restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize("bad", [None, "", "this_is_a_secret_key", "too_short"])
def test_invalid_jwt_secret_blocks_startup(restore_env, bad):
    if bad is None:
        os.environ.pop("CAT_JWT_SECRET", None)
    else:
        os.environ["CAT_JWT_SECRET"] = bad
    assert not validate_env().ok
    with pytest.raises(EnvironmentConfigError):
        ensure_valid_env()
    with pytest.raises(EnvironmentConfigError):
        create_app()


def test_invalid_numbers_and_origins(restore_env):
    os.environ["CAT_REDIS_PORT"] = "not_a_port"
    assert any("CAT_REDIS_PORT" in e for e in validate_env().errors)
    os.environ["CAT_REDIS_PORT"] = "6379"
    os.environ["CAT_CORS_ALLOWED_ORIGINS"] = "https://ok.example.org,not-an-origin"
    assert any("CAT_CORS_ALLOWED_ORIGINS" in e for e in validate_env().errors)


def test_valid_env():
    assert validate_env().ok


async def test_health_unhealthy_on_invalid_env(client, restore_env):
    assert (await client.get("/health/liveness")).status_code == 200
    os.environ["CAT_JWT_SECRET"] = "this_is_a_secret_key"
    for probe in ("/health/liveness", "/health/readiness"):
        res = await client.get(probe)
        assert res.status_code == 500
        # public endpoint: it must not reveal which variable is wrong
        assert "CAT_JWT_SECRET" not in res.text
