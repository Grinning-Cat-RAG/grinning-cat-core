import os

from cat.env import get_env, get_env_int, get_supported_env_variables


def test_get_env():
    # container envs
    assert get_env("PYTHONUNBUFFERED") == "1"
    assert get_env("UV_LINK_MODE") == "copy"
    assert get_env("UV_NO_CACHE") == "1"

    # unexisting
    assert get_env("UNEXISTING_ENV") is None
    assert get_env("CAT_UNEXISTING_ENV") is None

    # set new
    os.environ["FAKE_ENV"] = "meow1"
    os.environ["CAT_FAKE_ENV"] = "meow2"
    assert get_env("FAKE_ENV") == "meow1"
    assert get_env("CAT_FAKE_ENV") == "meow2"

    # default env variables
    for k, v in get_supported_env_variables().items():
        assert get_env(k) == os.getenv(k, v)


def test_get_env_int_ingestion_max_concurrency():
    # default value is 2
    assert get_env_int("CAT_INGESTION_MAX_CONCURRENCY") == 2

    # explicit env override is honored
    os.environ["CAT_INGESTION_MAX_CONCURRENCY"] = "5"
    assert get_env_int("CAT_INGESTION_MAX_CONCURRENCY") == 5
    del os.environ["CAT_INGESTION_MAX_CONCURRENCY"]


def test_get_env_int_ingestion_workers():
    # default value is 2
    assert get_env_int("CAT_INGESTION_WORKERS") == 2

    # explicit env override is honored
    os.environ["CAT_INGESTION_WORKERS"] = "5"
    assert get_env_int("CAT_INGESTION_WORKERS") == 5
    del os.environ["CAT_INGESTION_WORKERS"]


def test_get_env_int_ingestion_niceness():
    # default value is 5
    assert get_env_int("CAT_INGESTION_NICENESS") == 5

    # explicit env override is honored
    os.environ["CAT_INGESTION_NICENESS"] = "10"
    assert get_env_int("CAT_INGESTION_NICENESS") == 10
    del os.environ["CAT_INGESTION_NICENESS"]
