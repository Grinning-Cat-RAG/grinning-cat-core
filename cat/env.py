import os


def get_supported_env_variables():
    return {
        "CAT_ADMIN_DEFAULT_PASSWORD": "admin",
        "CAT_API_KEY": None,
        "CAT_DEBUG": "true",
        "CAT_LOG_LEVEL": "INFO",
        "CAT_CORS_ENABLED": "true",
        "CAT_CORS_ALLOWED_ORIGINS": None,
        "CAT_CORS_FORWARDED_ALLOW_IPS": "*",
        "CAT_REDIS_HOST": None,
        "CAT_REDIS_PORT": "6379",
        "CAT_REDIS_PASSWORD": "",
        "CAT_REDIS_DB": "0",
        "CAT_REDIS_TLS": False,
        "CAT_QDRANT_HOST": "grinning_cat_vector_memory",
        "CAT_QDRANT_API_KEY": None,
        "CAT_JWT_SECRET": "this_is_a_secret_key",
        "CAT_JWT_EXPIRE_MINUTES": str(60 * 24),  # JWT expires after 1 day
        "CAT_HTTPS_PROXY_MODE": "false",
        "CAT_HISTORY_EXPIRATION": None,  # in minutes
        "CAT_CRYPTO_KEY": "grinning_cat",
        "CAT_CRYPTO_SALT": "grinning_cat_salt",
        "CAT_INGESTION_MAX_CONCURRENCY": "2",
        "CAT_INGESTION_WORKERS": "2",
        "CAT_INGESTION_NICENESS": "5",
        "CAT_INGESTION_RESUME_INTERVAL_SECONDS": "60",
        "CAT_INGESTION_HEARTBEAT_SECONDS": "30",
    }


def get_env(name):
    """Utility to get an environment variable value. To be used only for supported Cat envs.
    - covers default supported variables and their default value
    - automagically handles legacy env variables missing the prefix "CAT_"
    """
    cat_default_env_variables = get_supported_env_variables()

    default = None
    if name in cat_default_env_variables:
        default = cat_default_env_variables[name]

    return os.getenv(name, default)


def get_env_bool(name):
    return get_env(name) in ("1", "true")


def get_env_float(name: str) -> float | None:
    value = get_env(name)
    if value is None:
        return None

    return float(value)


def get_env_int(name: str) -> int | None:
    value = get_env(name)
    if value is None:
        return None

    return int(value)
