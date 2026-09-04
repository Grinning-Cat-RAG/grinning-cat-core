"""Settings endpoints for the replaceable ingestion engine.

The engine configuration is stored in the global ``system:agent`` settings
list (category ``ingestion``) and served with the same shapes as the core
``/embedder/settings`` routes. SYSTEM READ/WRITE protected: plugin endpoints
are public by default, so the permission dependency is mandatory.
"""

from typing import Any

from pydantic import ValidationError

from cat import endpoint
from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.exceptions import CustomNotFoundException, CustomValidationException
from cat.services.factory.ingestion import (
    build_factory,
    resolved_config_name,
)


@endpoint.get("/settings", prefix="/ingestion", tags=["Ingestion"])
async def get_ingestion_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.READ),
) -> dict[str, Any]:
    """List the available ingestion engines with schemes and the effective choice."""
    sf = build_factory(info.lizard)
    # built manually: get_factory_settings breaks when no engine was ever saved
    schemas = await sf.get_schemas()
    settings = [
        {"name": name, "value": await _saved_value(sf, name), "scheme": scheme}
        for name, scheme in schemas.items()
    ]
    selected = await resolved_config_name(info.lizard)
    return {"settings": settings, "selected_configuration": selected}


async def _saved_value(sf, name: str) -> dict:
    """Saved config value (decrypted) or {} when the entry was never stored."""
    from cat.db.cruds import settings as crud_settings
    from cat.db.database import DEFAULT_SYSTEM_KEY

    entry = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, name)
    if entry is None:
        return {}
    try:
        factory_class = await sf._get_factory_class(name)
        return factory_class.parse_config(entry.get("value") or {})
    except Exception:  # noqa: BLE001
        return {}


@endpoint.get("/settings/{configuration_name}", prefix="/ingestion", tags=["Ingestion"])
async def get_ingestion_setting(
    configuration_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.READ),
) -> dict[str, Any]:
    """Settings and scheme of one ingestion engine configuration."""
    sf = build_factory(info.lizard)
    setting = await sf.get_factory_setting(configuration_name)
    return {"name": setting.name, "value": setting.value, "scheme": setting.scheme}


@endpoint.put("/settings/{configuration_name}", prefix="/ingestion", tags=["Ingestion"])
async def upsert_ingestion_setting(
    configuration_name: str,
    payload: dict[str, Any] = {},
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.WRITE),
) -> dict[str, Any]:
    """Update one engine configuration and select it as the engine to run."""
    sf = build_factory(info.lizard)

    schemas = await sf.get_schemas()
    if configuration_name not in schemas:
        raise CustomNotFoundException(
            f"Configuration {configuration_name} not found. Must be one of {list(schemas.keys())}"
        )

    try:
        config_class = await sf._get_factory_class(configuration_name)
        if config_class is not None:
            config_class.model_validate(payload)
    except ValidationError as e:
        raise CustomValidationException("\n".join(err["msg"] for err in e.errors())) from e

    result = await sf.upsert_service(configuration_name, payload)
    # selecting a configuration is automatic: the ingestion category holds a
    # SINGLE setting (the active engine), exactly like the embedder category —
    # upserting the config replaces it, no separate selection entry needed.
    return {"name": configuration_name, "value": result.get("value", payload)}
