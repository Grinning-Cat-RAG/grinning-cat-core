from typing import Dict

from fastapi import APIRouter, Body

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.routes.routes_utils import (
    GetSettingResponse,
    GetSettingsResponse,
    UpsertSettingResponse,
    has_write_permission,
)
from cat.services.service_factory import ServiceFactory

router = APIRouter(tags=["Ingestion"], prefix="/ingestion")


@router.get("/settings", response_model=GetSettingsResponse)
async def get_ingestion_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.INGESTION, AuthPermission.READ),
) -> GetSettingsResponse:
    """Get the list of the Ingestions"""
    lizard = info.lizard
    sf = ServiceFactory(
        agent_key=lizard.agent_key,  # type: ignore[arg-type]
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_ingestions",
        setting_category="ingestion",
        schema_name="ingestionName",
    )
    return await sf.get_factory_settings(reveal=has_write_permission(info.user.permissions, AuthResource.INGESTION))


@router.get("/settings/{ingestion_name}", response_model=GetSettingResponse)
async def get_ingestion_setting(
    ingestion_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.INGESTION, AuthPermission.READ),
) -> GetSettingResponse:
    """Get settings and scheme of the specified Ingestion"""
    lizard = info.lizard
    sf = ServiceFactory(
        agent_key=lizard.agent_key,  # type: ignore[arg-type]
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_ingestions",
        setting_category="ingestion",
        schema_name="ingestionName",
    )
    return await sf.get_factory_setting(
        ingestion_name,
        reveal=has_write_permission(info.user.permissions, AuthResource.INGESTION)
    )


@router.put("/settings/{ingestion_name}", response_model=UpsertSettingResponse)
async def upsert_ingestion_setting(
    ingestion_name: str,
    payload: Dict = Body(default={}),
    info: AuthorizedInfo = check_permissions(AuthResource.INGESTION, AuthPermission.WRITE),
) -> UpsertSettingResponse:
    """Upsert the Ingestion setting"""
    lizard = info.lizard
    sf = ServiceFactory(
        agent_key=lizard.agent_key,  # type: ignore[arg-type]
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_ingestions",
        setting_category="ingestion",
        schema_name="ingestionName",
    )

    result = await sf.upsert_service(ingestion_name, payload)
    return UpsertSettingResponse(**result)
