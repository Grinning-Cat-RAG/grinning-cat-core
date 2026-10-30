from typing import Dict
from fastapi import APIRouter, BackgroundTasks, Body

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.db.cruds import settings as crud_settings
from cat.routes.routes_utils import (
    GetSettingsResponse,
    GetSettingResponse,
    UpsertSettingResponse,
    has_write_permission,
    run_background_task,
)
from cat.services.service_factory import ServiceFactory

router = APIRouter(tags=["Chunking"], prefix="/chunking")


def _chunker_signature(setting: Dict | None) -> tuple | None:
    """Material identity of a chunker setting (name + value, no timestamp)."""
    if not isinstance(setting, dict):
        return None
    return setting.get("name"), setting.get("value")


@router.get("/settings", response_model=GetSettingsResponse)
async def get_chunker_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.CHUNKER, AuthPermission.READ),
) -> GetSettingsResponse:
    """Get the list of the Chunkers"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_chunkers",
        setting_category="chunker",
        schema_name="chunkerName",
    )
    return await sf.get_factory_settings(reveal=has_write_permission(info.user.permissions, AuthResource.CHUNKER))


@router.get("/settings/{chunker_name}", response_model=GetSettingResponse)
async def get_chunker_setting(
    chunker_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.AUTH_HANDLER, AuthPermission.READ),
) -> GetSettingResponse:
    """Get the settings of a specific Chunker"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_chunkers",
        setting_category="chunker",
        schema_name="chunkerName",
    )
    return await sf.get_factory_setting(chunker_name, reveal=has_write_permission(info.user.permissions, AuthResource.CHUNKER))


@router.put("/settings/{chunker_name}", response_model=UpsertSettingResponse)
async def upsert_chunker_setting(
    background_tasks: BackgroundTasks,
    chunker_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.AUTH_HANDLER, AuthPermission.WRITE),
    payload: Dict = Body(default={}),
) -> UpsertSettingResponse:
    """Upsert the settings of a specific Chunker"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_chunkers",
        setting_category="chunker",
        schema_name="chunkerName",
    )

    previous_chunker = await crud_settings.get_settings_by_category(ccat.agent_key, "chunker")  # type: ignore[union-attr]
    result = await sf.upsert_service(chunker_name, payload)
    current_chunker = await crud_settings.get_settings_by_category(ccat.agent_key, "chunker")  # type: ignore[union-attr]

    # a characterizing feature of the chunker has been updated: schedule the
    # replaceable ingestion engine, exactly like the embedder route does, so a
    # chunker change actually re-parses (marker slow-path in the phase machine)
    if _chunker_signature(previous_chunker) != _chunker_signature(current_chunker):
        engine = await info.lizard.ingestion()
        run_background_task(
            background_tasks,
            engine.run,
            info.lizard,
        )

    return UpsertSettingResponse(**result)
