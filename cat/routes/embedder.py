from typing import Dict
from fastapi import APIRouter, BackgroundTasks, Body

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.routes.routes_utils import (
    GetSettingResponse,
    GetSettingsResponse,
    UpsertSettingResponse,
    run_background_task,
    has_write_permission,
)
from cat.services.service_factory import ServiceFactory


async def _run_ingestion_on_embedder_change(lizard) -> None:
    """Re-embed pass on embedder change, via the replaceable ingestion engine.

    The engine is resolved through the ServiceFactory (``ingestion``
    category): the core provides ``BaseIngestionConfiguration`` (upstream
    parity) and plugins can register more efficient implementations through
    the ``factory_allowed_ingestions`` hook.
    """
    from cat.services.factory.ingestion import resolve_ingestion_engine

    engine = await resolve_ingestion_engine(lizard)
    if engine is not None:
        await engine.run(lizard)


router = APIRouter(tags=["Embedder"], prefix="/embedder")


# get configured Embedders and configuration schemas
@router.get("/settings", response_model=GetSettingsResponse)
async def get_embedders_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.EMBEDDER, AuthPermission.READ),
) -> GetSettingsResponse:
    """Get the list of the Embedders"""
    lizard = info.lizard
    sf = ServiceFactory(
        agent_key=lizard.agent_key,  # type: ignore[arg-type]
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_embedders",
        setting_category="embedder",
        schema_name="languageEmbedderName",
    )
    return await sf.get_factory_settings(reveal=has_write_permission(info.user.permissions, AuthResource.EMBEDDER))


@router.get("/settings/{embedder_name}", response_model=GetSettingResponse)
async def get_embedder_settings(
    embedder_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.EMBEDDER, AuthPermission.READ),
) -> GetSettingResponse:
    """Get settings and scheme of the specified Embedder"""
    lizard = info.lizard
    sf = ServiceFactory(
        agent_key=lizard.agent_key,  # type: ignore[arg-type]
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_embedders",
        setting_category="embedder",
        schema_name="languageEmbedderName",
    )
    return await sf.get_factory_setting(
        embedder_name,
        reveal=has_write_permission(info.user.permissions, AuthResource.EMBEDDER)
    )


@router.put("/settings/{embedder_name}", response_model=UpsertSettingResponse)
async def upsert_embedder_setting(
    background_tasks: BackgroundTasks,
    embedder_name: str,
    payload: Dict = Body(default={}),
    info: AuthorizedInfo = check_permissions(AuthResource.EMBEDDER, AuthPermission.WRITE),
) -> UpsertSettingResponse:
    """Upsert the Embedder setting"""
    lizard = info.lizard
    previous_embedder = await lizard.embedder()
    sf = ServiceFactory(
        agent_key=lizard.agent_key,  # type: ignore[arg-type]
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_embedders",
        setting_category="embedder",
        schema_name="languageEmbedderName",
    )

    result = await sf.upsert_service(embedder_name, payload)

    current_embedder = await lizard.embedder()

    # a characterizing feature of the embedder has been updated: run the
    # replaceable ingestion engine (factory: ingestion category)
    if previous_embedder != current_embedder:
        run_background_task(
            background_tasks,
            _run_ingestion_on_embedder_change,
            info.lizard,
        )

    return UpsertSettingResponse(**result)
