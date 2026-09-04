from typing import Dict
from fastapi import APIRouter, Body, BackgroundTasks

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.db.cruds import settings as crud_settings
from cat.routes.routes_utils import GetSettingsResponse, GetSettingResponse, UpsertSettingResponse, run_background_task, has_write_permission
from cat.services.service_factory import ServiceFactory

router = APIRouter(tags=["Vector Database"], prefix="/vector_database")


# get configured LLMs and configuration schemas
@router.get("/settings", response_model=GetSettingsResponse, summary="Get Vector Databases Settings")
async def get_vector_databases_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.VECTOR_DATABASE, AuthPermission.READ),
) -> GetSettingsResponse:
    """Get the list of the Vector Databases settings and their configuration schemas"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[arg-type]
        hook_manager=ccat.plugin_manager,  # type: ignore[arg-type]
        factory_allowed_handler_name="factory_allowed_vector_databases",
        setting_category="vector_database",
        schema_name="vectorDatabaseName",
    )
    return await sf.get_factory_settings(reveal=has_write_permission(info.user.permissions, AuthResource.VECTOR_DATABASE))


@router.get(
    "/settings/{vector_database_name}", response_model=GetSettingResponse, summary="Get Vector Database Settings"
)
async def get_vector_database_settings(
    vector_database_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.VECTOR_DATABASE, AuthPermission.READ),
) -> GetSettingResponse:
    """Get settings and scheme of the specified Vector Database"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[arg-type]
        hook_manager=ccat.plugin_manager,  # type: ignore[arg-type]
        factory_allowed_handler_name="factory_allowed_vector_databases",
        setting_category="vector_database",
        schema_name="vectorDatabaseName",
    )
    return await sf.get_factory_setting(vector_database_name, reveal=has_write_permission(info.user.permissions, AuthResource.VECTOR_DATABASE))


@router.put(
    "/settings/{vector_database_name}", response_model=UpsertSettingResponse, summary="Upsert Vector Database Settings"
)
async def upsert_vector_database_setting(
    background_tasks: BackgroundTasks,
    vector_database_name: str,
    payload: Dict = Body(default={}),
    info: AuthorizedInfo = check_permissions(AuthResource.VECTOR_DATABASE, AuthPermission.WRITE),
) -> UpsertSettingResponse:
    """Upsert the Vector Database setting"""
    ccat = info.cheshire_cat

    previous_vector_db = ccat.vector_memory_handler
    previous_config = await crud_settings.get_setting_by_name(ccat.agent_key, vector_database_name)  # type: ignore[union-attr]

    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_vector_databases",
        setting_category="vector_database",
        schema_name="vectorDatabaseName",
    )
    result = await sf.upsert_service(vector_database_name, payload)

    ccat.vector_memory_handler = await ccat.service_provider.get_vector_memory_handler(
        ccat.agent_key, ccat.plugin_manager,
    )
    if previous_vector_db != ccat.vector_memory_handler:
        run_background_task(background_tasks, ccat.transfer_vector_points_from, previous_vector_db)
    elif previous_config is not None and previous_config["value"] != payload:
        await ccat.plugin_manager.execute_hook(  # type: ignore[union-attr]
            "after_vector_database_settings_update",
            vector_database_name,
            previous_config["value"],
            payload,
            caller=ccat,
        )

    return UpsertSettingResponse(**result)
