from typing import Dict
from fastapi import APIRouter, Body

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.routes.routes_utils import GetSettingsResponse, GetSettingResponse, UpsertSettingResponse, has_write_permission
from cat.services.service_factory import ServiceFactory

router = APIRouter(tags=["Large Language Model"], prefix="/llm")


# get configured LLMs and configuration schemas
@router.get("/settings", response_model=GetSettingsResponse, summary="Get LLMs Settings")
async def get_llms_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.LLM, AuthPermission.READ),
) -> GetSettingsResponse:
    """Get the list of the Large Language Models"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_llms",
        setting_category="llm",
        schema_name="languageModelName",
    )
    return await sf.get_factory_settings(reveal=has_write_permission(info.user.permissions, AuthResource.LLM))


@router.get("/settings/{language_model_name}", response_model=GetSettingResponse, summary="Get LLM Settings")
async def get_llm_settings(
    language_model_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.LLM, AuthPermission.READ),
) -> GetSettingResponse:
    """Get settings and scheme of the specified Large Language Model"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_llms",
        setting_category="llm",
        schema_name="languageModelName",
    )
    return await sf.get_factory_setting(language_model_name, reveal=has_write_permission(info.user.permissions, AuthResource.LLM))


@router.put("/settings/{language_model_name}", response_model=UpsertSettingResponse, summary="Upsert LLM Settings")
async def upsert_llm_setting(
    language_model_name: str,
    payload: Dict = Body(default={}),
    info: AuthorizedInfo = check_permissions(AuthResource.LLM, AuthPermission.WRITE),
) -> UpsertSettingResponse:
    """Upsert the Large Language Model setting"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_llms",
        setting_category="llm",
        schema_name="languageModelName",
    )

    result = await sf.upsert_service(language_model_name, payload)
    return UpsertSettingResponse(**result)
