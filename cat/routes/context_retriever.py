from typing import Dict
from fastapi import APIRouter, Body

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.routes.routes_utils import GetSettingsResponse, GetSettingResponse, UpsertSettingResponse, has_write_permission
from cat.services.service_factory import ServiceFactory

router = APIRouter(tags=["Context Retriever"], prefix="/context_retriever")


@router.get("/settings", response_model=GetSettingsResponse)
async def get_context_retriever_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.CONTEXT_RETRIEVER, AuthPermission.READ),
) -> GetSettingsResponse:
    """Get the list of the Context Retrievers"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_context_retrievers",
        setting_category="context_retriever",
        schema_name="contextRetrieverName",
    )
    return await sf.get_factory_settings(reveal=has_write_permission(info.user.permissions, AuthResource.CONTEXT_RETRIEVER))


@router.get("/settings/{context_retriever_name}", response_model=GetSettingResponse)
async def get_context_retriever_setting(
    context_retriever_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.AUTH_HANDLER, AuthPermission.READ),
) -> GetSettingResponse:
    """Get the settings of a specific Context Retriever"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_context_retrievers",
        setting_category="context_retriever",
        schema_name="contextRetrieverName",
    )
    return await sf.get_factory_setting(context_retriever_name, reveal=has_write_permission(info.user.permissions, AuthResource.CONTEXT_RETRIEVER))


@router.put("/settings/{context_retriever_name}", response_model=UpsertSettingResponse)
async def upsert_context_retriever_setting(
    context_retriever_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.AUTH_HANDLER, AuthPermission.WRITE),
    payload: Dict = Body(default={}),
) -> UpsertSettingResponse:
    """Upsert the settings of a specific Context Retriever"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_context_retrievers",
        setting_category="context_retriever",
        schema_name="contextRetrieverName",
    )

    result = await sf.upsert_service(context_retriever_name, payload)
    return UpsertSettingResponse(**result)
