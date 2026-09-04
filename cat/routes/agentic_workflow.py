from typing import Dict
from fastapi import APIRouter, Body

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.routes.routes_utils import GetSettingsResponse, GetSettingResponse, UpsertSettingResponse, has_write_permission
from cat.services.service_factory import ServiceFactory


router = APIRouter(tags=["Agentic Workflow"], prefix="/agentic_workflow")


@router.get("/settings", response_model=GetSettingsResponse)
async def get_agentic_workflow_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.AGENTIC_WORKFLOW, AuthPermission.READ),
) -> GetSettingsResponse:
    """Get the list of the Agentic Workflow settings"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_agentic_workflows",
        setting_category="agentic_workflow",
        schema_name="agenticWorkflowName",
    )
    return await sf.get_factory_settings(reveal=has_write_permission(info.user.permissions, AuthResource.AGENTIC_WORKFLOW))


@router.get("/settings/{agentic_workflow_name}", response_model=GetSettingResponse)
async def get_agentic_workflow_setting(
    agentic_workflow_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.AGENTIC_WORKFLOW, AuthPermission.READ),
) -> GetSettingResponse:
    """Get the settings of a specific Agentic Workflow"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_agentic_workflows",
        setting_category="agentic_workflow",
        schema_name="agenticWorkflowName",
    )
    return await sf.get_factory_setting(
        agentic_workflow_name,
        reveal=has_write_permission(info.user.permissions, AuthResource.AGENTIC_WORKFLOW),
    )


@router.put("/settings/{agentic_workflow_name}", response_model=UpsertSettingResponse)
async def upsert_agentic_workflow_setting(
    agentic_workflow_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.AGENTIC_WORKFLOW, AuthPermission.WRITE),
    payload: Dict = Body(default={}),
) -> UpsertSettingResponse:
    """Upsert the settings of a specific Agentic Workflow"""
    ccat = info.cheshire_cat

    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_agentic_workflows",
        setting_category="agentic_workflow",
        schema_name="agenticWorkflowName",
    )
    result = await sf.upsert_service(agentic_workflow_name, payload)
    return UpsertSettingResponse(**result)
