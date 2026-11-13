import os
import shutil
from typing import List, Dict
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
import cat.db.cruds.settings as crud_settings
from cat.db.database import get_async_db
from cat.db.models import Setting
from cat.exceptions import UnknownAgentException
from cat.log import log
from cat.routes.routes_utils import startup_app, shutdown_app
import cat.utils as utils

router = APIRouter(tags=["Utilities"], prefix="/utils")


class AgentUpdateRequest(BaseModel):
    metadata: Dict | None = Field(default={})


class AgentCloneRequest(BaseModel):
    agent_id: str


class AgentCreateRequest(AgentCloneRequest, AgentUpdateRequest):
    pass


class AgentResponse(BaseModel):
    agent_id: str
    metadata: Dict


class AgentCreatedResponse(BaseModel):
    created: bool


class AgentUpdatedResponse(BaseModel):
    updated: bool


class AgentClonedResponse(BaseModel):
    cloned: bool = False


class ResetResponse(BaseModel):
    deleted_settings: bool
    deleted_memories: bool
    deleted_plugin_folders: bool


@router.post("/factory/reset", response_model=ResetResponse)
async def factory_reset(
    request: Request,
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.DELETE),
) -> ResetResponse:
    """
    Factory reset the entire application. This will delete all settings, memories, and metadata.
    """
    # remove memories
    cheshire_cats_ids = await crud_settings.get_agents_main_keys()
    deleted_memories = False
    for agent_id in cheshire_cats_ids:
        try:
            ccat = await info.lizard.get_cheshire_cat(agent_id)
        except UnknownAgentException:
            # a ghost namespace (peripheral keys, no master key): nothing to reset
            log.warning(f"Skipping ghost agent {agent_id} during factory reset")
            continue
        if not ccat:
            continue
        try:
            await ccat.destroy()
            deleted_memories = True
        except Exception as e:
            log.error(f"Error deleting memories for agent {agent_id}: {e}")
            deleted_memories = False

    await shutdown_app(request.app)

    try:
        await get_async_db().flushdb()
        deleted_settings = True
    except Exception as e:
        log.error(f"Error deleting settings: {e}")
        deleted_settings = False

    try:
        # empty the plugin folder
        plugin_folder = utils.get_plugins_path()
        for _, folders, _ in os.walk(plugin_folder):
            for folder in folders:
                item = os.path.join(plugin_folder, folder)  # type: ignore[union-attr]
                if os.path.isfile(item) or not os.path.exists(item):
                    continue
                shutil.rmtree(item)

        deleted_plugin_folders = True
    except Exception as e:
        log.error(f"Error deleting plugin folders: {e}")
        deleted_plugin_folders = False

    await startup_app(request.app)

    return ResetResponse(
        deleted_settings=deleted_settings,
        deleted_memories=deleted_memories,
        deleted_plugin_folders=deleted_plugin_folders,
    )


@router.get("/agents", dependencies=[check_permissions(AuthResource.CHESHIRE_CAT, AuthPermission.READ)])
async def get_agents() -> List[AgentResponse]:
    """
    Get all agents.
    """
    try:
        return [AgentResponse(**agent) for agent in await crud_settings.get_agents()]
    except Exception as e:
        log.error(f"Error creating agent: {e}")
        return []


@router.post("/agents/create", response_model=AgentCreatedResponse)
async def agent_create(
    request: AgentCreateRequest,
    info: AuthorizedInfo = check_permissions(AuthResource.CHESHIRE_CAT, AuthPermission.WRITE),
) -> AgentCreatedResponse:
    """
    Create a single agent.
    """
    try:
        await info.lizard.create_cheshire_cat(request.agent_id, request.metadata)
        return AgentCreatedResponse(created=True)
    except Exception as e:
        log.error(f"Error creating agent: {e}")
        return AgentCreatedResponse(created=False)


@router.put("/agents", response_model=AgentUpdatedResponse)
async def agent_update(
    request: AgentUpdateRequest,
    info: AuthorizedInfo = check_permissions(AuthResource.CHESHIRE_CAT, AuthPermission.WRITE),
) -> AgentUpdatedResponse:
    """
    Update the metadata of a specific agent.
    """
    try:
        await crud_settings.upsert_setting_by_name(
            info.cheshire_cat.agent_key, Setting(name="metadata", value=request.metadata)  # type: ignore[union-attr]
        )
        return AgentUpdatedResponse(updated=True)
    except Exception as e:
        log.error(f"Error updating agent: {e}")
        return AgentUpdatedResponse(updated=False)


@router.post("/agents/destroy", response_model=ResetResponse)
async def agent_destroy(
    info: AuthorizedInfo = check_permissions(AuthResource.CHESHIRE_CAT, AuthPermission.DELETE),
) -> ResetResponse:
    """
    Destroy a single agent. This will completely delete all settings, memories, and metadata, for the agent.
    This is a permanent action and cannot be undone.
    """
    deleted_settings = False
    deleted_memories = False

    try:
        await info.cheshire_cat.destroy()  # type: ignore[union-attr]
        deleted_settings = True
        deleted_memories = True
    except Exception as e:
        log.error(f"Error deleting settings: {e}")

    return ResetResponse(
        deleted_settings=deleted_settings,
        deleted_memories=deleted_memories,
        deleted_plugin_folders=False,
    )


@router.post("/agents/reset", response_model=ResetResponse)
async def agent_reset(
    info: AuthorizedInfo = check_permissions(AuthResource.CHESHIRE_CAT, AuthPermission.WRITE),
) -> ResetResponse:
    """
    Reset a single agent. This will delete all settings, memories, and metadata, for the agent.
    """
    ccat = info.cheshire_cat
    agent_id = ccat.agent_key  # type: ignore[union-attr]
    try:
        await ccat.destroy()  # type: ignore[union-attr]
        await info.lizard.create_cheshire_cat(agent_id)

        deleted_settings = True
        deleted_memories = True
    except Exception as e:
        log.error(f"Error deleting settings: {e}")
        deleted_settings = False
        deleted_memories = False

    return ResetResponse(
        deleted_settings=deleted_settings,
        deleted_memories=deleted_memories,
        deleted_plugin_folders=False,
    )


@router.post("/agents/clone", response_model=AgentClonedResponse)
async def agent_clone(
    request: AgentCloneRequest,
    info: AuthorizedInfo = check_permissions(AuthResource.CHESHIRE_CAT, AuthPermission.WRITE),
) -> AgentClonedResponse:
    """
    Clone a single agent. This will clone all settings, memories, and metadata, for the agent.
    """
    agent_id = request.agent_id
    if agent_id in await crud_settings.get_agents_main_keys():
        log.warning(f"Agent {agent_id} already exists. Cannot clone.")
        return AgentClonedResponse(cloned=False)

    ccat = info.cheshire_cat

    cloned_ccat = None
    try:
        cloned_ccat = await info.lizard.clone_cheshire_cat(ccat, agent_id)  # type: ignore[union-attr]
        return AgentClonedResponse(cloned=True)
    except Exception as e:
        log.error(f"Error cloning agent {ccat.agent_key}: {e}")  # type: ignore[union-attr]

        await info.lizard.rollback_cheshire_cat_creation(agent_id, cloned_ccat)
        return AgentClonedResponse(cloned=False)
