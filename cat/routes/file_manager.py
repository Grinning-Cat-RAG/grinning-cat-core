import os
from io import BytesIO
from typing import Dict, List, Tuple
from fastapi import APIRouter, BackgroundTasks, Body
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.db.cruds import ingestions as crud_ingestions
from cat.exceptions import CustomConflictException, CustomNotFoundException, CustomValidationException
from cat.rabbit_hole import DELETE_WAIT_SECONDS, SourceBusy
from cat.routes.routes_utils import (
    GetSettingResponse,
    GetSettingsResponse,
    UpsertSettingResponse,
    run_background_task,
    sanitize_source_name,
    has_write_permission,
)
from cat.services.factory.file_manager import FileResponse
from cat.services.memory.models import VectorMemoryType
from cat.services.service_factory import ServiceFactory

router = APIRouter(tags=["File Manager"], prefix="/file_manager")


class FileManagerDeletedFiles(BaseModel):
    deleted: bool


class FileManagerAttributes(BaseModel):
    files: List[FileResponse]
    size: int


def get_from_info(info: AuthorizedInfo) -> Tuple[str, VectorMemoryType, Dict]:
    ccat = info.cheshire_cat
    path = ccat.agent_key  # type: ignore[union-attr]
    if info.stray_cat:
        path = os.path.join(path, info.stray_cat.id)

    collection_id = VectorMemoryType.DECLARATIVE if not info.stray_cat else VectorMemoryType.EPISODIC
    metadata = {"chat_id": info.stray_cat.id} if info.stray_cat else {}

    return path, collection_id, metadata


# get configured Plugin File Managers and configuration schemas
@router.get("/settings", response_model=GetSettingsResponse)
async def get_file_managers_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.FILE_MANAGER, AuthPermission.READ),
) -> GetSettingsResponse:
    """Get the list of the File Managers and their settings"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_file_managers",
        setting_category="file_manager",
        schema_name="fileManagerName",
    )
    return await sf.get_factory_settings(reveal=has_write_permission(info.user.permissions, AuthResource.FILE_MANAGER))


@router.get("/settings/{file_manager_name}", response_model=GetSettingResponse)
async def get_file_manager_settings(
    file_manager_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.FILE_MANAGER, AuthPermission.READ),
) -> GetSettingResponse:
    """Get settings and scheme of the specified File Manager"""
    ccat = info.cheshire_cat
    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_file_managers",
        setting_category="file_manager",
        schema_name="fileManagerName",
    )
    return await sf.get_factory_setting(file_manager_name, reveal=has_write_permission(info.user.permissions, AuthResource.FILE_MANAGER))


@router.put("/settings/{file_manager_name}", response_model=UpsertSettingResponse)
async def upsert_file_manager_setting(
    background_tasks: BackgroundTasks,
    file_manager_name: str,
    payload: Dict = Body(default={}),
    info: AuthorizedInfo = check_permissions(AuthResource.FILE_MANAGER, AuthPermission.WRITE),
) -> UpsertSettingResponse:
    """Upsert the File Manager setting"""
    ccat = info.cheshire_cat

    previous_file_manager = ccat.file_manager

    sf = ServiceFactory(
        agent_key=ccat.agent_key,  # type: ignore[union-attr]
        hook_manager=ccat.plugin_manager,  # type: ignore[union-attr]
        factory_allowed_handler_name="factory_allowed_file_managers",
        setting_category="file_manager",
        schema_name="fileManagerName",
    )

    result = await sf.upsert_service(file_manager_name, payload)

    ccat.file_manager = await ccat.service_provider.get_file_manager(ccat.agent_key, ccat.plugin_manager)
    current_file_manager = ccat.file_manager
    if previous_file_manager != current_file_manager:
        run_background_task(background_tasks, ccat.transfer_files_from, previous_file_manager)

    return UpsertSettingResponse(**result)


@router.get("/", response_model=FileManagerAttributes)
async def get_attributes(
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> FileManagerAttributes:
    path, _, _ = get_from_info(info)

    list_files = info.cheshire_cat.file_manager.list_files(path)
    return FileManagerAttributes(files=list_files, size=sum(file.size for file in list_files))


@router.get("/files/{source_name:path}")
async def download_file(
    source_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> StreamingResponse:
    path, _, _ = get_from_info(info)

    sanitized_source = sanitize_source_name(source_name, path=path)

    # Download the file
    file_content = info.cheshire_cat.file_manager.download_file(os.path.join(path, sanitized_source))
    if file_content is None:
        raise CustomNotFoundException("File not found")

    # Sanitize the filename for the Content-Disposition header to prevent header injection
    safe_filename = sanitized_source.encode("ascii", "ignore").decode("ascii")
    safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in ".-_")

    return StreamingResponse(
        BytesIO(file_content),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
    )


async def _delete_source(info: AuthorizedInfo, path: str, name: str) -> bool:
    """Delete a source of the scope (the agent knowledge base, or the chat of the request) holding its lock: the
    uploaded file, then (the plugins notified) what its ingestion phases produced, its points among them, and its state.
    Returns whether the file was removed. ``SourceBusy`` when its ingestion is still running after the wait."""
    ccat = info.cheshire_cat
    scope = path.split(os.sep, 1)[1] if os.sep in path else "agent"

    async def remove_file() -> bool:
        removed = ccat.file_manager.remove_file(os.path.join(path, name))
        # the file is gone from storage: notify the plugins BEFORE the memory points are deleted
        await ccat.plugin_manager.execute_hook("before_file_manager_file_delete", name, str(scope), caller=ccat)
        return removed

    removed = await info.lizard.rabbit_hole.delete_source(
        ccat, name, info.stray_cat.id if info.stray_cat else None, wait=DELETE_WAIT_SECONDS, before=remove_file,
    )
    # the file is gone: notify the plugins
    await ccat.plugin_manager.execute_hook("after_file_manager_file_deleted", name, str(scope), caller=ccat)
    return removed


@router.delete("/files/{source_name:path}")
async def delete_file(
    source_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> FileManagerDeletedFiles:
    """Delete a file, with the source ingested from it. 409 while the source is being ingested."""
    path, _, _ = get_from_info(info)

    sanitized_source = sanitize_source_name(source_name, path=path)

    try:
        return FileManagerDeletedFiles(deleted=await _delete_source(info, path, sanitized_source))
    except SourceBusy:
        raise CustomConflictException(f"The source {sanitized_source} is being ingested: retry later")
    except Exception as e:
        raise CustomValidationException(f"Failed to delete memory points: {e}")


@router.delete("/files")
async def delete_files(
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> FileManagerDeletedFiles:
    """Delete every source of the scope (the agent knowledge base, or the chat of the request): its files and URLs,
    with what they produced. The sources being ingested are left; then 409 tells which ones."""
    path, _, _ = get_from_info(info)
    ccat = info.cheshire_cat
    chat_id = info.stray_cat.id if info.stray_cat else None

    try:
        # the sources with a state, and the files listed in the scope (sources ingested before the states existed)
        names = [row["source"] for row in await crud_ingestions.list_statuses(ccat.agent_key, chat_id)]
        names += [file.name for file in ccat.file_manager.list_files(path) if file.name not in names]

        busy = []
        for name in names:
            try:
                await _delete_source(info, path, name)
            except SourceBusy:
                busy.append(name)

        # a chat folder holds the chat only: remove what is left; the agent folder holds the chats too
        if chat_id is not None and not busy:
            ccat.file_manager.remove_folder(path)
    except Exception as e:
        raise CustomValidationException(f"Failed to delete memory points: {e}")

    if busy:
        raise CustomConflictException(f"Sources being ingested, not deleted (retry later): {', '.join(busy)}")
    return FileManagerDeletedFiles(deleted=True)

