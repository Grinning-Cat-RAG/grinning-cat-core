"""The ingestion states of the sources: the knowledge base of the agent, or the chat of the request (its header)."""
import asyncio
import contextlib
import copy
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds.ingestions import STATUS_LABELS, IngestionStatus
from cat.exceptions import CustomConflictException, CustomNotFoundException, CustomValidationException
from cat.log import log
from cat.rabbit_hole import DELETE_WAIT_SECONDS, chunks, engine
from cat.rabbit_hole.phases import IngestionPhase, IngestionSource, PhaseGraphError
from cat.routes.routes_utils import run_background_task
from cat.utils import is_url

router = APIRouter(tags=["Ingestion"], prefix="/ingestion")


class IngestionStatusResponse(BaseModel):
    source: str
    chat_id: str | None
    type: str
    content_type: str | None = None
    metadata: Dict[str, Any] = {}
    status: str
    status_label: str
    phase: str | None = None
    phase_label: str | None = None
    error: str | None = None
    embedder_name: str | None = None
    chunker_name: str | None = None
    completed_phases: Dict[str, Dict[str, Any]] = {}
    created_at: float
    updated_at: float


class PhaseResponse(BaseModel):
    id: str
    label: str
    order: int
    depends_on: List[str]


class StateResponse(BaseModel):
    id: str
    label: str


class PhasesResponse(BaseModel):
    phases: List[PhaseResponse]
    states: List[StateResponse]


class DismissedResponse(BaseModel):
    deleted: bool


class RetryResponse(BaseModel):
    retrying: bool


async def _phases(info: AuthorizedInfo) -> List[IngestionPhase]:
    """The phases of the agent, in execution order (400 when the phases of its plugins do not form a graph)."""
    rabbit_hole = copy.copy(info.lizard.rabbit_hole)
    await rabbit_hole.setup(info.cheshire_cat)
    try:
        return await engine.collect_phases(rabbit_hole, info.cheshire_cat)
    except PhaseGraphError as e:
        raise CustomValidationException(str(e))


def _chat_id(info: AuthorizedInfo) -> str | None:
    return info.stray_cat.id if info.stray_cat else None


@router.get("/status", response_model=List[IngestionStatusResponse])
async def get_ingestion_statuses(
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> List[IngestionStatusResponse]:
    """The ingestion states of the sources of the knowledge base, or of the chat of the request."""
    ccat = info.cheshire_cat
    chat_id = _chat_id(info)
    try:
        await crud_ingestions.reconcile_statuses(ccat.agent_key, ccat, chat_id)
    except Exception as e:  # noqa: BLE001 - a reconcile failure must not hide the states
        log.error(f"Ingestion status reconcile failed: {e}")
    labels = {phase.id: phase.display_label for phase in await _phases(info)}
    rows = await crud_ingestions.list_statuses(ccat.agent_key, chat_id)
    return [
        IngestionStatusResponse(
            **{k: v for k, v in row.items() if k in IngestionStatusResponse.model_fields},
            status_label=STATUS_LABELS.get(row["status"], row["status"]),
            phase_label=labels.get(row["phase"], row["phase"]) if row.get("phase") else None,
        )
        for row in sorted(rows, key=lambda r: r["created_at"])
    ]


@router.get("/phases", response_model=PhasesResponse)
async def get_ingestion_phases(
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> PhasesResponse:
    """The phases of the ingestion of the agent, in execution order, and the states a source goes through."""
    phases = await _phases(info)
    return PhasesResponse(
        phases=[
            PhaseResponse(id=phase.id, label=phase.display_label, order=order, depends_on=list(phase.depends_on))
            for order, phase in enumerate(phases, start=1)
        ],
        states=[StateResponse(id=status.value, label=STATUS_LABELS[status.value]) for status in IngestionStatus],
    )


@contextlib.asynccontextmanager
async def _failed_source(info: AuthorizedInfo, source: str, action: str):
    """Hold the lock of a failed source of the scope of the request, yielding its row: 404 when it has no state, 409
    while it is being ingested or when it did not fail."""
    ccat = info.cheshire_cat
    chat_id = _chat_id(info)
    if await crud_ingestions.get_status(ccat.agent_key, chat_id, source) is None:
        raise CustomNotFoundException(f"No ingestion state for the source {source}")

    ingestion = IngestionSource(ccat.agent_key, chat_id, source, "url" if is_url(source) else "file")
    try:
        async with engine.source_lock(ingestion, wait=DELETE_WAIT_SECONDS):
            row = await crud_ingestions.get_status(ccat.agent_key, chat_id, source)
            if row is None:
                raise CustomNotFoundException(f"No ingestion state for the source {source}")
            if row["status"] != IngestionStatus.ERROR.value:
                raise CustomConflictException(
                    f"Only a failed ingestion can be {action} ({source} is {row['status']})"
                )
            yield row
    except engine.LockError:
        raise CustomConflictException(f"The source {source} is being ingested: retry later")


@router.delete("/status", response_model=DismissedResponse)
async def dismiss_ingestion_status(
    source: str = Query(description="The source (file name or URL) whose failed ingestion is dismissed"),
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> DismissedResponse:
    """Dismiss a failed ingestion: its state and the files kept to process it go; the uploaded file and the memory
    stay (at a change of the embedder, such a source gets its state again and is processed again from its file). Only
    a failed source can be dismissed (409 otherwise, also while it is being ingested): a source is removed by deleting
    its file."""
    ccat = info.cheshire_cat
    chat_id = _chat_id(info)
    async with _failed_source(info, source, "dismissed: delete the file to remove it"):
        await asyncio.to_thread(chunks.delete_all, ccat.file_manager, ccat.agent_key, chat_id, source)
        await crud_ingestions.delete_status(ccat.agent_key, chat_id, source)
    return DismissedResponse(deleted=True)


@router.post("/retry", response_model=RetryResponse)
async def retry_ingestion(
    background_tasks: BackgroundTasks,
    source: str = Query(description="The source (file name or URL) whose failed ingestion is retried"),
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> RetryResponse:
    """Retry a failed ingestion: its failed phases run again, with the phases depending on them (the others are kept,
    e.g. a graph extraction is retried without parsing and embedding the source again; a failed phase no longer active
    is not run). The source turns ``uploaded`` at once and runs in background: follow it with GET /ingestion/status.
    409 when the source did not fail (a retry already going on included), while it is being ingested, or when it
    failed before its phases (e.g. a download): upload it again."""
    ccat = info.cheshire_cat
    chat_id = _chat_id(info)
    async with _failed_source(info, source, "retried") as row:
        failed = [
            phase_id for phase_id, entry in (row.get("completed_phases") or {}).items()
            if entry.get("state") == engine.STATE_ERROR
        ]
        if not failed:
            raise CustomConflictException(f"No phase of {source} failed, it failed before them: upload it again")
        # holding the lock: the failed phases are due again and the source waits for its worker, so the client sees
        # the retry at once and a second retry is refused; a dead pod leaves it to the automatic resume
        await crud_ingestions.reset_phase_entries(ccat.agent_key, chat_id, source, failed)
        await crud_ingestions.update_status(
            ccat.agent_key, chat_id, source, status=IngestionStatus.UPLOADED, phase=None, error=None, owner=None,
        )
    run_background_task(background_tasks, info.lizard.rabbit_hole.rerun_source, ccat, source, chat_id)
    return RetryResponse(retrying=True)
