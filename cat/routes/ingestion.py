"""The ingestion states of the sources: the knowledge base of the agent, or the chat of the request (its header)."""
import asyncio
import copy
from typing import Any, Dict, List

from fastapi import APIRouter, Query
from pydantic import BaseModel

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds.ingestions import STATUS_LABELS, IngestionStatus
from cat.exceptions import CustomConflictException, CustomNotFoundException, CustomValidationException
from cat.rabbit_hole import DELETE_WAIT_SECONDS, chunks, engine
from cat.rabbit_hole.phases import IngestionPhase, IngestionSource, PhaseGraphError
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
    labels = {phase.id: phase.display_label for phase in await _phases(info)}
    rows = await crud_ingestions.list_statuses(info.cheshire_cat.agent_key, _chat_id(info))
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


@router.delete("/status", response_model=DismissedResponse)
async def dismiss_ingestion_status(
    source: str = Query(description="The source (file name or URL) whose failed ingestion is dismissed"),
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> DismissedResponse:
    """Dismiss a failed ingestion: its state and the files kept to process it go; the uploaded file and the memory
    stay. Only a failed source can be dismissed (409 otherwise, also while it is being ingested): a source is removed
    by deleting its file."""
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
                    f"Only a failed ingestion can be dismissed ({source} is {row['status']}): delete the file to remove it"
                )
            await asyncio.to_thread(chunks.delete_all, ccat.file_manager, ccat.agent_key, chat_id, source)
            await crud_ingestions.delete_status(ccat.agent_key, chat_id, source)
    except engine.LockError:
        raise CustomConflictException(f"The source {source} is being ingested: retry later")
    return DismissedResponse(deleted=True)
