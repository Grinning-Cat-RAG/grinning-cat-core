import json
import mimetypes
import os
from io import BytesIO
from typing import Dict, List

import httpx
from fastapi import APIRouter, BackgroundTasks, Form, UploadFile
from pydantic import BaseModel, Field

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.exceptions import CustomValidationException
from cat.log import log
from cat.routes.routes_utils import run_background_task
from cat.services.memory.models import VectorMemoryType
from cat.utils import guess_file_type, write_temp_file

router = APIRouter(tags=["Rabbit Hole"], prefix="/rabbithole")


class UploadURLConfig(BaseModel, extra="forbid"):  # type: ignore[call-arg]
    url: str = Field(
        description="URL of the website to which you want to save the content"
    )
    metadata: Dict = Field(
        default={},
        description="Metadata to be stored with each chunk (e.g. author, category, etc.)"
    )


class UploadSingleFileResponse(BaseModel):
    filename: str
    content_type: str
    info: str


class UploadUrlResponse(BaseModel):
    url: str
    info: str


class AllowedMimeTypesResponse(BaseModel):
    allowed: List[str]


async def _on_upload_single_file(
    info: AuthorizedInfo,
    path: str,
    filename: str,
    metadata: Dict | None = None,
):
    lizard = info.lizard
    cat = info.stray_cat or info.cheshire_cat

    try:
        with open(path, "rb") as f:
            file_bytes = BytesIO(f.read())
            # Get file mime type
            content_type, _ = guess_file_type(file_bytes)

            # check if MIME type of uploaded file is supported
            fh = await cat.file_handlers()  # type: ignore[union-attr]
            admitted_types = fh.keys()
            if content_type not in admitted_types:
                raise CustomValidationException(
                    f"MIME type {content_type} not supported. Admitted types: {' - '.join(admitted_types)}"
                )

            # upload file to long-term memory through the configured ingestion
            # engine (a plugin may replace the flow with its own phase machine;
            # the default engine wraps the original rabbit_hole.ingest_file)
            from cat.services.factory.ingestion import resolve_ingestion_engine

            engine = await resolve_ingestion_engine(lizard)
            if engine is not None:
                await engine.ingest_file(
                    cat=cat,
                    file=file_bytes,
                    filename=filename,
                    content_type=content_type,
                    metadata=metadata or {},
                    store_file=True,
                )
            else:
                # fallback: no engine resolvable -> original upstream flow
                await lizard.rabbit_hole.ingest_file(
                    cat=cat,
                    file=file_bytes,
                    filename=filename,
                    content_type=content_type,
                    metadata=metadata or {},
                )
    finally:
        if os.path.exists(path):
            os.remove(path)


# receive files via http endpoint
@router.post("/batch", response_model=Dict[str, UploadSingleFileResponse])
@router.post("/batch/{chat_id}", response_model=Dict[str, UploadSingleFileResponse])
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile],
    metadata: str = Form(
        default="{}",
        description="Metadata to be stored where each key is the name of a file being uploaded, and the corresponding value is another dictionary containing metadata specific to that file. "
                    "Since we are passing this along side form data, metadata must be a JSON string (use `json.dumps(metadata)`)."
    ),
    info: AuthorizedInfo = check_permissions(AuthResource.UPLOAD, AuthPermission.WRITE),
) -> Dict[str, UploadSingleFileResponse]:
    """Batch upload multiple files containing text (.txt, .md, .pdf, etc.). File content will be extracted and segmented into chunks.
    Chunks will be then vectorized and stored into documents memory.

    Note
    ----------
    `metadata` must be passed as a JSON-formatted string into the form data.
    This is necessary because the HTTP protocol does not allow file uploads to be sent as JSON.
    The maximum number of files you can upload is 1000.

    Example
    ----------
    ```
    files = []
    files_to_upload = {"sample.pdf":"application/pdf", "sample.txt":"application/txt"}

    for file_name in files_to_upload:
        content_type = files_to_upload[file_name]
        file_path = f"tests/mocks/{file_name}"
        files.append(  ("files", ((file_name, open(file_path, "rb"), content_type))) )


    metadata = {
        "sample.pdf":{
            "source": "sample.pdf",
            "title": "Test title",
            "author": "Test author",
            "year": 2020
        },
        "sample.txt":{
            "source": "sample.txt",
            "title": "Test title",
            "author": "Test author",
            "year": 2021
        }
    }

    # upload file endpoint only accepts form-encoded data
    payload = {
        "metadata": json.dumps(metadata)
    }

    response = requests.post(
        "http://localhost:1865/rabbithole/batch",
        files=files,
        data=payload
    )
    ```
    """
    async def task(file_name: str, content: bytes):
        temp_path = await write_temp_file(file_name, content)
        # if file_name in dictionary pass the stringified metadata, otherwise pass an empty dictionary-like string
        metadata_dict_current = metadata_dict[file_name] if file_name in metadata_dict else {}
        await _on_upload_single_file(info=info, path=temp_path, filename=file_name, metadata=metadata_dict_current)

    log.info(f"Uploading {len(files)} files down the rabbit hole")

    response = {}
    metadata_dict = json.loads(metadata)

    for file in files:
        filename = file.filename
        run_background_task(background_tasks, task, file_name=filename, content=await file.read())
        response[filename] = UploadSingleFileResponse(
            filename=filename, content_type=file.content_type, info="File is being ingested asynchronously"  # type: ignore[arg-type]
        )

    return response


@router.post("/web", response_model=UploadUrlResponse)
@router.post("/web/{chat_id}", response_model=UploadUrlResponse)
async def upload_url(
    background_tasks: BackgroundTasks,
    upload_config: UploadURLConfig,
    info: AuthorizedInfo = check_permissions(AuthResource.UPLOAD, AuthPermission.WRITE),
) -> UploadUrlResponse:
    """Upload an url. Website content will be extracted and segmented into chunks.
    Chunks will be then vectorized and stored into documents memory."""
    try:
        # Send a HEAD request to the specified URL
        async with httpx.AsyncClient() as client:
            response = await client.head(
                upload_config.url, headers={"User-Agent": "Magic Browser"}, follow_redirects=True
            )

            response.raise_for_status()

            # resolve the configured ingestion engine and upload the URL in the
            # background through it (a plugin may replace the flow; the default
            # engine wraps the original rabbit_hole.ingest_file)
            from cat.services.factory.ingestion import resolve_ingestion_engine

            engine = await resolve_ingestion_engine(info.lizard)
            ingest_callable = engine.ingest_file if engine is not None else info.lizard.rabbit_hole.ingest_file
            run_background_task(
                background_tasks,
                ingest_callable,
                cat=info.stray_cat or info.cheshire_cat,
                file=upload_config.url,
                metadata=upload_config.metadata,
            )
            return UploadUrlResponse(url=upload_config.url, info="URL is being ingested asynchronously")
    except httpx.HTTPError as e:
        raise CustomValidationException(f"Unable to reach the URL {upload_config.url}: {e}")


@router.post("/memory", response_model=UploadSingleFileResponse)
async def upload_memory(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.WRITE),
) -> UploadSingleFileResponse:
    """Upload a memory json file to the cat memory"""
    # Get file mime type
    content_type, _ = mimetypes.guess_type(file.filename)  # type: ignore[assignment]
    log.info(f"Uploading {content_type} down the rabbit hole")
    if content_type != "application/json":
        raise CustomValidationException(
            f'MIME type {content_type} not supported. Admitted types: "application/json"'
        )

    # Ingest memories in background and notify client
    run_background_task(
        background_tasks,
        info.lizard.rabbit_hole.ingest_memory,
        cat=info.cheshire_cat,
        file=BytesIO(await file.read()),
        filename=file.filename,
    )

    # reply to client
    return UploadSingleFileResponse(
        filename=file.filename, content_type=file.content_type, info="Memory is being ingested asynchronously",  # type: ignore[arg-type]
    )


@router.get("/allowed-mimetypes", response_model=AllowedMimeTypesResponse)
async def get_allowed_mimetypes(
    info: AuthorizedInfo = check_permissions(AuthResource.UPLOAD, AuthPermission.WRITE),
) -> AllowedMimeTypesResponse:
    """Retrieve the allowed mimetypes that can be ingested by the Rabbit Hole"""
    file_handlers = await info.cheshire_cat.file_handlers()  # type: ignore[union-attr]
    return AllowedMimeTypesResponse(allowed=list(file_handlers.keys()))


@router.get("/web", response_model=List[str])
async def get_source_urls(
    info: AuthorizedInfo = check_permissions(AuthResource.UPLOAD, AuthPermission.READ),
) -> List[str]:
    """Retrieve the list of source URLs that have been uploaded to the Rabbit Hole"""
    collection = str(VectorMemoryType.DECLARATIVE if not info.stray_cat else VectorMemoryType.EPISODIC)

    # Get all points
    memory_points, _ = await info.cheshire_cat.vector_memory_handler.get_all_tenant_points_from_web(collection)

    # retrieve all the memory points where the metadata["source"] is a URL
    result = []
    for memory_point in memory_points:
        metadata = memory_point.payload.get("metadata", {})  # type: ignore[union-attr]
        if info.stray_cat and metadata.get("chat_id") != info.stray_cat.id:
            continue

        source = metadata.get("source")
        if source is None or not source.startswith("http"):
            continue

        result.append(source)

    return result


@router.post("/", response_model=UploadSingleFileResponse)
@router.post("/{chat_id}", response_model=UploadSingleFileResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    metadata: str = Form(
        default="{}",
        description="Metadata to be stored with each chunk (e.g. author, category, etc.). Since we are passing this along side form data, must be a JSON string (use `json.dumps(metadata)`)."
    ),
    info: AuthorizedInfo = check_permissions(AuthResource.UPLOAD, AuthPermission.WRITE),
) -> UploadSingleFileResponse:
    """Upload a file containing text (.txt, .md, .pdf, etc.). File content will be extracted and segmented into chunks.
    Chunks will be then vectorized and stored into documents memory.

    Note
    ----------
    `metadata` must be passed as a JSON-formatted string into the form data.
    This is necessary because the HTTP protocol does not allow file uploads to be sent as JSON.

    Example
    ----------
    ```
    content_type = "application/pdf"
    file_name = "sample.pdf"
    file_path = f"tests/mocks/{file_name}"
    with open(file_path, "rb") as f:
        files = {"file": (file_name, f, content_type)}

        metadata = {
            "source": "sample.pdf",
            "title": "Test title",
            "author": "Test author",
            "year": 2020,
        }
        # upload file endpoint only accepts form-encoded data
        payload = {
            "metadata": json.dumps(metadata)
        }

        response = requests.post(
            "http://localhost:1865/rabbithole/",
            files=files,
            data=payload
        )
    ```
    """
    async def task(content: bytes):
        temp_path = await write_temp_file(filename, content)  # type: ignore[arg-type]
        await _on_upload_single_file(
            info=info, path=temp_path, filename=filename, metadata=json.loads(metadata) if metadata else {},  # type: ignore[arg-type]
        )

    filename = file.filename
    run_background_task(background_tasks, task, content=await file.read())

    return UploadSingleFileResponse(
        filename=filename, content_type=file.content_type, info="File is being ingested asynchronously",  # type: ignore[arg-type]
    )
