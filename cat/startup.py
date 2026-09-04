from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from scalar_fastapi import get_scalar_api_reference

from cat.db.database import get_async_db
from cat.env import get_env
from cat.exceptions import (
    CustomForbiddenException,
    CustomNotFoundException,
    CustomUnauthorizedException,
    CustomValidationException,
    LoadMemoryException,
    ManagementModeException,
)
from cat.log import log
from cat.routes import (
    agentic_workflow,
    auth,
    auth_handler,
    base,
    chunker,
    context_retriever,
    embedder,
    file_manager,
    llm,
    plugins,
    rabbit_hole,
    users,
    utilities,
    vector_database,
    websocket,
)
from cat.routes.openapi import get_openapi_configuration_function
from cat.routes.routes_utils import shutdown_app, startup_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    #       ^._.^
    #
    # loads Manager and plugins
    # Every endpoint can access the manager instance via request.app.state.lizard
    # - Not using middleware because I can't make it work with both http and websocket;
    # - Not using "Depends" because it only supports callables (not instances)
    # - Starlette allows this: https://www.starlette.io/applications/#storing-state-on-the-app-instance

    await startup_app(app)

    # startup message with admin, public and swagger addresses
    log.welcome()

    yield

    await shutdown_app(app)

    await get_async_db().aclose()


def custom_generate_unique_id(route: APIRoute):
    return f"{route.name}"


# REST API
def create_app() -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        generate_unique_id_function=custom_generate_unique_id,
        docs_url=None,
        redoc_url=None,
        title="Grinning Cat API",
        license_info={"name": "GPL-3", "url": "https://www.gnu.org/licenses/gpl-3.0.en.html"},
    )

    # Configures the CORS middleware for the FastAPI app
    cors_enabled = get_env("CAT_CORS_ENABLED")
    if cors_enabled == "true":
        cors_allowed_origins_str = get_env("CAT_CORS_ALLOWED_ORIGINS")
        origins = cors_allowed_origins_str.split(",") if cors_allowed_origins_str else ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add routers to the middleware stack.
    app.include_router(base.router)
    app.include_router(agentic_workflow.router)
    app.include_router(auth_handler.router)
    app.include_router(embedder.router)
    app.include_router(chunker.router)
    app.include_router(context_retriever.router)
    app.include_router(file_manager.router)
    app.include_router(llm.router)
    app.include_router(plugins.router)
    app.include_router(rabbit_hole.router)
    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(utilities.router)
    app.include_router(vector_database.router)
    app.include_router(websocket.router)


    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        log.error(f"An unexpected error occurred: {exc}")
        return JSONResponse(status_code=500, content={"detail": str(exc)})


    @app.exception_handler(ValueError)
    async def value_error_exception_handler(request, exc):
        log.error(f"An unexpected value error occurred: {exc}")
        return JSONResponse(status_code=500, content={"detail": str(exc)})


    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        # Convert error objects to JSON-serializable format
        serializable_errors = []
        for error in exc.errors():
            serializable_error = {
                "loc": error.get("loc", []),
                "msg": str(error.get("msg", "")),
                "type": str(error.get("type", ""))
            }
            # Add context if it exists and is serializable
            if "ctx" in error:
                try:
                    serializable_error["ctx"] = error["ctx"]
                except (TypeError, ValueError):
                    # If ctx is not serializable, convert to string
                    serializable_error["ctx"] = str(error["ctx"])

            serializable_errors.append(serializable_error)

        return JSONResponse(
            status_code=400,
            content={"detail": serializable_errors}
        )


    @app.exception_handler(LoadMemoryException)
    async def load_memory_exception_handler(request, exc):
        log.error(exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})


    @app.exception_handler(CustomValidationException)
    async def custom_validation_exception_handler(request, exc):
        log.error(exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})


    @app.exception_handler(CustomNotFoundException)
    async def custom_not_found_exception_handler(request, exc):
        log.error(exc)
        return JSONResponse(status_code=404, content={"detail": str(exc)})


    @app.exception_handler(CustomForbiddenException)
    async def custom_forbidden_exception_handler(request, exc):
        log.error(exc)
        return JSONResponse(status_code=403, content={"detail": str(exc)})


    @app.exception_handler(ManagementModeException)
    async def management_mode_exception_handler(request, exc):
        log.info("System in management mode")
        return JSONResponse(status_code=403, content={"detail": str(exc)})


    @app.exception_handler(CustomUnauthorizedException)
    async def custom_unauthorized_exception_handler(request, exc):
        log.error(exc)
        return JSONResponse(status_code=401, content={"detail": str(exc)})


    # openapi customization
    app.openapi = get_openapi_configuration_function(app)

    if get_env("CAT_DEBUG") == "true":
        @app.get("/docs", include_in_schema=False)
        async def scalar_docs():
            return get_scalar_api_reference(
                openapi_url=app.openapi_url,
                title=app.title,
                scalar_favicon_url="https://cheshirecat.ai/wp-content/uploads/2023/10/Logo-Cheshire-Cat.svg",
            )

    return app

grinning_cat_api = create_app()
