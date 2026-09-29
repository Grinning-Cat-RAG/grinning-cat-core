from enum import Enum
from typing import Callable, List, Any
from fastapi import APIRouter, FastAPI
from starlette.routing import compile_path

from cat.log import log


# class to represent a @endpoint
class CatEndpoint:
    def __init__(
        self,
        prefix: str,
        path: str,
        function: Callable,
        methods: set[str] | List[str] | None = None,
        tags: List[str | Enum] | None = None,
        plugin_id: str | None = None,
        **kwargs,
    ):
        self.api_route = None
        self.prefix = prefix
        self.path = path
        self.function = function
        self.tags = tags
        self.methods = methods
        self.kwargs = kwargs
        self.name = self.prefix + self.path
        self.plugin_id = plugin_id

    def __repr__(self) -> str:
        return f"CustomEndpoint(path={self.name} methods={self.methods})"

    def __eq__(self, other) -> bool:
        """Two endpoints are equal if they have the same name and methods"""
        if not isinstance(other, CatEndpoint):
            return False
        return self.name == other.name and set(self.methods or []) == set(other.methods or [])

    def __hash__(self) -> int:
        """Make CustomEndpoint hashable for use in sets/dicts"""
        methods_tuple = tuple(sorted(self.methods)) if self.methods else ()
        return hash((self.name, methods_tuple))

    def activate(self, grinning_cat_api: FastAPI | None = None):
        if not grinning_cat_api:
            return

        log.info(f"Activating custom endpoint {self.methods} {self.name}")

        # Set the fastapi api_route into the Custom Endpoint
        if any(api_route.path == self.name and api_route.methods == self.methods for api_route in grinning_cat_api.routes):  # type: ignore[attr-defined]
            log.debug(f"There is already an active {self.methods} endpoint with path {self.name}")
            return

        plugins_router = APIRouter()
        plugins_router.add_api_route(
            path=self.path,
            endpoint=self.function,
            methods=self.methods,
            tags=self.tags,
            **self.kwargs,
        )

        try:
            grinning_cat_api.include_router(plugins_router, prefix=self.prefix)
        except Exception as e:
            log.error(f"Error activating custom endpoint {self.methods} {self.name}: {e}")
            return

        grinning_cat_api.openapi_schema = None  # Flush the cache of openapi schema

        # Set the fastapi api_route into the Custom Endpoint
        for api_route in grinning_cat_api.routes:
            if api_route.path == self.name and api_route.methods == self.methods:  # type: ignore[attr-defined]
                self.api_route = api_route
                break
        
        assert self.api_route.path == self.name  # type: ignore[attr-defined]

    def deactivate(self, grinning_cat_api: FastAPI | None = None):
        if not grinning_cat_api:
            return

        # Seems there is no official way to remove a route:
        # https://github.com/fastapi/fastapi/discussions/8088
        # https://github.com/fastapi/fastapi/discussions/9855
        to_remove = None
        for api_route in grinning_cat_api.routes:
            if api_route.path == self.name and api_route.methods == self.methods:  # type: ignore[attr-defined]
                to_remove = api_route
                break

        if to_remove:
            log.info(f"Deactivating custom endpoint {self.methods} {self.name}")

            grinning_cat_api.routes.remove(to_remove)
            grinning_cat_api.openapi_schema = None  # Flush the cached openapi schema

    @property
    def real_path(self) -> str:
        """
        Returns the real path of the endpoint, including the prefix.
        This is useful for logging and debugging purposes.
        """
        return f"{self.prefix}{self.path}"

    def matches(self, path: str, methods: set[str] | List[str] | None = None, method: str | None = None) -> bool:
        """
        Whether a request path (e.g. ``/items/42``) is served by this endpoint (e.g. ``/items/{item_id}``), as FastAPI
        matches it; with ``methods``, the endpoint must have exactly those methods, with ``method`` that method.
        """
        # compiled once per path (the prefix of an endpoint may change before its activation)
        if getattr(self, "_compiled", (None,))[0] != self.real_path:
            self._compiled = (self.real_path, compile_path(self.real_path)[0])
        if not self._compiled[1].match(path):
            return False
        if methods is not None and set(self.methods or []) != set(methods):
            return False
        # FastAPI serves GET for a route without methods
        return method is None or method.upper() in {m.upper() for m in self.methods or ["GET"]}

class Endpoint:
    default_prefix = "/custom"
    default_tags = ["Custom Endpoints"]

    # @endpoint decorator. Any function in a plugin decorated by @endpoint.endpoint will be exposed as FastAPI operation
    def endpoint(
        self,
        path: str,
        methods: set[str] | List[str] | None = None,
        prefix: str | None = default_prefix,
        tags: List[str | Enum] | None = None,
        **kwargs,
    ) -> Callable:
        """
        Define a custom API endpoint, parameters are the same as FastAPI path operation.
        Examples:
            .. code-block:: python
                from cat.mad_hatter.decorators import endpoint

                @endpoint.endpoint(path="/hello", methods=["GET"])
                def my_endpoint():
                    return {"Hello":"Alice"}
        """
        tags = tags or self.default_tags  # type: ignore[assignment]

        def _make_endpoint(endpoint_function: Callable):
            custom_endpoint = CatEndpoint(
                prefix=prefix,  # type: ignore[arg-type]
                path=path,
                function=endpoint_function,
                methods=set(methods),  # type: ignore[arg-type]
                tags=tags,
                **kwargs,
            )

            return custom_endpoint

        return _make_endpoint

    # Any function in a plugin decorated by @endpoint.get will be exposed as FastAPI GET operation
    def get(
        self,
        path: str,
        prefix: str | None = default_prefix,
        response_model: Any = None,
        tags: List[str | Enum] | None = None,
        **kwargs,
    ) -> Callable:
        """
        Define a custom API endpoint for GET operation, parameters are the same as FastAPI path operation.
        Examples:
            .. code-block:: python
                from cat.mad_hatter.decorators import endpoint

                @endpoint.get(path="/hello")
                def my_get_endpoint():
                    return {"Hello":"Alice"}
        """
        tags = tags or self.default_tags  # type: ignore[assignment]

        return self.endpoint(
            path=path,
            methods={"GET"},
            prefix=prefix,
            response_model=response_model,
            tags=tags,
            **kwargs,
        )

    # Any function in a plugin decorated by @endpoint.post will be exposed as FastAPI POST operation
    def post(
        self,
        path: str,
        prefix: str | None = default_prefix,
        response_model: Any = None,
        tags: List[str | Enum] | None = None,
        **kwargs,
    ) -> Callable:
        """
        Define a custom API endpoint for POST operation, parameters are the same as FastAPI path operation.
        Examples:
            .. code-block:: python

                from cat.mad_hatter.decorators import endpoint
                from pydantic import BaseModel

                class Item(BaseModel):
                    name: str
                    description: str

                @endpoint.post(path="/hello")
                def my_post_endpoint(item: Item):
                    return {"Hello": item.name, "Description": item.description}
        """
        tags = tags or self.default_tags  # type: ignore[assignment]

        return self.endpoint(
            path=path,
            methods={"POST"},
            prefix=prefix,
            response_model=response_model,
            tags=tags,
            **kwargs,
        )

    # Any function in a plugin decorated by @endpoint.put will be exposed as FastAPI PUT operation
    def put(
        self,
        path: str,
        prefix: str | None = default_prefix,
        response_model: Any = None,
        tags: List[str | Enum] | None = None,
        **kwargs,
    ) -> Callable:
        """
        Define a custom API endpoint for PUT operation, parameters are the same as FastAPI path operation.
        Examples:
            .. code-block:: python

                from cat.mad_hatter.decorators import endpoint
                from pydantic import BaseModel

                class Item(BaseModel):
                    name: str
                    description: str

                @endpoint.put(path="/hello")
                def my_put_endpoint(item: Item):
                    return {"Hello": item.name, "Description": item.description}
        """
        tags = tags or self.default_tags  # type: ignore[assignment]

        return self.endpoint(
            path=path,
            methods={"PUT"},
            prefix=prefix,
            response_model=response_model,
            tags=tags,
            **kwargs,
        )

    # Any function in a plugin decorated by @endpoint.delete will be exposed as FastAPI DELETE operation
    def delete(
        self,
        path: str,
        prefix: str | None = default_prefix,
        response_model: Any = None,
        tags: List[str | Enum] | None = None,
        **kwargs,
    ) -> Callable:
        """
        Define a custom API endpoint for DELETE operation, parameters are the same as FastAPI path operation.
        Examples:
            .. code-block:: python

                from cat.mad_hatter.decorators import endpoint
                from pydantic import BaseModel

                class Item(BaseModel):
                    name: str
                    description: str

                @endpoint.delete(path="/hello")
                def my_delete_endpoint(item: Item):
                    return {"Hello": item.name, "Description": item.description}
        """
        tags = tags or self.default_tags  # type: ignore[assignment]

        return self.endpoint(
            path=path,
            methods={"DELETE"},
            prefix=prefix,
            response_model=response_model,
            tags=tags,
            **kwargs,
        )


endpoint = None

if not endpoint:
    endpoint = Endpoint()
