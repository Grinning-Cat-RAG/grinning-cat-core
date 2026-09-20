import json
from typing import Dict, List
import tomli
from fastapi import APIRouter, Body, Depends, Request
from fastapi_healthz import (
    HealthCheckRegistry,
    HealthCheckRedis,
    HealthCheckStatusEnum,
    HealthCheckAbstract,
    health_check_route,
)
from pydantic import BaseModel, Field

from cat import utils
from cat.auth.auth_utils import is_jwt, extract_token_from_request
from cat.auth.tokens import decode_access_token
from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
import cat.db.cruds.settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY, get_db_connection_string
from cat.env_check import validate_env
from cat.exceptions import CustomUnauthorizedException, CustomNotFoundException
from cat.looking_glass import StrayCat, ChatResponse
from cat.log import log
from cat.routes.routes_utils import log_user_agent
from cat.services.memory.messages import UserMessage

router = APIRouter()


class HealthCheckLocal(HealthCheckAbstract):
    @property
    def service(self) -> str:
        return "grinning-cat"

    @property
    def connection_uri(self) -> str:
        return utils.get_base_url()

    @property
    def tags(self) -> List[str]:
        return ["grinning-cat", "local"]

    @property
    def comments(self) -> list[str]:
        with open("pyproject.toml", "rb") as f:
            project_toml = tomli.load(f)["project"]
            return [f"version: {project_toml['version']}"]

    def check_health(self) -> HealthCheckStatusEnum:
        return HealthCheckStatusEnum.HEALTHY


class HealthCheckConfig(HealthCheckAbstract):
    """Unhealthy if a fundamental env variable is missing/invalid. Details are only logged: health endpoints are
    public, so the response never says which variable is wrong."""
    @property
    def service(self) -> str:
        return "configuration"

    @property
    def connection_uri(self) -> str | None:
        return None

    @property
    def tags(self) -> List[str]:
        return ["grinning-cat", "config"]

    @property
    def comments(self) -> list[str]:
        return []

    def check_health(self) -> HealthCheckStatusEnum:
        report = validate_env()
        if report.ok:
            return HealthCheckStatusEnum.HEALTHY
        for error in report.errors:
            log.error(f"Health check - invalid configuration: {error}")
        return HealthCheckStatusEnum.UNHEALTHY


# Add Health Checks
_healthChecks = HealthCheckRegistry()
_healthChecks.add_many([
    HealthCheckConfig(),
    HealthCheckLocal(),
    HealthCheckRedis(get_db_connection_string())
])

router.add_api_route(
    "/health/readiness",
    endpoint=health_check_route(registry=_healthChecks),
    methods=["GET"],
    name="readiness_probe",
    include_in_schema=False,
    dependencies=[Depends(log_user_agent)],
)

router.add_api_route(
    "/health/liveness",
    endpoint=health_check_route(registry=_healthChecks),
    methods=["GET"],
    name="liveness_probe",
    include_in_schema=False,
    dependencies=[Depends(log_user_agent)],
)


class User(BaseModel):
    id: str
    username: str
    permissions: Dict[str, List[str]]
    created_at: float
    updated_at: float

    def __init__(self, **data):
        permissions = data.get("permissions")
        if not permissions:
            data["permissions"] = {}
        for key, value in data["permissions"].items():
            if isinstance(value, dict):
                data["permissions"][key] = list(value.keys())

        super().__init__(**data)


class AgentMatch(BaseModel):
    agent_name: str
    user: User


class MeResponse(BaseModel):
    success: bool
    agents: List[AgentMatch] = Field(default_factory=list)
    auto_selected: bool


@router.get("/", name="index", include_in_schema=False)
async def home() -> str:
    return "We're all mad here, dear!"


@router.post("/message", response_model=ChatResponse, tags=["Message"])
async def http_chat(
    payload: Dict = Body(...),
    info: AuthorizedInfo = check_permissions(AuthResource.CHAT, AuthPermission.WRITE, is_chat=True),
) -> ChatResponse:
    """Get a response from the Cat"""
    stray_cat = info.stray_cat or await StrayCat.from_cat(user_data=info.user, cat=info.cheshire_cat)

    user_message = UserMessage(**payload)
    answer = await stray_cat.run_http(user_message)
    return answer


@router.get("/me", response_model=MeResponse)
async def me(request: Request) -> MeResponse:
    token = extract_token_from_request(request)
    if token is None:
        raise CustomUnauthorizedException("Unauthorized")

    if not is_jwt(token):
        raise CustomNotFoundException("Not Found")

    # NEVER trust an unverified token: a forged JWT would otherwise be enough to enumerate every agent
    token_info = decode_access_token(token)
    if token_info is None:
        raise CustomUnauthorizedException("Unauthorized")

    matches_raw = token_info.get("agents", [])
    if not matches_raw:
        return MeResponse(success=True, agents=[], auto_selected=False)

    valid_agents = []
    has_matching_system = False
    valid_agents_names = set()
    for match_str in matches_raw:
        match = json.loads(match_str)

        if DEFAULT_SYSTEM_KEY == match["agent_name"]:
            has_matching_system = True

        valid_agents.append(AgentMatch(
            agent_name=match["agent_name"],
            user=User(**match["user"])
        ))
        valid_agents_names.add(match["agent_name"])

    if has_matching_system:
        system_agent = [agent for agent in valid_agents if agent.agent_name == DEFAULT_SYSTEM_KEY][0]
        missing_agents = [
            AgentMatch(agent_name=agent_name, user=system_agent.user)
            for agent_name in await crud_settings.get_agents_main_keys()
            if agent_name not in valid_agents_names
        ]
        valid_agents.extend(missing_agents)

    return MeResponse(
        success=True,
        agents=valid_agents,
        auto_selected=len(valid_agents) == 1
    )
