from typing import Dict, List, Annotated
from fastapi import APIRouter, Depends, Request, Response

from cat.auth.permissions import get_full_permissions
from cat.auth.rate_limit import (
    check_login_allowed,
    check_refresh_allowed,
    register_login_failure,
    register_login_success,
)
from cat.auth.tokens import (
    build_agents_claim,
    create_access_token,
    issue_refresh_token,
    revoke_refresh_token,
    rotate_refresh_token,
)
from cat.exceptions import CustomUnauthorizedException
from cat.routes.routes_utils import UserCredentials, JWTResponse, RefreshTokenRequest, authenticate_credentials
from cat.services.redis_search import RedisSearchService, get_redis_search_service

router = APIRouter(tags=["User Auth"], prefix="/auth")


def _no_store(response: Response) -> None:
    # tokens must never be cached by browsers or proxies
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


@router.get("/available-permissions", response_model=Dict[str, List[str]])
async def get_available_permissions() -> Dict[str, List[str]]:
    """Returns all available resources and permissions."""
    permissions = get_full_permissions()
    return {resource: perms for resource, perms in permissions.items()}


@router.post("/token", response_model=JWTResponse)
async def auth_token(
    credentials: UserCredentials,
    request: Request,
    response: Response,
    redis_search_service: Annotated[RedisSearchService, Depends(get_redis_search_service)],
) -> JWTResponse:
    """Endpoint called from client to get a JWT (+ refresh token) from the local identity provider.
    It receives username and password as JSON body, validates credentials and issues the tokens.
    """
    _no_store(response)
    # brute-force protection: checked before any password verification
    await check_login_allowed(request, credentials.username)
    try:
        matches = await authenticate_credentials(credentials, redis_search_service)
    except CustomUnauthorizedException:
        await register_login_failure(credentials.username)
        raise
    await register_login_success(credentials.username)

    access_token, expires_in = create_access_token(credentials.username, build_agents_claim(matches))
    refresh = await issue_refresh_token(credentials.username, matches)

    return JWTResponse(
        access_token=access_token,
        expires_in=expires_in,
        refresh_token=refresh[0] if refresh else None,
        refresh_expires_in=refresh[1] if refresh else None,
    )


@router.post("/refresh", response_model=JWTResponse)
async def auth_refresh(payload: RefreshTokenRequest, request: Request, response: Response) -> JWTResponse:
    """Exchanges a valid refresh token for a new access token AND a new refresh token (rotation).
    The submitted refresh token is consumed: reusing it later revokes the whole session.
    """
    _no_store(response)
    await check_refresh_allowed(request)
    result = await rotate_refresh_token(payload.refresh_token)
    if result is None:
        raise CustomUnauthorizedException("Invalid refresh token")

    username, agents_claim, (new_refresh_token, refresh_expires_in) = result
    access_token, expires_in = create_access_token(username, agents_claim)

    return JWTResponse(
        access_token=access_token,
        expires_in=expires_in,
        refresh_token=new_refresh_token,
        refresh_expires_in=refresh_expires_in,
    )


@router.post("/logout", status_code=204)
async def auth_logout(payload: RefreshTokenRequest) -> Response:
    """Revokes the session (refresh token family). Always 204, so it does not reveal whether the token existed.
    Already-issued access tokens stay valid until their (short) expiry."""
    await revoke_refresh_token(payload.refresh_token)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
