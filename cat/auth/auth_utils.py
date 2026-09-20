import hmac
from typing import Dict, List
from urllib.parse import urlparse
import bcrypt
import jwt
from fastapi.requests import HTTPConnection
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel

from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.env import get_env_bool
from cat.env_check import parse_allowed_origins

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_JWT_ALGORITHM = "HS256"


class UserInfo(BaseModel):
    user_id: str
    username: str
    permissions: Dict[str, List[str]]


def is_jwt(token: str) -> bool:
    """
    Returns whether a given string is a JWT.
    """
    try:
        # Decode the JWT without verification to check its structure
        jwt.decode(token, options={"verify_signature": False})
        return True
    except InvalidTokenError:
        return False

    
def hash_password(password: str) -> str:
    try:
        # Generate a salt
        salt = bcrypt.gensalt()
        # Hash the password
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")
    except:
        # if you try something strange, you'll stay out
        return bcrypt.gensalt().decode("utf-8")


def check_password(password: str, hashed: str) -> bool:
    try:
        # Check if the password matches the hashed password
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except:
        return False

def _extract_key_from_request(request: HTTPConnection, key: str, key_header: str) -> str:
    "look for a parameter for the first found in PATH -> QUERY -> HEADER order"
    return request.path_params.get( key,
           request.query_params.get(key,
           request.headers.get(     key_header
           )))


def extract_agent_id_from_request(request: HTTPConnection) -> str | None:
    return _extract_key_from_request(request, "agent_id", "X-Agent-ID")


def extract_chat_id_from_request(request: HTTPConnection) -> str | None:
    return _extract_key_from_request(request, "chat_id", "X-Chat-ID")


async def extract_user_info_on_api_key(agent_key: str, user_id: str | None = None) -> UserInfo | None:
    from cat.db.cruds import users as crud_users

    user = None
    if user_id:
        user = await crud_users.get_user(agent_key, user_id)
    elif agent_key == DEFAULT_SYSTEM_KEY:
        # backward compatibility
        user = await crud_users.get_user_by_username(agent_key, DEFAULT_ADMIN_USERNAME)

    if not user:
        return None

    return UserInfo(
        user_id=user["id"], username=user["username"], permissions=user["permissions"] # type: ignore
    )


def secure_compare(a: str | None, b: str | None) -> bool:
    """Constant-time string comparison (prevents timing attacks on API keys)."""
    if a is None or b is None:
        return False
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def extract_token_from_request(request: HTTPConnection) -> str | None:
    """
    Extract the token from a request. The `Authorization: Bearer <token>` header has priority; otherwise the
    `jwt` cookie is used. It returns the token if it is found, otherwise it returns None.

    SECURITY RULE: credentials (JWT or API key) are NEVER read from the querystring, for HTTP nor for WebSocket.
    URLs are written to proxy/server access logs, browser history and Referer headers. Do not add such a fallback
    (tests/routes/auth/test_no_token_in_querystring.py guards this).

    Args:
        request: the Starlette request to extract the token from (HTTP or Websocket)

    Returns:
        The token if it is found, None otherwise.
    """
    authorization = request.headers.get("Authorization")
    if authorization:
        scheme, _, credentials = authorization.partition(" ")
        credentials = credentials.strip()
        # an explicit but malformed Authorization header must not silently fall back to the cookie
        if scheme.lower() != "bearer" or not credentials or " " in credentials:
            return None
        return credentials

    # parse cookies properly: the jwt cookie is not necessarily the first one, and other cookies must not
    # end up concatenated to the token
    token = request.cookies.get("jwt")
    if not token or not token.strip():
        return None

    # the browser attaches cookies to cross-site requests too (CSRF) and to cross-site WebSocket handshakes, which
    # are NOT covered by CORS (Cross-Site WebSocket Hijacking): a cookie is honoured only from an allowed Origin
    if not is_origin_allowed(request):
        from cat.log import log
        log.warning(f"Cookie authentication refused for Origin {request.headers.get('origin')!r}")
        return None
    return token.strip()


def is_origin_allowed(request: HTTPConnection) -> bool:
    """
    True if the request's Origin is in CAT_CORS_ALLOWED_ORIGINS or is the same origin as the server.
    Requests without Origin are allowed: browsers always send it on WebSocket handshakes and on cross-origin
    requests, so its absence means a same-origin navigation or a non-browser client (not exposed to CSRF).
    """
    origin = request.headers.get("origin")
    if origin is None:
        return True
    origin = origin.strip().rstrip("/").lower()
    if origin == "null":  # sandboxed iframes, file:// pages, ...
        return False

    if origin in (parse_allowed_origins() or []):
        return True

    # same-origin: the Origin host must match the Host the client connected to
    parsed = urlparse(origin)
    hosts = {request.headers.get("host", "").lower()}
    if get_env_bool("CAT_HTTPS_PROXY_MODE"):
        # behind a trusted reverse proxy the public host is in X-Forwarded-Host
        hosts.add(request.headers.get("x-forwarded-host", "").split(",")[0].strip().lower())
    return bool(parsed.netloc) and parsed.netloc in hosts - {""}
