"""
Redis-backed, fixed-window rate limiting for the authentication endpoints (shared across workers and replicas).

Checks happen BEFORE the password is verified, so bcrypt cost cannot be abused either.
- per IP: every login attempt counts (CAT_AUTH_MAX_ATTEMPTS_PER_IP), every refresh call counts
  (CAT_AUTH_MAX_REFRESH_PER_IP);
- per username: only failed logins count (CAT_AUTH_MAX_FAILURES_PER_USER), reset by a successful login. The key is
  the username hash, applied to existing and non-existing usernames alike (no account enumeration).
"""
import hashlib
from fastapi.requests import HTTPConnection

from cat.db.database import get_async_db
from cat.env import get_env_int
from cat.exceptions import CustomTooManyRequestsException
from cat.log import log

_PREFIX = "auth:ratelimit"


def _window() -> int:
    return get_env_int("CAT_AUTH_RATE_LIMIT_WINDOW_SECONDS")  # type: ignore[return-value]


def _username_key(username: str) -> str:
    digest = hashlib.sha256(username.strip().lower().encode("utf-8")).hexdigest()
    return f"{_PREFIX}:user_fail:{digest}"


def client_ip(request: HTTPConnection) -> str:
    # behind a proxy, uvicorn (CAT_HTTPS_PROXY_MODE + CAT_CORS_FORWARDED_ALLOW_IPS) already rewrites request.client
    # from X-Forwarded-For for trusted proxies only: never read the header directly here, it is client-controlled
    return request.client.host if request.client else "unknown"


async def _hit(key: str) -> tuple[int, int]:
    """Increments the counter of the current window. Returns (count, seconds_to_reset)."""
    window = _window()
    async with get_async_db().pipeline(transaction=True) as pipe:
        pipe.set(key, 0, ex=window, nx=True)  # starts the window only on first hit
        pipe.incr(key)
        pipe.ttl(key)
        _, count, ttl = await pipe.execute()
    return int(count), int(ttl) if ttl and ttl > 0 else window


async def _deny(what: str, retry_after: int) -> None:
    log.warning(f"Rate limit exceeded: {what}")
    raise CustomTooManyRequestsException("Too many attempts, try again later", retry_after=retry_after)


async def check_login_allowed(request: HTTPConnection, username: str) -> None:
    ip = client_ip(request)
    count, ttl = await _hit(f"{_PREFIX}:login_ip:{ip}")
    if count > get_env_int("CAT_AUTH_MAX_ATTEMPTS_PER_IP"):  # type: ignore[operator]
        await _deny(f"login attempts from IP {ip}", ttl)

    db = get_async_db()
    key = _username_key(username)
    failures = await db.get(key)
    if failures is not None and int(failures) >= get_env_int("CAT_AUTH_MAX_FAILURES_PER_USER"):  # type: ignore[operator]
        await _deny("failed logins for a username", max(await db.ttl(key), 1))


async def register_login_failure(username: str) -> None:
    await _hit(_username_key(username))


async def register_login_success(username: str) -> None:
    await get_async_db().delete(_username_key(username))


async def check_refresh_allowed(request: HTTPConnection) -> None:
    ip = client_ip(request)
    count, ttl = await _hit(f"{_PREFIX}:refresh_ip:{ip}")
    if count > get_env_int("CAT_AUTH_MAX_REFRESH_PER_IP"):  # type: ignore[operator]
        await _deny(f"refresh calls from IP {ip}", ttl)
