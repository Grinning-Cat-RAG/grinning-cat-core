"""
Access-token (JWT) and refresh-token management for the local identity provider.

Access token
    Short-lived, signed JWT (HS256) carrying `sub`, `iss`, `aud`, `iat`, `nbf`, `exp`, `jti`, `typ=access`
    and the `agents` claim (list of JSON strings `{"agent_name": ..., "user": {...}}`) that binds the token
    to the specific (agent, user_id) pairs whose credentials were verified at login.

Refresh token
    Opaque random string (never a JWT), stored in Redis only as its SHA-256 hash. Every refresh rotates it
    (one-time use). Reusing an already-rotated token is treated as theft: the whole token family (session)
    is revoked. Each refresh re-validates the users against the DB, so deleting a user or changing their
    password ends the session at the next refresh.
"""
import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import jwt

from cat.env import get_env, get_env_float
from cat.env_check import EnvironmentConfigError, jwt_secret_problem
from cat.log import log

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "grinning-cat"
JWT_AUDIENCE = "grinning-cat-api"
ACCESS_TOKEN_TYPE = "access"

_REFRESH_PREFIX = "auth:refresh"
_USER_PUBLIC_EXCLUDED_KEYS = {"password"}


# ---------------------------------------------------------------------------------------------------------------------
# Secret
# ---------------------------------------------------------------------------------------------------------------------
def get_jwt_secret() -> str:
    """
    Returns the signing secret. The app refuses to start without a strong CAT_JWT_SECRET (see cat.env_check); this
    check is repeated here so that tokens are never signed or verified with a missing/weak key, not even if the
    environment changes at runtime.
    """
    secret = get_env("CAT_JWT_SECRET")
    if problem := jwt_secret_problem(secret):
        log.error(f"JWT disabled: {problem}")
        raise EnvironmentConfigError(problem)
    return secret  # type: ignore[return-value]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def access_token_ttl_seconds() -> int:
    return int(get_env_float("CAT_JWT_EXPIRE_MINUTES") * 60)  # type: ignore[operator]


def refresh_token_ttl_seconds() -> int:
    return int(get_env_float("CAT_JWT_REFRESH_EXPIRE_MINUTES") * 60)  # type: ignore[operator]


def refresh_session_max_seconds() -> int:
    return int(get_env_float("CAT_JWT_REFRESH_MAX_LIFETIME_MINUTES") * 60)  # type: ignore[operator]


# ---------------------------------------------------------------------------------------------------------------------
# Access token (JWT)
# ---------------------------------------------------------------------------------------------------------------------
def create_access_token(username: str, agents: List[str]) -> Tuple[str, int]:
    """Returns (jwt, expires_in_seconds)."""
    ttl = access_token_ttl_seconds()
    now = _now()
    payload = {
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "sub": username,
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(seconds=ttl),
        "jti": secrets.token_hex(16),
        "typ": ACCESS_TOKEN_TYPE,
        "agents": agents,
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM), ttl


def decode_access_token(token: str) -> Dict[str, Any] | None:
    """Verifies signature, algorithm, expiry, issuer, audience and token type. Returns the payload or None."""
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[JWT_ALGORITHM],  # pinned: no "none", no algorithm confusion
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError:
        log.debug("Access token expired")
        return None
    except EnvironmentConfigError:
        return None
    except jwt.InvalidTokenError as e:
        log.debug(f"Invalid access token: {e}")
        return None

    if payload.get("typ") != ACCESS_TOKEN_TYPE or not isinstance(payload.get("sub"), str):
        return None
    return payload


def get_agent_bindings(payload: Dict[str, Any]) -> Dict[str, str]:
    """Maps agent_name -> user_id from the (already verified) `agents` claim."""
    bindings: Dict[str, str] = {}
    for raw in payload.get("agents") or []:
        try:
            match = json.loads(raw) if isinstance(raw, str) else raw
            bindings[str(match["agent_name"])] = str(match["user"]["id"])
        except (ValueError, KeyError, TypeError):
            continue
    return bindings


def public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in user.items() if k not in _USER_PUBLIC_EXCLUDED_KEYS}


def build_agents_claim(matches: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    """Same format historically used by /auth/token and read by /me."""
    return [json.dumps({"agent_name": agent_name, "user": public_user(user)}) for agent_name, user in matches]


# ---------------------------------------------------------------------------------------------------------------------
# Refresh token (opaque, rotating, stored hashed in Redis)
# ---------------------------------------------------------------------------------------------------------------------
def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _password_fingerprint(user_full: Dict[str, Any]) -> str:
    # changes whenever the password changes -> invalidates refresh sessions
    return _hash(str(user_full.get("password", "")))


def _token_key(token_hash: str) -> str:
    return f"{_REFRESH_PREFIX}:token:{token_hash}"


def _used_key(token_hash: str) -> str:
    return f"{_REFRESH_PREFIX}:used:{token_hash}"


def _family_key(family_id: str) -> str:
    return f"{_REFRESH_PREFIX}:family:{family_id}"


async def _store_refresh_token(
    db, family_id: str, username: str, bindings: List[Dict[str, str]], family_expires_at: float
) -> Tuple[str, int] | None:
    ttl = min(refresh_token_ttl_seconds(), int(family_expires_at - _now().timestamp()))
    if ttl <= 0:
        return None

    token = secrets.token_urlsafe(48)
    token_hash = _hash(token)
    record = {
        "family": family_id,
        "username": username,
        "bindings": bindings,
        "family_expires_at": family_expires_at,
    }
    async with db.pipeline(transaction=True) as pipe:
        pipe.set(_token_key(token_hash), json.dumps(record), ex=ttl)
        # the family points to its only currently valid token
        pipe.set(_family_key(family_id), token_hash, ex=ttl)
        await pipe.execute()
    return token, ttl


async def _bindings_from_matches(matches: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, str]]:
    from cat.db.cruds import users as crud_users

    bindings = []
    for agent_name, user in matches:
        full = await crud_users.get_user(agent_name, user["id"], full=True)
        if not full:
            continue
        bindings.append({
            "agent_name": agent_name,
            "user_id": user["id"],
            "pwd_fp": _password_fingerprint(full),
        })
    return bindings


async def issue_refresh_token(username: str, matches: List[Tuple[str, Dict[str, Any]]]) -> Tuple[str, int] | None:
    """Starts a new session (token family) after a successful password login."""
    from cat.db.database import get_async_db

    bindings = await _bindings_from_matches(matches)
    if not bindings:
        return None
    family_expires_at = _now().timestamp() + refresh_session_max_seconds()
    return await _store_refresh_token(get_async_db(), secrets.token_hex(16), username, bindings, family_expires_at)


async def revoke_family(family_id: str) -> None:
    from cat.db.database import get_async_db

    db = get_async_db()
    current = await db.get(_family_key(family_id))
    async with db.pipeline(transaction=True) as pipe:
        pipe.delete(_family_key(family_id))
        if current:
            pipe.delete(_token_key(current))
        await pipe.execute()


async def revoke_refresh_token(token: str) -> None:
    """Logout: revokes the session the token belongs to. Idempotent, never reveals whether the token existed."""
    from cat.db.database import get_async_db

    db = get_async_db()
    token_hash = _hash(token)
    raw = await db.get(_token_key(token_hash))
    family_id = json.loads(raw)["family"] if raw else await db.get(_used_key(token_hash))
    if family_id:
        await revoke_family(family_id)


async def rotate_refresh_token(token: str) -> Tuple[str, List[str], Tuple[str, int]] | None:
    """
    Consumes a refresh token and returns (username, agents_claim, (new_refresh_token, ttl)), or None if the token is
    invalid, expired, revoked or reused. Reuse of a rotated token revokes the whole family.
    """
    from cat.db.cruds import users as crud_users
    from cat.db.database import get_async_db

    db = get_async_db()
    token_hash = _hash(token)

    # atomic one-time consumption: concurrent requests with the same token cannot both succeed
    raw = await db.getdel(_token_key(token_hash))
    if not raw:
        if family_id := await db.get(_used_key(token_hash)):
            log.warning(f"Refresh token reuse detected: revoking session family {family_id}")
            await revoke_family(family_id)
        return None

    record = json.loads(raw)
    family_id = record["family"]

    # the family must still exist and point to this very token (otherwise it was revoked)
    if await db.get(_family_key(family_id)) != token_hash:
        return None

    remaining = int(record["family_expires_at"] - _now().timestamp())
    if remaining <= 0:
        await revoke_family(family_id)
        return None
    await db.set(_used_key(token_hash), family_id, ex=remaining)

    # re-validate every binding against the current DB state
    valid_bindings: List[Dict[str, str]] = []
    matches: List[Tuple[str, Dict[str, Any]]] = []
    for binding in record["bindings"]:
        full = await crud_users.get_user(binding["agent_name"], binding["user_id"], full=True)
        if not full or full.get("username") != record["username"]:
            continue
        if not secrets.compare_digest(_password_fingerprint(full), binding["pwd_fp"]):
            continue
        valid_bindings.append(binding)
        matches.append((binding["agent_name"], full))

    if not valid_bindings:
        await revoke_family(family_id)
        return None

    new_refresh = await _store_refresh_token(
        db, family_id, record["username"], valid_bindings, record["family_expires_at"]
    )
    if new_refresh is None:
        await revoke_family(family_id)
        return None

    return record["username"], build_agents_claim(matches), new_refresh
