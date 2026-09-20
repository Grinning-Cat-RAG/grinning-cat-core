"""
Validation of the fundamental environment variables.

- `ensure_valid_env()` is called when the FastAPI app is created and again in the lifespan startup: any error aborts
  the startup (the process exits instead of serving requests with an insecure or broken configuration).
- `HealthCheckConfig` (see cat.routes.base) re-runs the validation on every /health probe, so a misconfigured
  instance is reported as unhealthy (HTTP 500) instead of healthy.
"""
from dataclasses import dataclass, field
from typing import Callable, List
from urllib.parse import urlparse

from cat.env import get_env

# values that were shipped as defaults or are trivially guessable: never accepted as signing secret
WEAK_JWT_SECRETS = {"this_is_a_secret_key", "secret", "changeme", "change_me", "jwt_secret", "meow"}
MIN_JWT_SECRET_LENGTH = 32


class EnvironmentConfigError(RuntimeError):
    pass


@dataclass
class EnvReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def jwt_secret_problem(secret: str | None) -> str | None:
    """Returns a description of what is wrong with the JWT secret, or None if it is acceptable."""
    if secret is None or not secret.strip():
        return "CAT_JWT_SECRET is not set"
    if secret.strip().lower() in WEAK_JWT_SECRETS:
        return "CAT_JWT_SECRET uses a known default/weak value"
    if len(secret) < MIN_JWT_SECRET_LENGTH:
        return f"CAT_JWT_SECRET must be at least {MIN_JWT_SECRET_LENGTH} characters long"
    return None


def _check_number(
    report: EnvReport, name: str, cast: Callable, min_value: float | None = None, max_value: float | None = None
) -> None:
    raw = get_env(name)
    if raw is None or str(raw).strip() == "":
        report.errors.append(f"{name} is not set")
        return
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        report.errors.append(f"{name} must be a valid {cast.__name__} (got {raw!r})")
        return
    if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
        report.errors.append(f"{name} is out of range (got {raw!r})")


def _check_required(report: EnvReport, name: str) -> None:
    value = get_env(name)
    if value is None or not str(value).strip():
        report.errors.append(f"{name} is not set")


def parse_allowed_origins() -> List[str] | None:
    """Explicit CORS/Origin allow-list, normalized. None means "not configured" (wildcard)."""
    raw = get_env("CAT_CORS_ALLOWED_ORIGINS")
    if not raw:
        return None
    origins = [o.strip().rstrip("/").lower() for o in raw.split(",") if o.strip()]
    if not origins or "*" in origins:
        return None
    return origins


def validate_env() -> EnvReport:
    report = EnvReport()

    # --- authentication -------------------------------------------------------------------------------------------
    if problem := jwt_secret_problem(get_env("CAT_JWT_SECRET")):
        report.errors.append(problem)
    _check_number(report, "CAT_JWT_EXPIRE_MINUTES", float, min_value=0.01)
    _check_number(report, "CAT_JWT_REFRESH_EXPIRE_MINUTES", float, min_value=1)
    _check_number(report, "CAT_JWT_REFRESH_MAX_LIFETIME_MINUTES", float, min_value=1)
    _check_number(report, "CAT_AUTH_RATE_LIMIT_WINDOW_SECONDS", int, min_value=1)
    _check_number(report, "CAT_AUTH_MAX_ATTEMPTS_PER_IP", int, min_value=1)
    _check_number(report, "CAT_AUTH_MAX_FAILURES_PER_USER", int, min_value=1)
    _check_number(report, "CAT_AUTH_MAX_REFRESH_PER_IP", int, min_value=1)

    api_key = get_env("CAT_API_KEY")
    if api_key is not None and len(api_key) < 16:
        report.warnings.append("CAT_API_KEY is shorter than 16 characters")

    # --- storage --------------------------------------------------------------------------------------------------
    _check_required(report, "CAT_REDIS_HOST")
    _check_number(report, "CAT_REDIS_PORT", int, min_value=1, max_value=65535)
    _check_number(report, "CAT_REDIS_DB", int, min_value=0)
    _check_required(report, "CAT_QDRANT_HOST")

    # --- encryption of settings at rest -------------------------------------------------------------------------
    # only warnings: existing data may be encrypted with the defaults, changing them would make it unreadable
    _check_required(report, "CAT_CRYPTO_KEY")
    _check_required(report, "CAT_CRYPTO_SALT")
    if get_env("CAT_CRYPTO_KEY") == "grinning_cat" or get_env("CAT_CRYPTO_SALT") == "grinning_cat_salt":
        report.warnings.append("CAT_CRYPTO_KEY / CAT_CRYPTO_SALT use the public defaults")
    if get_env("CAT_ADMIN_DEFAULT_PASSWORD") == "admin":
        report.warnings.append("CAT_ADMIN_DEFAULT_PASSWORD uses the public default 'admin'")

    # --- CORS / Origin allow-list ---------------------------------------------------------------------------------
    for origin in parse_allowed_origins() or []:
        parsed = urlparse(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path or parsed.query:
            report.errors.append(f"CAT_CORS_ALLOWED_ORIGINS contains an invalid origin: {origin!r}")

    return report


def ensure_valid_env() -> EnvReport:
    """Raises EnvironmentConfigError if a fundamental variable is missing or invalid; logs the warnings."""
    from cat.log import log

    report = validate_env()
    for warning in report.warnings:
        log.warning(f"Insecure configuration: {warning}")
    if not report.ok:
        for error in report.errors:
            log.error(f"Invalid configuration: {error}")
        raise EnvironmentConfigError(
            "Refusing to start, invalid environment configuration: " + "; ".join(report.errors)
        )
    return report
