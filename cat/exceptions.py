class LoadMemoryException(Exception):
    pass


class VectorMemoryError(Exception):
    pass


class CustomValidationException(Exception):
    pass


class CustomNotFoundException(Exception):
    pass


class CustomForbiddenException(Exception):
    pass


class ManagementModeException(CustomNotFoundException):
    """Raised when the instance is in management mode (mgmt_message plugin):
    every route other than the plugin's own ones behaves as if it did not
    exist.

    It is a 404 like CustomNotFoundException, but enables clients to
    distinguish a deliberate management gate from a genuinely missing route,
    and lets the core log it at INFO level instead of ERROR.
    """


class CustomUnauthorizedException(Exception):
    pass


class CustomTooManyRequestsException(Exception):
    def __init__(self, message: str = "Too Many Requests", retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class UnknownAgentException(ValueError):
    """
    Raised when the requested agent does not exist. It is a `ValueError` so that callers that do not care about the
    distinction keep working; the auth layer turns it into a 401 to avoid disclosing which agents exist.
    """
