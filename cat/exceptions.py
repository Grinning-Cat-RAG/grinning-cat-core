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


class ManagementModeException(CustomForbiddenException):
    """Raised when the instance is in management mode (mgmt_message plugin)
    and a principal without SYSTEM permission tries to access the app.

    It is a 403 like CustomForbiddenException, but enables clients to
    distinguish a deliberate management gate from a generic permission error,
    and lets the core log it at INFO level instead of ERROR.
    """


class CustomUnauthorizedException(Exception):
    pass
