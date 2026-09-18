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
