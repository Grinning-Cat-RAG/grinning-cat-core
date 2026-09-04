"""Hooks to modify the Cat's *Authentication* flow.

Here is a collection of methods to hook into the *Authentication* pipeline.
"""

from cat import hook


@hook(priority=0)
async def auth_request(local_user, agent_id, connection, **kwargs):
    """
    Default no-op: allow the request.

    Plugins (e.g. mgmt-message) override this with higher priority to deny
    requests by returning a non-empty string, which the auth gateway turns
    into an HTTP 403 or a WebSocket close 1008.

    Args:
        local_user: the authenticated user info.
        agent_id: the agent id of the request (may be None).
        connection: the HTTP/WebSocket connection being authorized.
        **kwargs: contextual objects (e.g. ``lizard`` or ``cat``).

    Returns:
        None: allow the request.
    """
    return