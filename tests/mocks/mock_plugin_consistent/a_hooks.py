from cat import hook

# imported from a sibling module: the same class as the one of the settings model
from .z_settings import ConsistentSettings


@hook
def agent_prompt_prefix(prefix, cat):
    return f"{prefix} {ConsistentSettings().value}"
