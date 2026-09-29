from pydantic import BaseModel

from cat import plugin


class ConsistentSettings(BaseModel):
    value: int = 1


@plugin
def settings_model():
    return ConsistentSettings
