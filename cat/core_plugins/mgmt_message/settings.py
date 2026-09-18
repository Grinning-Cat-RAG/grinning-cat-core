"""Settings of the ``mgmt_message`` core plugin.

The plugin stores them the standard way, through the base ``Plugin``
load/save: on the system agent they land on ``system:plugins:mgmt_message``,
exactly like every other system plugin. Every read and write pins the system
agent, because these settings are global for the whole instance.
"""
from pydantic import BaseModel

from cat import plugin

MGMT_SETTING_NAME = "mgmt_message"


class PluginSettings(BaseModel):
    management_message: str = ""
    management_active: bool = False
    global_message: str = ""
    show_global_msg: bool = False


@plugin
def settings_schema() -> dict:
    return PluginSettings.model_json_schema()


@plugin
def settings_model():
    return PluginSettings
