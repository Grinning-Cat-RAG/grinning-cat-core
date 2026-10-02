"""The modules of a plugin are loaded once and consistently: a class imported by a sibling module is the same class."""
import shutil
import sys

from cat.looking_glass.mad_hatter.plugin import Plugin

SOURCE = "tests/mocks/mock_plugin_consistent"
PATH = "tests/mocks/mock_plugin_folder/mock_plugin_consistent"
PACKAGE = "tests.mocks.mock_plugin_folder.mock_plugin_consistent"


def copy_plugin():
    shutil.rmtree(PATH, ignore_errors=True)
    shutil.copytree(SOURCE, PATH)


def load_plugin():
    plugin = Plugin(PATH)
    plugin._load_decorated_functions()  # as on the activation of the plugin
    return plugin


def test_sibling_modules_share_their_classes():
    # regression: every module was imported and then reloaded, in the order of the files: the module importing a
    # sibling kept the classes of the first execution, the sibling had new ones (isinstance and identity broke)
    copy_plugin()
    plugin = load_plugin()
    settings, hooks = sys.modules[f"{PACKAGE}.z_settings"], sys.modules[f"{PACKAGE}.a_hooks"]
    assert hooks.ConsistentSettings is settings.ConsistentSettings
    assert plugin.settings_model() is settings.ConsistentSettings


def test_a_plugin_loaded_again_runs_its_current_code():
    copy_plugin()
    load_plugin()
    with open(f"{PATH}/z_settings.py") as f:
        code = f.read()
    with open(f"{PATH}/z_settings.py", "w") as f:
        f.write(code.replace("value: int = 1", "value: int = 2"))

    plugin = load_plugin()  # e.g. the plugin upgraded
    hook, = [h for h in plugin.hooks if h.name == "agent_prompt_prefix"]
    assert plugin.settings_model()().value == 2
    assert hook.function("prefix", None) == "prefix 2"
    assert sys.modules[f"{PACKAGE}.a_hooks"].ConsistentSettings is plugin.settings_model()
