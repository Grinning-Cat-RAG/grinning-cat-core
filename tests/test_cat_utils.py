import os
import pytest

from cat import utils

from tests.utils import agent_id


def test_get_base_path():
    assert utils.get_base_path() == "/app/cat/"


def test_get_plugin_path():
    # plugin folder is "cat/plugins/" in production, "tests/mocks/mock_plugin_folder/" during tests
    assert utils.get_plugins_path() == "tests/mocks/mock_plugin_folder/"


def test_get_data_path(client):
    assert utils.get_data_path() == os.path.join(utils.get_project_path(), "data")


def test_parse_json():
    json_string = """{
    "a": 2
}"""

    expected_json = {"a": 2}

    prefixed_json = "anything \n\t```json\n" + json_string
    assert utils.parse_json(prefixed_json) == expected_json

    suffixed_json = json_string + "\n``` anything"
    assert utils.parse_json(suffixed_json) == expected_json

    unclosed_json = """{"a":2"""
    assert utils.parse_json(unclosed_json) == expected_json

    unclosed_key_json = """{"a":2, "b":"""
    assert utils.parse_json(unclosed_key_json) == expected_json

    invalid_json = """yaml is better"""
    with pytest.raises(Exception) as e:
        utils.parse_json(invalid_json)
    assert "substring not found" in str(e.value)


def test_load_settings_raise_exception(stray_no_memory):
    with pytest.raises(Exception) as e:
        stray_no_memory.plugin_manager.get_plugin()
        assert e == "get_plugin() can only be called from within a plugin"


async def test_load_settings_no_agent_key_from_stray_cat(stray_no_memory, cheshire_cat):
    original_fnc = utils.inspect_calling_folder
    utils.inspect_calling_folder = lambda: "base_plugin"

    await stray_no_memory.plugin_manager.get_plugin().load_settings()
    assert utils.inspect_calling_agent().agent_key == agent_id

    utils.inspect_calling_folder = original_fnc


async def test_load_settings_no_agent_key_from_cheshire_cat(cheshire_cat):
    original_fnc = utils.inspect_calling_folder
    utils.inspect_calling_folder = lambda: "base_plugin"

    await cheshire_cat.plugin_manager.get_plugin().load_settings()
    assert utils.inspect_calling_agent().agent_key == agent_id

    utils.inspect_calling_folder = original_fnc


ModelNamesForTest = utils.Enum("ModelNamesForTest", {"bge_base": "BAAI/bge-base-en"})


def test_get_nlp_object_name_with_enum_model_name():
    from types import SimpleNamespace

    obj = SimpleNamespace(model_name=ModelNamesForTest.bge_base)
    # cat.utils.Enum.__str__ returns .value, so the enum normalizes like its model id.
    assert utils.get_nlp_object_name(obj, "default_embedder") == "baai_bge_base_en"


def test_get_nlp_object_name_with_plain_str_model_name():
    from types import SimpleNamespace

    obj = SimpleNamespace(model_name="BAAI/bge-base-en")
    assert utils.get_nlp_object_name(obj, "default_embedder") == "baai_bge_base_en"


def test_get_nlp_object_name_with_int_model_name_does_not_raise():
    from types import SimpleNamespace

    obj = SimpleNamespace(model_name=123)
    assert utils.get_nlp_object_name(obj, "default_embedder") == "123"


def test_get_nlp_object_name_llm_path_enum_model_name():
    from typing import Any

    from cat.services.factory.llm import LargeLanguageModel

    class DummyLLM(LargeLanguageModel):
        model_name: Any = None

        @property
        def _llm_type(self):
            return "dummy"

        def _call(self, *args, **kwargs):
            return ""

    llm = DummyLLM(model_name=ModelNamesForTest.bge_base)
    assert llm.name == "baai_bge_base_en"
