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


@pytest.mark.parametrize("char", ["è", "€", "😀"])  # 2, 3 and 4 bytes in UTF-8
def test_guess_file_type_utf8_text_with_a_character_across_the_sample_limit(char):
    """Regression: the first 8192 bytes were decoded strictly: a multi-byte character cut by the limit made a valid
    UTF-8 text file "unknown" (None), and its upload or its re-embed failed."""
    from io import BytesIO

    encoded = char.encode("utf-8")
    for cut in range(1, len(encoded)):
        data = b"a" * (8192 - cut) + encoded + b" resto del testo"
        bytes_io = BytesIO(data)
        bytes_io.seek(5)
        assert utils.guess_file_type(bytes_io) == ("text/plain", "txt"), cut
        assert bytes_io.tell() == 5, "the position of the stream is kept"


def test_guess_file_type_invalid_utf8_is_not_text():
    from io import BytesIO

    # the whole file ends with a truncated character: not cut by the sample, invalid
    assert utils.guess_file_type(BytesIO(b"a" * 8190 + "€".encode("utf-8")[:2])) == (None, None)
    assert utils.guess_file_type(BytesIO(b"testo \xff\xfe non utf-8")) == (None, None)
    # an invalid byte before the limit, in a longer file
    assert utils.guess_file_type(BytesIO(b"a" * 100 + b"\xff" + b"a" * 9000)) == (None, None)
    assert utils.guess_file_type(BytesIO("testo è".encode("utf-8"))) == ("text/plain", "txt")
