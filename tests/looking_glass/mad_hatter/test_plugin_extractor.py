import os
import pytest

from cat.looking_glass.mad_hatter.plugin_extractor import PluginExtractor

from tests.utils import create_mock_plugin_zip


# plugin_is_flat is False: zip file contains just one folder, inside that folder we find the plugin
# plugin_is_flat is True: zip file does not contain a folder, but the plugin files directly
@pytest.mark.parametrize("plugin_is_flat", [True, False])
def test_unpackage_zip(plugin_is_flat):
    plugins_folder = "tests/mocks/mock_plugin_folder"

    zip_path = create_mock_plugin_zip(flat=plugin_is_flat)
    extractor = PluginExtractor(zip_path)
    extracted = extractor.extract(plugins_folder)
    assert extracted == plugins_folder + "/mock_plugin"
    assert os.path.exists(f"{plugins_folder}/mock_plugin")
    assert os.path.exists(f"{plugins_folder}/mock_plugin/mock_tool.py")


@pytest.mark.parametrize("plugin_is_flat", [True, False])
def test_get_id_and_extension(plugin_is_flat):
    zip_path = create_mock_plugin_zip(flat=plugin_is_flat)
    extractor = PluginExtractor(zip_path)
    assert extractor.id == "mock_plugin"
    assert extractor.extension == "zip"


def test_raise_exception_if_a_wrong_extension_is_provided():
    try:
        PluginExtractor("./tests/infrastructure/plugin.wrong")
    except Exception as e:
        assert (
            str(e)
            == "Invalid package extension. Valid extensions are: ['application/zip', 'application/x-tar']"
        )
