import pytest
from langchain_core.language_models import BaseLanguageModel

from cat.core_plugins.base_plugin.parsers import PyMuPDFParser
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.looking_glass import MadHatter
from cat.services.factory.chunker import BaseChunker
from cat.services.factory.embedder import Embeddings
from cat.services.factory.file_manager import BaseFileManager
from cat.services.factory.vector_db import BaseVectorDatabaseHandler
from cat.services.factory.embedder import DumbEmbedder
from cat.services.factory.llm import LLMDefault

from tests.utils import just_installed_plugin


async def test_main_modules_loaded(cheshire_cat):
    assert isinstance(cheshire_cat.plugin_manager, MadHatter)
    assert isinstance(cheshire_cat.large_language_model, BaseLanguageModel)
    assert isinstance(cheshire_cat.file_manager, BaseFileManager)
    assert isinstance(cheshire_cat.chunker, BaseChunker)
    assert isinstance(await cheshire_cat.embedder(), Embeddings)
    assert isinstance(cheshire_cat.vector_memory_handler, BaseVectorDatabaseHandler)


async def test_default_llm_loaded(cheshire_cat):
    assert isinstance(cheshire_cat.large_language_model, LLMDefault)


async def test_default_embedder_loaded(lizard):
    embedder = await lizard.embedder()
    assert isinstance(embedder, DumbEmbedder)

    sentence = "I'm smarter than a random embedder BTW"
    sample_embed = DumbEmbedder().embed_query(sentence)
    out = embedder.embed_query(sentence)
    assert sample_embed == out


async def test_cheshire_cat_created_with_system_key(lizard):
    with pytest.raises(ValueError):
        await lizard.create_cheshire_cat(DEFAULT_SYSTEM_KEY)


async def test_file_handler_pdf(lizard, cheshire_cat, secure_client, secure_client_headers):
    file_handlers = await cheshire_cat.file_handlers()
    assert "application/pdf" in file_handlers
    assert file_handlers["application/pdf"].__class__.__name__ == PyMuPDFParser.__name__

    # manually install the plugin with the fake multimodal embedder that also adds a new file handler for pdfs, to check
    # that the plugin's file handler is properly registered and available in the agent's file handlers
    await just_installed_plugin(secure_client, secure_client_headers, plugin_id="mock_plugin_multimodal_embedder")

    # activate for the new agent
    await secure_client.put(
        "/plugins/toggle/mock_plugin_multimodal_embedder",
        headers=secure_client_headers | {"X-Agent-ID": cheshire_cat.agent_key}
    )

    # now, change the embedder with the fake multimodal one
    new_embedder = "EmbedderMultimodalDumbConfig"
    response = await secure_client.put(
        f"/embedder/settings/{new_embedder}", json={}, headers=secure_client_headers
    )
    assert response.status_code == 200

    file_handlers = await cheshire_cat.file_handlers()
    assert "application/pdf" in file_handlers
