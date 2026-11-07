# Key differences of this version
The current version is a multi-tenant fork of the original [Cheshire Cat](https://www.github.com/cheshire-cat-ai/core). Here are the main differences:

## Customizable Agentic Workflows
The original version had a fixed set of agentic workflows, meaning that it could only use a specific set of workflows.
This version allows you to configure the agentic workflows per chatbot, meaning that you can use your own agentic workflows.
**The current version supports**:
- **multiple agentic workflows**, such as RAG, Code Interpreter, etc.
- **the extension of the list of allowed agentic workflows**, so you can use your own agentic workflows.

## Multimodal RAG
The original version was designed to work with text documents only.
This version is designed to work with any kind of documents, such as PDFs, Word documents, images, audio files, etc.
This is possible by using the [LangChain](https://www.langchain.com/) framework, that allows to easily integrate different types of documents.

## Multitenancy
The original version was designed to be a single-tenant application, meaning that it could only manage one chatbot at a time.
This version is designed to be multi-tenant, meaning that it can manage multiple chatbots at the same time, each with its own settings, plugins, LLMs, etc.
**The way of "injecting" the identification of the Chatbot (RAG) is simple**:
  - **in case of the HTTP API endpoints, use the `X-Agent-ID` key into the request headers or as a querystring parameter;**
  - **in case of the WebSocket API, use the `agent_id` into the URL, e.g., `/ws/{agent_id}`.**

## Cloud ready
This version can be deployed in a cluster environment. Whilst the original version stored the settings into
JSON files, **this version requires a Redis database** to store the settings, the conversation histories, the plugins and so
forth.

You can **configure the Redis database by environment variables**. The [`compose.yml`](./compose.yml) file is provided as an example. 
Hence, the current version is multi-tenant, meaning that you can manage multiple RAGs and other language models at the same time.

The Grinning Cat is still stateless, so it can be easily scaled. In case of a cluster environment, we suggest to use a shared storage,
mounted in the `cat/plugins` folder, to share the plugins.

Enabling the **MarchHare** is recommended in a cluster environment, so that the installation
of plugins can be synchronized along all the PODs and the management of the Grinning Cat can be done in a distributed way. Its
configuration is done via environment variables, too. The [`compose.yml`](./compose.yml) file is provided as an example.

## RAG Customization
The original version used a fixed RAG implementation, meaning that it could only use a specific vector database and chunking strategy.
This version allows you to configure the RAG per chatbot, meaning that you can use your own vector database and chunking strategy.
**The current version supports**:
- **multiple vector databases**, such as Qdrant, Pinecone, Weaviate, etc.
- **multiple chunking strategies**, such as text splitting or Semantic chunking.

## MCP clients
The Grinning Cat can integrate several MCP clients, such as
[LangSmith](https://www.langchain.com/langsmith) or [LlamaIndex](https://www.llamaindex.ai/).
The original version did not support any MCP client.
MCP clients can be added via plugins, by using the `@mcp_client` decorator, similarly to the `@form` decorator.

```python
from typing import List, Any

from cat import CatMcpClient, mcp_client


@mcp_client
class MyMcpClient(CatMcpClient):
    name = "your_mcp_client"
    description = "Description of your MCP Client"

    @property
    def init_args(self) -> List | Dict[str, Any]:
        """
        Define the input arguments to be passed to the constructor of the MCP client

        Returns:
            List of arguments, or a dictionary identifying each name of the arguments with the corresponding value
        """
        pass
```

## Security
The original project is developed as a framework that could be used for a personal use as well as for single-tenant production.
In the latter case, the original [documentation](https://cheshire-cat-ai.github.io/docs/) clearly states to set up a secure environment
by using an API Key. **If not configured properly (e.g. by setting up an API Key), the current version will not work, indeed**.
In this way, I tried to make the Grinning Cat more secure and production-ready.

## Customizable LLM
The original version used a fixed LLM implementation, meaning that it could only use a specific language model.
This version allows you to configure the LLM per chatbot, meaning that you can use your own language model.
**The current version supports**:
- **multiple language models**, such as OpenAI, Ollama, Google, HuggingFace, etc.
- **multiple LLMs**, meaning that you can use different language models for different chatbots.

## Customizable Storage
The original did not use any storage solution for the documents composing your RAG, meaning that you were able to store the documents into the knowledge base of each RAG, but not into a remote storage.
This version allows you to configure the storage per chatbot, meaning that you can use your own storage solution.
**The current version supports**:
- **multiple storage solutions**, such as S3, MinIO, etc.
- **multiple file managers**, meaning that you can use different file managers for different chatbots.

## Customizable Chunking strategy
The original version used a fixed chunking strategy, meaning that it could only use a specific chunking strategy.
This version allows you to configure the chunking strategy per chatbot, meaning that you can use your own chunking strategy.
**The current version supports**:
- **multiple chunking strategies**, such as text splitting or Semantic chunking;
- **multiple chunkers**, meaning that you can use different chunkers for different chatbots;
- **the extension of the list of allowed chunkers**, so you can use your own chunking strategy.

## Customizable Vector Database
The original version used a fixed vector database, meaning that it could only use a specific vector database.
This version allows you to configure the vector database per chatbot, meaning that you can use your own vector database.
**The current version supports**:
- **multiple vector databases**, such as Qdrant, Pinecone, Weaviate, etc.
- **multiple vector databases**, meaning that you can use different vector databases for different chatbots;
- **the extension of the list of allowed vector databases**, so you can use your own vector database.

## Multiple chat histories
The original version did not store the chat histories, meaning that the chat history was lost when the chatbot was restarted.
This version stores the chat histories into the Redis database, meaning that the chat history is preserved even when the chatbot is restarted.

A consequence of this is that **the current version supports multiple chat histories**, meaning that you can have different chat histories for different chatbots.

## New features
Here, I have introduced some new features and improvements, such as:
- The `Embedder` is centralized and can be used by multiple RAGs and other language models.
- New API admin endpoints allowing to configure the `Embedder`.
- New API endpoints allowing to configure the `File Manager`, per chatbot.
- New API endpoints allowing to configure the chunking strategy, per chatbot.
- New API endpoints allowing to configure the vector database, per chatbot.
- A new event system that allows you to get fine-grained control over the AI.
- **The ability to manage multiple RAGs and other language models at the same time**.
- **The current version is agnostic to the vector database and chunking strategy**, meaning that you can use your own
  vector database and chunking strategy.
- The current version provides **histories of conversations and documents**.
- **Stateful, resumable ingestion**: every source goes through recorded phases (parsing and chunking, embedding, plus the
  phases added by plugins); a change of the embedder re-embeds the kept chunks without parsing the files again, and an
  ingestion interrupted by a crash or a restart is resumed automatically (see the `Ingestion` section of the README).
