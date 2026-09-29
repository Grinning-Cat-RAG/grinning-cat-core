# Grinning Cat: AI agent as a microservice

![GitHub Repo stars](https://img.shields.io/github/stars/Grinning-Cat-RAG/grinning-cat-core?style=social)
![GitHub Release](https://img.shields.io/github/v/release/Grinning-Cat-RAG/grinning-cat-core)
![GitHub commits since latest release](https://img.shields.io/github/commits-since/Grinning-Cat-RAG/grinning-cat-core/latest)
![GitHub issues](https://img.shields.io/github/issues/Grinning-Cat-RAG/grinning-cat-core)
![GitHub Release Date](https://img.shields.io/github/release-date/Grinning-Cat-RAG/grinning-cat-core.svg)
![GitHub tag (with filter)](https://img.shields.io/github/v/tag/Grinning-Cat-RAG/grinning-cat-core)
![GitHub top language](https://img.shields.io/github/languages/top/Grinning-Cat-RAG/grinning-cat-core)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Grinning-Cat-RAG/grinning-cat-core)

# Origin

This project originated as a fork of the Cheshire Cat AI core (https://github.com/cheshire-cat-ai/core).

It has since evolved independently with significant architectural and functional changes.

# Why use the Grinning Cat?
The Grinning Cat is a framework to build custom AI agents:

- 🤖 Build your own AI agent in minutes, not months
- 🧠 Make it smart with Retrieval Augmented Generation (RAG)
- 🏆 Multi-modality, to build the RAG with any kind of documents
- 💬 Multi-tenancy, to manage multiple chatbots at the same time, each with its own settings, plugins, LLMs, etc.
- ⚡️ API first, to easily add a conversational layer to your app
- ☁️ Cloud Ready, working even with horizontal autoscaling
- 🔐 Secure by design, with API Key and granular permissions
- 🏗 Production ready, cloud native, and scalable
- 🐋 100% dockerized, to run anywhere
- 🛠 Easily extendable with plugins
- 🧩 Built-in plugins
  - 🪛 Extend core components (file managers, LLMs, vector databases)
  - ✂️ Customizable chunking and embedding
  - 🛠 Custom tools, forms, endpoints, MCP clients
  - 🪛 LLM callbacks
- 🌐 Customizable integration of **MCP clients**, such as LangSmith or LlamaIndex 
- 🏛 Easy to use Admin Panel (available with the repository [Grinning-Cat-RAG/grinning-cat-admin](https://www.github.com/Grinning-Cat-RAG/grinning-cat-admin))
- 🦄 Easy to understand [docs](https://deepwiki.com/Grinning-Cat-RAG/grinning-cat-core)
- 🌍 Supports any language model via LangChain

We are committed to openness, privacy and creativity, we want to bring AI to the long tail. If you want to know more
about our vision and values, read the [Code of Ethics](docs/CODE-OF-ETHICS.md).

# Key differences of this version
The current version is a multi-tenant fork of the original [Cheshire Cat](https://www.github.com/cheshire-cat-ai/core).
The main differences are reported in the [DIFFERENCE WITH CHESHIRECAT report](docs/DIFFERENCES-WITH-CHESHIRECAT.md).

# Quickstart
To make Grinning Cat run on your machine, you just need [`docker`](https://docs.docker.com/get-docker/) installed:

```bash
docker run --rm -it -p 1865:80 \
  -e CAT_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(64))')" \
  -e CAT_REDIS_HOST=<your_redis_host> \
  ghcr.io/Grinning-Cat-RAG/grinning-cat-core:latest
```

> [!IMPORTANT]
> The Grinning Cat **refuses to start** if a fundamental environment variable is missing or invalid (see
> [Configuration and security](#configuration-and-security)). In particular `CAT_JWT_SECRET` has no default and
> `CAT_REDIS_HOST` must point to a Redis Stack instance. With `docker compose up`, export `CAT_JWT_SECRET` in your
> shell or put it in `.env` (see `.env.example`).
- Chat with the Grinning Cat by downloading the [Admin Panel](https://www.github.com/Grinning-Cat-RAG/grinning-cat-admin).
- Try out the REST API on [localhost:1865/docs](http://localhost:1865/docs).

This fork is intended as a microservice.

As a first thing, set the **Embedder** for the Grinning Cat. A favourite **LLM** must be set for each chatbot; each
chatbot can have its own language model, with custom settings.
Everything can be done via the [Admin Panel](https://www.github.com/Grinning-Cat-RAG/grinning-cat-admin) or via the REST API endpoints.

> [!IMPORTANT]
> The following `core plugins` are enabled by default:
> - `Conversation History`: to store and retrieve the conversation history;
> - `Factories`: extending objects like LLMs, Embedders, File Managers, Chunkers;
> - `Interactions`: add the interaction handler to the language model;
> - `March Hare`: handling events via RabbitMQ;
> - `Memory`: interacting with Working Memory and adding a handler to trace the activities of the Embedder;
> - `Multimodality`: a plugin that adds multimodal capabilities to the Grinning Cat framework, enabling the processing of images;
> - `White Rabbit`: cron and schedule tasks;
> - `Why`: add the context and the reasoning behind the answers of the LLM.
> - `Analytics`: recover the analytics data about the usage of the Grinning Cat, which depends on the `Interactions` and `Memory` plugins.
>
> You can disable one or more (e.g., `March Hare` if you don't need to autoscale over cloud PODs) by using the Admin Toggle endpoint.

Enjoy the Grinning Cat!

# Configuration and security

All settings are environment variables; the full list with defaults is in [`.env.example`](.env.example).

## Startup validation and health checks
At startup (both when the FastAPI app is created and in its lifespan), the fundamental variables are validated. Any
error stops the process with an explicit message in the logs:

| Variable                                                                                           | Rule                                                                                    |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `CAT_JWT_SECRET`                                                                                   | **required**, at least 32 characters, not a known default (e.g. `this_is_a_secret_key`) |
| `CAT_REDIS_HOST`                                                                                   | required                                                                                |
| `CAT_REDIS_PORT` / `CAT_REDIS_DB`                                                                  | valid integers (port 1-65535, db >= 0)                                                  |
| `CAT_QDRANT_HOST`                                                                                  | required                                                                                |
| `CAT_JWT_EXPIRE_MINUTES`, `CAT_JWT_REFRESH_EXPIRE_MINUTES`, `CAT_JWT_REFRESH_MAX_LIFETIME_MINUTES` | positive numbers                                                                        |
| `CAT_AUTH_*` rate limit variables                                                                  | positive integers                                                                       |
| `CAT_CORS_ALLOWED_ORIGINS`                                                                         | if set, a comma separated list of valid origins (`scheme://host[:port]`)                |

Only a warning is logged when `CAT_CRYPTO_KEY` / `CAT_CRYPTO_SALT` or `CAT_ADMIN_DEFAULT_PASSWORD` keep their public
defaults, or when `CAT_API_KEY` is shorter than 16 characters. The crypto values are not enforced because settings
already stored are encrypted with them: change them only on a fresh installation.

The same validation runs on every call to `/health/liveness` and `/health/readiness`: an invalid configuration makes
the probes answer `500` (`configuration` entity `unhealthy`). The response does not say which variable is wrong,
since these endpoints are public: the details are in the logs.

## Authentication
Requests are authenticated with **one** of:
- `Authorization: Bearer <token>`, where the token is either a JWT issued by `/auth/token` or the `CAT_API_KEY`
  (with the API key, the user is selected via the `X-User-ID` header, or the `user_id` querystring on WebSockets);
- the `jwt=<token>` cookie, only accepted from an allowed Origin (see below).

The `Authorization` header has priority; a malformed one is rejected, without falling back to the cookie.

> [!CAUTION]
> Tokens and API keys are **never** accepted in the querystring (e.g. `?token=...`), neither over HTTP nor over
> WebSocket: URLs end up in proxy and server access logs, browser history and `Referer` headers. A credential sent
> that way is ignored and the request is rejected as unauthorized. Browser WebSocket clients, which cannot set
> headers, must use the `jwt` cookie from an allowed Origin.

### Tokens
| Endpoint             | Body                                     | Result                                                              |
|----------------------|------------------------------------------|---------------------------------------------------------------------|
| `POST /auth/token`   | `{"username": "...", "password": "..."}` | `access_token`, `expires_in`, `refresh_token`, `refresh_expires_in` |
| `POST /auth/refresh` | `{"refresh_token": "..."}`               | a new access token **and** a new refresh token                      |
| `POST /auth/logout`  | `{"refresh_token": "..."}`               | `204`, the session is revoked                                       |

- The **access token** is an HS256 JWT with `iss`, `aud`, `exp`, `nbf`, `iat`, `jti` and `typ=access`. It is valid
  only for the agents (and user ids) whose password was verified at login: a user of agent A cannot impersonate a
  user with the same username on agent B. Users of the `system` agent (super-admins) keep access to every agent.
  Lifetime: `CAT_JWT_EXPIRE_MINUTES` (default 1 day; with refresh tokens, 15 minutes is recommended).
- The **refresh token** is an opaque random string, stored in Redis only as a SHA-256 hash. It is **single-use**: every
  refresh rotates it. Presenting an already used refresh token is treated as theft and revokes the whole session.
  At every refresh the user must still exist, with the same username and password: deleting the user or changing
  the password ends the session. Lifetime: `CAT_JWT_REFRESH_EXPIRE_MINUTES` of inactivity (default 7 days), capped by
  `CAT_JWT_REFRESH_MAX_LIFETIME_MINUTES` from login (default 30 days).
- Do not call `/auth/refresh` concurrently with the same refresh token (e.g. from several browser tabs): the second
  call counts as a reuse and closes the session.
- `/auth/logout` does not invalidate access tokens already issued: they stay valid until they expire.
- After a user is added to a new agent, a new login is needed to access it.
- `/me` only accepts a verified access token.

### Brute-force protection
`/auth/token` and `/auth/refresh` are rate limited with counters shared through Redis, in a fixed window of
`CAT_AUTH_RATE_LIMIT_WINDOW_SECONDS` (default 900). The checks run before the password is verified:

| Variable                         | Default | Counts                                                            |
|----------------------------------|---------|-------------------------------------------------------------------|
| `CAT_AUTH_MAX_ATTEMPTS_PER_IP`   | 30      | every login attempt from the same IP                              |
| `CAT_AUTH_MAX_FAILURES_PER_USER` | 10      | failed logins for the same username (reset by a successful login) |
| `CAT_AUTH_MAX_REFRESH_PER_IP`    | 120     | refresh calls from the same IP                                    |

Over the limit the answer is `429 Too Many Requests` with a `Retry-After` header. The per-username limit applies to
existing and non-existing usernames alike, so it does not reveal which accounts exist; as a trade-off, repeated wrong
passwords can temporarily lock a user out for at most one window.

## CORS, cookies and reverse proxies
- `CAT_CORS_ALLOWED_ORIGINS` is the allow-list for CORS **and** for cookie authentication: a `jwt` cookie sent from any
  other cross-site Origin, over HTTP or WebSocket, is ignored. Same-origin requests and requests without an `Origin`
  header (non-browser clients) are accepted. If a frontend (e.g. the admin panel) is served from a different origin
  and authenticates with the cookie, add its origin here.
- With `CAT_CORS_ALLOWED_ORIGINS` unset, CORS allows any origin but **without credentials**.
- Behind a reverse proxy set `CAT_HTTPS_PROXY_MODE=true` and `CAT_CORS_FORWARDED_ALLOW_IPS` to the proxy IPs: otherwise
  the rate limiter sees the proxy IP for every client, and the same-origin check cannot see the public host
  (`X-Forwarded-Host`).

# Admin panel and UI widget
You can install an admin panel by using the [`grinning-cat-admin`](https://www.github.com/Grinning-Cat-RAG/grinning-cat-admin) repository.
The admin panel is a separate project that allows you to manage the Grinning Cat and its settings, plugins, and chatbots.
It is built with Streamlit and is designed to be easy to use and customizable.

# API Usage

## For Streaming Responses (Real-time chat)
- **Use WebSocket connection** at `/ws`, `/ws/{agent_id}` or `/ws/{agent_id}/{chat_id}`; authenticate with the
`Authorization: Bearer <token or API key>` header of the handshake, or, from browsers, with the `jwt` cookie (see
[Authentication](#authentication))
- Receive tokens in real-time as they're generated: message type `chat_token` for individual tokens; message type `chat`
for complete responses

## For Non-Streaming Responses (Simple API calls)
- **Use HTTP POST** to `/message`
- Receive complete response in single API call
- Better for integrations, batch processing, or simple request/response patterns

## Conversations
The chat id is chosen by the client (`X-Chat-ID` header, `chat_id` query or path parameter). A conversation belongs to
the first user of the agent who uses its chat id, from any endpoint (a message, an upload, the file manager); every
other user of the agent is refused (`401`, as for credentials that are not valid) until the owner deletes the
conversation, since its files, its episodic memories and its deletion are identified by the chat id alone. The system
users (administrators of every agent) access every conversation, without owning it. Deleting a conversation removes its files, its
episodic memories and its history; deleting a user removes all their conversations.

# Webhooks

Some activities are currently asynchronous: you can register a webhook to be notified when they are completed.
Currently, the following events are supported:
- `knowledge_source_loaded`, triggered when a knowledge source is loaded;
- `plugin_installed`, triggered when a plugin is installed;
- `plugin_uninstalled`, triggered when a plugin is uninstalled.
- `embedder_updated`, triggered when the Embedder is updated.
- `knowledge_source_files_transferred`, triggered when the file manager for a CheshireCat has been changed and the
  process of file transferring from the previous storage to the new one has been completed.
- `vector_memory_files_transferred`, triggered when the vector database for a CheshireCat has been changed and the
  process of point transferring from the previous vector database to the new one has been completed.

To register a webhook, use the `/webhooks` endpoint with a `POST` request. You can specify the event type and the URL
to be called when the event occurs. To register a webhook,, you need to provide the following parameters:
- `event`: the event type to listen for (e.g., `knowledge_source_loaded`, etc.);
- `url`: the URL to be called when the event occurs;
- `header_key`: the header key to be used for authentication;
- `secret`: the secret to be used for authentication.

Likewise you can register a webhook, you can delete it by using the `/webhooks` endpoint with a `DELETE` request and
the same payload as the `POST` request.

The webhook will be called by the Grinning Cat with a `POST` request, authenticated with the provided header and secret
and containing one of the following payloads:
- `knowledge_source_loaded`:
```json
{
  "agent": <the agent id>,
  "chat": <the chat id>,
  "source": <the knowledge source id>,
  "points": <the list of metadata of the stored points>,
  "success": <true if the operation was successful, false otherwise>
}
```
- `plugin_installed`:
```json
{
  "plugin_id": <the id of the installed plugin,
  "success": <true if the operation was successful, false otherwise>
}
```
- `plugin_uninstalled`:
```json
{
  "plugin_id": <the id of the uninstalled plugin>,
  "success": <true if the operation was successful, false otherwise>
}
```
- `embedder_updated`:
```json
{
  "success": <true if the operation was successful, false otherwise>
}
```
- `after_file_manager_transfer_on_agent`:
```json
{
  "agent": <the agent id>,
  "success": <true if the operation was successful, false otherwise>
}
```
- `after_vector_memory_transfer_on_agent`:
```json
{
  "agent": <the agent id>,
  "success": <true if the operation was successful, false otherwise>
}
```

> [!IMPORTANT]
> If you do not use the API Key to communicate with the Grinning Cat, you need to be authenticated as an user with
> SYSTEM:WRITE permission to register or delete webhooks.
> 
> Do not forget to specify the `X-Agent-ID` header for registering webhooks for the `knowledge_source_loaded` event.

# Compatibility 
This new version is no more completely compatible with the original version, since the architecture has been changed.
Please, refer to [COMPATIBILITY.md](docs/COMPATIBILITY.md) for more information.

# Best practices

## Custom endpoints and permissions

When implementing custom endpoints, you can use the `@endpoint` decorator to create a new endpoint. Please, refer to the
[documentation](https://deepwiki.com/Grinning-Cat-RAG/grinning-cat-core) for more information.

> [!IMPORTANT]
> **Each implemented custom endpoint must use the `check_permissions` method to authenticate**. See this
[`example`](https://github.com/Grinning-Cat-RAG/grinning-cat-core/blob/main/tests/mocks/mock_plugin/mock_endpoint.py#L28).

## Minimal plugin example

<details>
    <summary>
        Hooks (events)
    </summary>

```python
from cat import hook


# hooks are an event system to get fine-grained control over your assistant
@hook
def agent_prompt_prefix(prefix, cat):
    prefix = """You are Marvin the socks seller, a poetic vendor of socks.
You are an expert in socks, and you reply with exactly one rhyme.
"""
    return prefix
```
</details>

<details>
    <summary>
        Tools
    </summary>

```python
from cat import tool


# langchain inspired tools (function calling)
@tool
def socks_prices(color, cat):
    """How much do socks cost? Input is the sock color."""
    prices = {
        "black": 5,
        "white": 10,
        "pink": 50,
    }

    price = prices.get(color, 0)
    return f"{price} bucks, meeeow!" 
```
</details>

<details>
    <summary>
        Conversational Forms
    </summary>

```python
from enum import Enum
from pydantic import BaseModel, Field

from cat import CatForm, form


class PizzaBorderEnum(Enum):
    HIGH = "high"
    LOW = "low"


# simple pydantic model
class PizzaOrder(BaseModel):
    pizza_type: str
    pizza_border: PizzaBorderEnum
    phone: str = Field(max_length=10)


@form
class PizzaForm(CatForm):
    name = "pizza_order"
    description = "Pizza Order"
    model_class = PizzaOrder
    examples = ["order a pizza", "I want pizza"]
    stop_examples = [
        "stop pizza order",
        "I do not want a pizza anymore",
    ]

    ask_confirm: bool = True

    def submit(self, form_data) -> str:
        return f"Form submitted: {form_data}"
```
</details>

<details>
    <summary>
        MCP Clients
    </summary>

```python
from cat import hook, log, mcp_client, plugin, CatMcpClient


# 1. Define your MCP client
@mcp_client
class WeatherMcpClient(CatMcpClient):
    """MCP client for weather information"""
    
    @property
    def init_args(self):
        # Return the connection parameters for your MCP server
        return {
            "server_url": "http://localhost:3000",
            # or for stdio: ["python", "path/to/mcp_server.py"]
        }

# 2. (Optional) Settings for your plugin
@plugin
def settings_schema():
  return {
    "mcp_server_url": {
      "title": "MCP Server URL",
      "type": "string",
      "default": "http://localhost:3000"
    }
  }
```
</details>

> [!IMPORTANT]
> A new feature has been added to the plugins of the Grinning Cat: the possibility to **list the dependencies on other plugins**. This feature allows specifying that a plugin requires other plugins to be installed to work properly. This feature is optional, but it is recommended to use it to avoid issues with missing dependencies. To specify the dependencies of a plugin, you can use the `dependencies` attribute in the `plugin.json` file, listing the names of the plugins that the current plugin requires.

# Docs and Resources

**For your PHP based projects**, I developed a [PHP SDK](https://www.github.com/Grinning-Cat-RAG/grinning-cat-php-sdk) that allows you to
easily interact with the Cat. Please, refer to the [SDK documentation](https://www.github.com/Grinning-Cat-RAG/grinning-cat-php-sdk/blob/master/README.md) for more information.

**For your Node.js / React.js / Vue.js based projects**, I developed a [Typescript library](https://www.github.com/Grinning-Cat-RAG/grinning-cat-typescript-client) that allows you to
easily interact with the Grinning Cat. Please, refer to the [library documentation](https://www.github.com/Grinning-Cat-RAG/grinning-cat-typescript-client/blob/master/README.md) for more information.

List of resources:
- [Official Documentation](https://deepwiki.com/Grinning-Cat-RAG/grinning-cat-core) of the current fork
- [PHP SDK](https://www.github.com/Grinning-Cat-RAG/grinning-cat-php-sdk)
- [Typescript SDK](https://www.github.com/Grinning-Cat-RAG/grinning-cat-typescript-client)
- [Python SDK](https://www.github.com/Grinning-Cat-RAG/grinning-cat-python-sdk)
- [Tutorial - Write your first plugin](https://cheshirecat.ai/write-your-first-plugin/)

# Roadmap & Contributing

All contributions are welcome! Fork the project, create a branch, and make your changes.
Then, follow the [contribution guidelines](docs/CONTRIBUTING.md) to submit your pull request.

If you like this fork, give it a star ⭐! It is very important to have your support. Thanks again!🙏

# License and trademark

Code is licensed under [GPL3](LICENSE).  
The Grinning Cat AI logo and name are property of Piero Savastano (founder and maintainer). The current fork is created,
refactored and maintained by [Matteo Cacciola](mailto:matteo.cacciola@gmail.com).
