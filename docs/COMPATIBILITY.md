# Code compatibility
The `episodic_memories` is now used to store documents uploaded on-the-fly within a chat, so continuing working for the `StrayCat`'s working memory.
The `declarative_memories` and `procedural_memories` are still available on a `CheshireCat` level.

# Compatibility with plugins
This new version is no more completely compatible with the original version, since the architecture has been deeply changed.
However, **most of the plugins developed for the original version should work with this version**.
Few plugins may require minor changes to work with this version.
In this case, please feel free to contact me for support.

## Dependencies in Plugins:
A new feature has been added to the plugins of the Grinning Cat: the possibility to list the dependencies on other plugins.
This feature allows specifying that a plugin requires other plugins to be installed to work properly.
This feature is optional, but it is recommended to use it to avoid issues with missing dependencies.
To specify the dependencies of a plugin, you can use the `dependencies` attribute in the `plugin.json` file, listing the
names of the plugins that the current plugin requires.

## List of available hooks
The Grinning Cat provides a set of hooks that can be used to customize the behavior of the AI agent. Hooks are events that can be
triggered at specific points in the conversation, allowing you to modify the behavior of the AI agent or to add custom functionality.
The list of available hooks is available in the [documentation](https://deepwiki.com/Grinning-Cat-RAG/grinning-cat-core).
The current version introduces also the following additional hooks:

### Bill The Lizard
- `before_lizard_bootstrap`: to add custom logic before the Lizard bootsraps the conversation
- `after_lizard_bootstrap`: to add custom logic after the Lizard bootsraps the conversation
- `after_cheshire_cat_creation`: to add custom logic after the Cheshire Cat is created
- `lizard_notify_plugin_installation`: to add custom logic when a plugin is installed
- `lizard_notify_plugin_uninstallation`: to add custom logic when a plugin is uninstalled
- `before_lizard_shutdown`: to add custom logic before the Lizard shuts down
- `after_all_cheshire_cats_embedded`: to add custom logic after all Cheshire Cats are embedded, executed after the
   embedder has been updated

### CheshireCat:
- `after_plugin_toggling_on_agent`: to add custom logic after a plugin is enabled on the Cheshire Cat
- `after_plugin_settings_update`: to add custom logic after a plugin settings are updated
- `after_file_manager_transfer_on_agent`: to add custom logic after a file manager is updated on the Cheshire Cat and
  all the files have been transferred to the new file manager
- `after_vector_memory_transfer_on_agent`: to add custom logic after a vector memory is updated on the Cheshire Cat and
  all the documents have been transferred to the new vector memory

### Factories:
- `factory_allowed_file_managers`: to extend the list of allowed file managers
- `factory_allowed_chunkers`: to extend the list of allowed chunkers
- `factory_allowed_vector_databases`: to extend the list of allowed vector databases (so allowing to use your own vector database)
- `factory_allowed_agentic_workflows`: to extend the list of allowed agentic workflows (so allowing to use your own agentic workflow)

### Callbacks:
- `llm_callbacks`: add custom callbacks to the LangChain LLM/ChatModel

### RabbitHole:
- `before_rabbithole_splits_documents` replaces the removed `before_rabbithole_splits_text`

### Memory:
- `before_cat_recalls_memories` replaces the removed `before_cat_recalls_declarative_memories`

## List of suppressed hooks

### Agent:
- `before_agent_starts`, please use `before_cat_reads_message` instead
- `cat_recall_query`

### RabbitHole:
- `before_rabbithole_splits_text`
- `rabbithole_instantiates_splitter`
- `before_rabbithole_insert_memory`, please use `before_rabbithole_stores_documents` instead

### Memory:
- `before_cat_recalls_episodic_memories`
- `before_cat_recalls_declarative_memories`
- `before_cat_recalls_procedural_memories`

## Code changes
Since the Vector Database could be specific for each instance of the Grinning Cat, the syntax `cat.memory.vectors.vector_db`
is no more available. Use `cat.vector_memory_handler` instead. In case of missing methods in the Vector Database, you can
create your own Vector Handler extending `BaseVectorDatabaseHandler`, or you can extend the existing
Qdrant-based Vector Handler, `QdrantHandler`.
