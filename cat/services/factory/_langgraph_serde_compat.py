"""Narrow compatibility shim for an import-time langgraph deprecation warning.

langgraph builds its checkpoint JSON reviver at module level without an explicit
``allowed_objects`` value::

    # langgraph/checkpoint/serde/jsonplus.py
    LC_REVIVER = Reviver()

langchain-core >= 1.3.3 therefore emits a ``LangChainPendingDeprecationWarning``
("The default value of `allowed_objects` will change in a future version") the
first time langgraph's serde is imported - which happens through
``langchain.agents``. The reviver is created inside the library: neither
``JsonPlusSerializer`` nor ``create_agent(checkpointer=...)`` accepts
``allowed_objects``, so there is no construction site we own to fix.

Current behaviour, which we keep relying on, is ``allowed_objects="core"``: when
the argument is ``None`` langchain-core assigns ``"core"`` (see
``langchain_core/load/load.py``, ``Reviver.__init__``).

Filter scope: exactly this message, exactly ``LangChainPendingDeprecationWarning``,
only the langgraph serde modules that emit it. No global warning suppression.
Upstream: langgraph ``checkpoint/serde/jsonplus.py`` (``LC_REVIVER``).
"""

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change in a future version",
    category=LangChainPendingDeprecationWarning,
    module=r"langgraph\.checkpoint\.serde\.(jsonplus|encrypted)",
)
