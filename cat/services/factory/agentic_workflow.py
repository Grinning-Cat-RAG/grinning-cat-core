import re
from abc import ABC, abstractmethod
from typing import Type, List, Dict, Tuple
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field

from cat.log import log
from cat.looking_glass.models import AgenticWorkflowOutput, AgenticWorkflowTask
from cat.services.factory.models import BaseFactoryConfigModel


class BaseAgenticWorkflowHandler(ABC):
    """
    Base class to build a custom Agentic Workflow.
    MUST be implemented by subclasses.
    """
    def __init__(self, metadata: Dict | None = None):
        self._metadata = metadata or {}

        self._task: AgenticWorkflowTask | None = None
        self._llm: BaseLanguageModel | None = None
        self._callbacks: List[BaseCallbackHandler] | None = None
        self._can_bind_tools = False

    @property
    def metadata(self) -> Dict:
        return self._metadata

    @staticmethod
    def _clean_response(response: str) -> str:
        # parse the `response` string and get the text from <answer></answer> tag
        if "<answer>" in response and "</answer>" in response:
            start = response.index("<answer>") + len("<answer>")
            end = response.index("</answer>")
            return response[start:end].strip()

        # otherwise, remove from `response` all the text between whichever tag and then return the remaining string
        # This pattern matches any complete tag pair: <tagname>content</tagname>
        cleaned = re.sub(r'<([^>]+)>.*?</\1>', '', response, flags=re.DOTALL)
        return cleaned.strip()

    @staticmethod
    def _extract_steps(messages: List) -> List[Tuple[Tuple[str | None, Dict, Dict] | None, str | None]]:
        """Rebuild ``intermediate_steps`` from the agent message history.

        Every AIMessage with ``tool_calls`` produces one step
        ``((tool_name, tool_input, usage_metadata), tool_output)``; the matching
        ToolMessage is looked up by its ``tool_call_id``.
        """
        tool_results = {
            msg.tool_call_id: str(msg.content)
            for msg in messages
            if isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", None)
        }
        tooled_messages = [msg for msg in messages if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)]

        return [
            (
                (tc.get("name"), tc.get("args", {}), getattr(msg, "usage_metadata", {}) or {}),
                tool_results.get(tc.get("id"))
            )
            for msg in tooled_messages
            for tc in msg.tool_calls
        ]

    async def run(
        self, task: AgenticWorkflowTask, llm: BaseLanguageModel, callbacks: List[BaseCallbackHandler] = None  # type: ignore[assignment]
    ) -> AgenticWorkflowOutput:
        """
        Executes the agent with the given prompt and procedures. It processes the LLM output, handles tool
        calls, and generates the final response. It also cleans the response from any tags.

        Args:
            task (AgenticWorkflowTask): The input containing the task details.
            llm (BaseLanguageModel): The language model to use for generating responses.
            callbacks (List[BaseCallbackHandler], optional): List of callback handlers for logging and monitoring, by default None.

        Returns:
            AgentOutput
                The final output from the agent, including text and any actions taken.
        """
        self._task = task
        self._llm = llm
        self._callbacks = callbacks or []

        prompt = ChatPromptTemplate.from_messages([
            *([SystemMessagePromptTemplate.from_template(template=task.system_prompt)] if task.system_prompt else []),
            *self._task.history,  # type: ignore[misc, union-attr]
            HumanMessagePromptTemplate.from_template(template=task.user_prompt),
        ])

        # Attach recalled multimodal images (full image_url content parts) as a
        # concrete HumanMessage so both the no-tool and tool-binding paths carry them.
        if task.images:
            prompt = ChatPromptTemplate.from_messages(
                prompt.messages + [HumanMessage(content=task.images)]
            )

        # Intrinsic detection of tool binding support
        self._can_bind_tools = task.tools and hasattr(llm, "bind_tools")  # type: ignore[assignment]
        if not self._can_bind_tools:
            res = await self._run_no_tool_binding(prompt)
            return res

        try:
            res = await self._run_tool_binding(prompt)
            return res
        except Exception as e:
            log.warning(f"Tool binding failed with error: {e}. Falling back to direct LLM invocation.")

            res = await self._run_no_tool_binding(prompt)
            return res

    @abstractmethod
    async def _run_no_tool_binding(self, prompt: ChatPromptTemplate) -> AgenticWorkflowOutput:
        """
        Executes the agentic workflow without using tool binding. This method is used when tools are not enabled
        or when the language model does not support tool binding.

        Args:
            prompt (ChatPromptTemplate): The prompt template to use for the agent.

        Returns:
            AgenticWorkflowOutput: The output of the agentic workflow execution.
        """
        pass

    @abstractmethod
    async def _run_tool_binding(self, prompt: ChatPromptTemplate) -> AgenticWorkflowOutput:
        """
        Executes the agentic workflow using tool binding. This method is used when tools are enabled and the
        language model supports tool binding.

        Args:
            prompt (ChatPromptTemplate): The prompt template to use for the agent.

        Returns:
            AgenticWorkflowOutput: The output of the agentic workflow execution.
        """
        pass


class CoreAgenticWorkflow(BaseAgenticWorkflowHandler):
    async def _run_no_tool_binding(self, prompt: ChatPromptTemplate) -> AgenticWorkflowOutput:
        prompt_variables = self._task.prompt_variables or {}  # type: ignore[union-attr]

        chain = prompt | self._llm
        langchain_msg = await chain.ainvoke(prompt_variables, config=RunnableConfig(callbacks=self._callbacks))

        output = getattr(langchain_msg, "content", str(langchain_msg))
        return AgenticWorkflowOutput(output=self._clean_response(output))

    async def _run_tool_binding(self, prompt: ChatPromptTemplate) -> AgenticWorkflowOutput:
        task = self._task
        variables = task.prompt_variables or {}  # type: ignore[union-attr]

        # Format the system prompt with the task variables (e.g. the recalled context)
        system_prompt = None
        if task.system_prompt:
            system_prompt = task.system_prompt
            try:
                system_prompt = task.system_prompt.format(**variables)
            except (KeyError, AttributeError):
                pass  # keep the raw template if it cannot be formatted

        # Build the message history exactly like the legacy agent worked:
        # history first, then the optional multimodal images, then the user prompt
        messages = list(task.history or [])  # type: ignore[union-attr]

        if task.images:
            messages.append(HumanMessage(content=task.images))

        user_prompt = task.user_prompt
        try:
            user_prompt = task.user_prompt.format(**variables)
        except (KeyError, AttributeError):
            pass  # keep the raw template if it cannot be formatted
        messages.append(HumanMessage(content=user_prompt))

        # Create the agent (langchain v1 `create_agent`, replaces the legacy
        # `langchain_classic.create_tool_calling_agent` + `AgentExecutor`)
        agent = create_agent(
            model=self._llm,  # type: ignore[arg-type]
            tools=task.tools or [],
            system_prompt=system_prompt,
        )
        # Run the agent
        langchain_msg = await agent.ainvoke(
            {"messages": messages}, config=RunnableConfig(callbacks=self._callbacks)
        )

        # Extract the final answer: the last AIMessage content
        final_messages = langchain_msg.get("messages", [])
        final_content = ""
        for msg in reversed(final_messages):
            if isinstance(msg, HumanMessage):
                continue
            final_content = getattr(msg, "content", str(msg))
            break

        cleaned_output = self._clean_response(final_content).strip()
        extracted_steps = self._extract_steps(final_messages)
        return AgenticWorkflowOutput(output=cleaned_output, intermediate_steps=extracted_steps)


class AgenticWorkflowConfig(BaseFactoryConfigModel, ABC):
    metadata: Dict | None = Field(default={}, description="Metadata associated with the workflow.")

    @classmethod
    def base_class(cls) -> Type[BaseAgenticWorkflowHandler]:
        return BaseAgenticWorkflowHandler

    @classmethod
    @abstractmethod
    def pyclass(cls) -> Type[BaseAgenticWorkflowHandler]:
        pass


class CoreAgenticWorkflowConfig(AgenticWorkflowConfig):
    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Single-agent Workflow",
            "description": "Core built-in single-agent workflow.",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CoreAgenticWorkflow]:
        return CoreAgenticWorkflow
