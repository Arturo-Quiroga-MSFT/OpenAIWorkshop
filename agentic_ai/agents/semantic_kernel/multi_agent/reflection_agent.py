import asyncio
import logging
import re
from semantic_kernel.agents import (
    ChatCompletionAgent,
    GroupChatOrchestration,
    RoundRobinGroupChatManager,
    ChatHistoryAgentThread,
)
from fastapi.encoders import jsonable_encoder

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel.contents import ChatMessageContent
from agents.base_agent import BaseAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Agent(BaseAgent):
    def __init__(self, state_store, session_id) -> None:
        super().__init__(state_store, session_id)

        # Keys scoped by session_id to isolate data per user/session
        self.thread_key = f"{session_id}_thread"
        self.customer_key = f"{session_id}_customer_id"
        self.chat_history_key = f"{session_id}_chat_history"

        # Restore state from persistent store
        import inspect
        self._thread = self.state_store.get(self.thread_key)
        if isinstance(self._thread, dict):
            valid_keys = inspect.signature(ChatHistoryAgentThread).parameters.keys()
            filtered = {k: v for k, v in self._thread.items() if k in valid_keys}
            self._thread = ChatHistoryAgentThread(**filtered)
        self._customer_id: str | None = self.state_store.get(self.customer_key)
        self._conversation_history: list[dict] = self.state_store.get(self.chat_history_key, [])

        self._agents = None
        self._mcp_plugin = None
        self._initialized = False
        self._orchestration: GroupChatOrchestration | None = None

    async def setup_agents(self) -> None:
        if self._initialized:
            return

        self._mcp_plugin = MCPStreamableHttpPlugin(
            name="ContosoMCP",
            description="Contoso MCP Plugin",
            url=self.mcp_server_uri,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        await self._mcp_plugin.connect()

        primary_agent = ChatCompletionAgent(
            service=AzureChatCompletion(deployment_name=self.azure_deployment),
            name="PrimaryAgent",
            description="You are a helpful assistant answering customer questions for internet provider Contosso.",
            instructions=(
                "You are a helpful assistant. Use the available MCP tools to find info and answer questions. "
                "You send your response to the secondary agent for review before replying to the user. "
                "Again **DON'T** assume ID, only pay close attention to previous chat context and secondary agent's response to see if user provided it. "
                "If unsure of the customer ID, **ALWAYS ASK NEVER ASSUME**. "
                "Respond to the user whatever was final answer from the secondary agent."
            ),
            plugins=[self._mcp_plugin],
        )

        secondary_agent = ChatCompletionAgent(
            service=AzureChatCompletion(deployment_name=self.azure_deployment),
            name="SecondaryAgent",
            description="You are a supervisor assistant who the primary agent reports to before answering user",
            instructions=(
                "Make sure you double check the primary agent's response for accuracy and completeness, you can provide improvement feedback if needed. "
                "If NOT, **YOU MUST ASK** user to provide it. "
                "After reviewing, suggest the primary response what to answer to the user finally and include details on specifics such as invoice, bill, refund, promotion offers, etc. "
                "If unsure of the customer ID, **ALWAYS ASK NEVER ASSUME**."
            ),
            plugins=[self._mcp_plugin],
        )

        self._agents = [primary_agent, secondary_agent]
        self._initialized = True

        if self._orchestration is None:
            def agent_response_callback(message: ChatMessageContent) -> None:
                logger.info(f"**{message.name}**\n{message.content}")

            self._orchestration = GroupChatOrchestration(
                members=self._agents,
                manager=RoundRobinGroupChatManager(max_rounds=3),
                agent_response_callback=agent_response_callback,
            )

    async def chat_async(self, user_input: str) -> str:
        await self.setup_agents()

        # Extract customer ID from input and persist
        match = re.search(r"customer\s*id[:\s]*([0-9]+)", user_input, re.IGNORECASE)
        if match:
            self._customer_id = match.group(1)
            self.state_store[self.customer_key] = self._customer_id

        # Prepend customer ID if known and not already present in input
        if self._customer_id and "customer id" not in user_input.lower():
            user_input = f"Customer ID: {self._customer_id}\n{user_input}"

        # Get response from primary agent, passing persistent thread
        response = await self._agents[0].get_response(messages=user_input, thread=self._thread)

        # Update thread and persist
        self._thread = response.thread
        if self._thread:
            self.state_store[self.thread_key] = jsonable_encoder(self._thread)

        response_content = str(response.content)

        # Update and persist conversation history for UI
        self._conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response_content},
        ])
        self.state_store[self.chat_history_key] = self._conversation_history

        return response_content
