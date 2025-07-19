import asyncio
import logging
from typing import List, Optional

from base_agent import BaseAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt

from semantic_kernel import Kernel
import asyncio

from semantic_kernel.agents import Agent, ChatCompletionAgent, HandoffOrchestration, OrchestrationHandoffs
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import AuthorRole, ChatMessageContent, FunctionCallContent, FunctionResultContent
from semantic_kernel.functions import KernelPlugin

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# This class implements a multi-domain collaborative agent that orchestrates tasks among specialists.
class Agent(BaseAgent):
    """
    Multi‑domain, SK‑based collaborative agent.

    Participants
    ------------
    • analysis_planning        – orchestrator, produces FINAL ANSWER
    • crm_billing              – billing specialist
    • product_promotions       – promotions / offers specialist
    • security_authentication  – security specialist
    """

    def __new__(cls, state_store: dict, session_id: str):
        # Return the existing instance if it exists in the session store.
        if session_id in state_store:
            return state_store[session_id]
        instance = super().__new__(cls)
        state_store[session_id] = instance
        return instance

    def __init__(self, state_store: dict, session_id: str) -> None:
        # Prevent re‑initialization if the instance was already constructed.
        if hasattr(self, "_constructed"):
            return
        self._constructed = True
        super().__init__(state_store, session_id)

    # Helper to create a plugin instance.
    async def _create_plugin(self, name) -> MCPSsePlugin:
        plugin = MCPSsePlugin(
            name=name,
            description=f"{name} Contoso MCP Plugin",
            url=self.mcp_server_uri,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        await plugin.connect()
        return plugin

    # Helper to create a kernel instance.
    def _create_kernel(self) -> Kernel:
        kernel = Kernel()
        kernel.add_service(
            service=AzureChatCompletion(
                api_key=self.azure_openai_key,
                endpoint=self.azure_openai_endpoint,
                api_version=self.api_version,
                deployment_name=self.azure_deployment,
            )
        )
        return kernel
    
    # Helper to create an agent instance.
    async def _create_agent(self, name: str, instructions: str, included_tools: Optional[List[str]] = []) -> ChatCompletionAgent:
        kernel = self._create_kernel()
        settings = kernel.get_prompt_execution_settings_from_service_id("default")
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
            filters={"included_functions": included_tools}
        )
        plugin = await self._create_plugin(name)
        agent = ChatCompletionAgent(
            name=name,
            instructions=instructions,
            kernel=kernel,
            arguments=KernelArguments(settings),
            plugins=[plugin],
        )
        return agent

    # Setup a team of participants for multi-agent collaboration.
    async def _setup_team(self) -> None:
        if getattr(self, "_initialized", False):
            return

        analysis_planning_agent = await self._create_agent(
            name="analysis_planning",
            instructions="You are the Analysis & Planning Agent (the planner/orchestrator).\n"
            "\n"
            "1. Decide if the user’s request can be satisfied directly:\n"
            "   - If YES (e.g. greetings, very simple Q&A), answer immediately using the prefix:\n"
            "     FINAL ANSWER: <your reply>\n"
            "\n"
            "2. Otherwise you MUST delegate atomic sub‑tasks one‑by‑one to specialists.\n"
            "   - Output format WHEN DELEGATING (strict):\n"
            "       <specialist_name>: <task>\n"
            "     – No other text, no quotation marks, no ‘FINAL ANSWER’.\n"
            "   - Delegate only one sub‑task per turn, then wait for the specialist’s reply.\n"
            "\n"
            "3. After all required information is gathered, compose ONE comprehensive response and\n"
            "   send it to the user prefixed with:\n"
            "   FINAL ANSWER: <your synthesized reply>\n"
            "\n"
            "4. If you need clarification from the user, ask it immediately and prefix with\n"
            "   FINAL ANSWER: <your question>\n"
            "\n"
            "Specialist directory – choose the SINGLE best match for each sub‑task:\n"
            "- crm_billing – Accesses CRM & billing systems for account, subscription, invoice,\n"
            "  payment status, refunds and policy compliance questions.\n"
            "- product_promotions – Provides product catalogue details, current promotions,\n"
            "  discount eligibility rules and T&Cs from structured sources & FAQs.\n"
            "- security_authentication – Investigates authentication logs, account lock‑outs,\n"
            "  security incidents; references security KBs and recommends remediation steps.\n"
            "\n"
            "STRICT RULES:\n"
            "- Do not emit planning commentary or bullet lists to the user.\n"
            "- Only ‘FINAL ANSWER’ messages or specialist delegations are allowed.\n"
            "- Never include ‘FINAL ANSWER’ when talking to a specialist.\n",
            included_tools=[],
        )

        crm_billing_agent = await self._create_agent(
            name="crm_billing",
            instructions="You are the CRM & Billing Agent.\n"
            "- Query structured CRM / billing systems for account, subscription, "
            "invoice, and payment information as needed.\n"
            "- For each response you **MUST** cross‑reference relevant *Knowledge Base* articles on billing policies, payment "
            "processing, refund rules, etc., to ensure responses are accurate "
            "and policy‑compliant.\n"
            "- Reply with concise, structured information and flag any policy "
            "concerns you detect.\n"
            "Only respond with data you retrieve using your tools.\n"
            "DO NOT respond to anything out of your domain.",
            included_tools=[
                "ContosoMCP-get_all_customers",
                "ContosoMCP-get_customer_detail",
                "ContosoMCP-get_subscription_detail",
                "ContosoMCP-get_invoice_payments",
                "ContosoMCP-pay_invoice",
                "ContosoMCP-get_data_usage",
                "ContosoMCP-search_knowledge_base",
                "ContosoMCP-get_customer_orders",
                "ContosoMCP-update_subscription",
                "ContosoMCP-get_billing_summary",
            ],
        )

        product_promotions_agent = await self._create_agent(
            name="product_promotions",
            instructions="You are the Product & Promotions Agent.\n"
            "- Retrieve promotional offers, product availability, eligibility "
            "criteria, and discount information from structured sources.\n"
            "- For each response you **MUST** cross‑reference relevant *Knowledge Base* FAQs, terms & conditions, "
            "and best practices.\n"
            "- Provide factual, up‑to‑date product/promo details."
            "Only respond with data you retrieve using your tools.\n"
            "DO NOT respond to anything out of your domain.",
            included_tools=[
                "ContosoMCP-get_all_customers",
                "ContosoMCP-get_customer_detail",
                "ContosoMCP-get_promotions",
                "ContosoMCP-get_eligible_promotions",
                "ContosoMCP-search_knowledge_base",
                "ContosoMCP-get_products",
                "ContosoMCP-get_product_detail",
            ],
        )

        security_authentication_agent = await self._create_agent(
            name="security_authentication",
            instructions="You are the Security & Authentication Agent.\n"
            "- Investigate authentication logs, account lockouts, and security "
            "incidents in structured security databases.\n"
            "- For each response you **MUST** cross‑reference relevant *Knowledge Base* security policies and "
            "lockout troubleshooting guides.\n"
            "- Return clear risk assessments and recommended remediation steps."
            "Only respond with data you retrieve using your tools.\n"
            "DO NOT respond to anything out of your domain.",
            included_tools=[
                "ContosoMCP-get_all_customers",
                "ContosoMCP-get_customer_detail",
                "ContosoMCP-get_security_logs",
                "ContosoMCP-search_knowledge_base",
                "ContosoMCP-unlock_account",
            ],
        )

        handoffs = (
            OrchestrationHandoffs()
            .add_many(
                source_agent=analysis_planning_agent.name,
                target_agents={
                    crm_billing_agent.name: "Transfer to this agent if the issue is related to CRM & billing systems for account, subscription, invoice,payment status, refunds and policy compliance questions",
                    product_promotions_agent.name: "Transfer to this agent if the issue is related to product catalogue details, current promotions, discount eligibility rules and T&Cs from structured sources & FAQs",
                    security_authentication_agent.name: "Transfer to this agent if the issue is about investigating authentication logs, account lock‑outs,security incidents; references security KBs and recommends remediation steps",
                },
            )
            .add(
                source_agent=crm_billing_agent.name,
                target_agent=analysis_planning_agent.name,
                description="Transfer to this agent if the issue is not related to either of the these - CRM & billing systems for account, subscription, invoice,payment status, refunds and policy compliance questions",
            )
            .add(
                source_agent=product_promotions_agent.name,
                target_agent=analysis_planning_agent.name,
                description="Transfer to this agent if the issue is not related to either of these- product catalogue details, current promotions, discount eligibility rules and T&Cs from structured sources & FAQs",
            )
            .add(
                source_agent=security_authentication_agent.name,
                target_agent=analysis_planning_agent.name,
                description="Transfer to this agent if the issue is not related to either of these- investigating authentication logs, account lock‑outs, security incidents; references security KBs and recommends remediation steps",
            )
        )

        def agent_response_callback(message: ChatMessageContent):
            logger.info(f"{message.name}: {message.content}")
            for item in message.items:
                if isinstance(item, FunctionCallContent):
                    logger.info(f"Calling '{item.name}' with args: {item.arguments}")
                if isinstance(item, FunctionResultContent):
                    logger.info(f"Result from '{item.name}': {item.result}")

        def human_response_function() -> ChatMessageContent:
            user_input = input("User: ")
            return ChatMessageContent(role=AuthorRole.USER, content=user_input)

        self._orchestration = HandoffOrchestration(
            members=[
                analysis_planning_agent,
                crm_billing_agent,
                product_promotions_agent,
                security_authentication_agent,
            ],
            handoffs=handoffs,
            agent_response_callback=agent_response_callback,
            human_response_function=human_response_function,
        )

        self._initialized = True

    # ------------------------------------------------------------------ #
    #                              CHAT API                               #
    # ------------------------------------------------------------------ #
    async def chat_async(self, user_input: str) -> str:
        await self._setup_team()

        runtime = InProcessRuntime()
        runtime.start()

        final_result = ""
        try:
            orchestration_result = await self._orchestration.invoke(task=user_input, runtime=runtime)
            final_result = await orchestration_result.get()
        except Exception as e:
            final_result = f"Error during orchestration: {e}"
        finally:
            await runtime.stop_when_idle()

        self.append_to_chat_history([
            {"role": "user", "content": str (user_input)},
            {"role": "assistant", "content": str (final_result)},
        ])

        return str(final_result)

# --------------------------- Manual test helper --------------------------- #
async def _demo():
    dummy_state = {}
    agent = Agent(dummy_state, session_id="demo")
    while True:
        question = input(">>> ")                    # My customer id is 101, why is my internet bill so high?
        if question.lower() in {"exit", "quit"}:
            break
        answer = await agent.chat_async(question)
        print("\n>>> Assistant reply:\n", answer)

if __name__ == "__main__":
    asyncio.run(_demo())
