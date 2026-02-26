import logging

from core.claude import Claude
from core.tools import ToolManager
from client.mcp_client import MCPClient
from anthropic.types import MessageParam

logger = logging.getLogger(__name__)


class Chat:
    """
    Manages a multi-turn conversation with Claude, including agentic tool-use loops.
    """

    def __init__(self, claude_service: Claude, clients: dict[str, MCPClient]):
        """
        Initialize the Chat instance.

        Args:
            claude_service (Claude): The Claude API wrapper used to send messages.
            clients (dict[str, MCPClient]): A dictionary mapping client names to MCPClient instances.
        """
        self.claude_service = claude_service
        self.clients = clients
        self.messages: list[MessageParam] = []

    async def run(
        self,
        query: str,
        max_iterations: int = 10,
    ) -> str:
        """
        Send a user query and run the agentic loop until a final text response is produced.

        Args:
            query (str): The user's input message.
            max_iterations (int, optional): Maximum number of agentic loop iterations. Defaults to 10.

        Returns:
            str: The final text response from Claude.

        Raises:
            RuntimeError: If the loop exceeds max_iterations without producing a final response.
        """
        self.messages.append({"role": "user", "content": query})

        tools = await ToolManager.get_all_tools(self.clients)
        tool_client_map = await ToolManager.build_tool_client_map(self.clients)

        for _ in range(max_iterations):
            response = self.claude_service.chat(
                messages=self.messages,
                tools=tools,
            )

            self.claude_service.add_assistant_message(self.messages, response)

            if response.stop_reason != "tool_use":
                return self.claude_service.text_from_message(response)

            logger.info(self.claude_service.text_from_message(response))

            tool_result_parts = await ToolManager.execute_tool_requests(
                tool_client_map, response
            )

            self.claude_service.add_user_message(self.messages, tool_result_parts)

        raise RuntimeError(
            f"Chat exceeded maximum iterations ({max_iterations})"
        )
