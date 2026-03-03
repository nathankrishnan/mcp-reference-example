import json
from typing import Any, Literal
from client.mcp_client import MCPClient
from mcp.types import TextContent
from anthropic.types import Message, ToolResultBlockParam


class ToolManager:
    """
    A utility class for managing and executing tool requests across multiple MCP clients.
    """

    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict[str, Any]]:
        """
        Gets all tools from the provided clients.

        Args:
            clients (dict[str, MCPClient]): A dictionary mapping client names to MCPClient instances.

        Returns:
            list[dict[str, Any]]: A list of dictionaries representing the available tools, including their name, description, and input schema.
        """
        tools = []

        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tool_models
            ]

        return tools

    @classmethod
    async def build_tool_client_map(
        cls,
        clients: dict[str, MCPClient],
    ) -> dict[str, MCPClient]:
        """
        Builds a mapping from tool name to the client that provides it.

        Args:
            clients (dict[str, MCPClient]): A dictionary mapping client names to MCPClient instances.

        Returns:
            dict[str, MCPClient]: A mapping from tool name to the MCPClient that provides it.
        """
        tool_client_map: dict[str, MCPClient] = {}

        for client in clients.values():
            tools = await client.list_tools()
            for tool in tools:
                tool_client_map[tool.name] = client

        return tool_client_map

    @classmethod
    def _build_tool_result_part(
        cls,
        tool_use_id: str,
        text: str,
        status: Literal["success", "error"],
    ) -> ToolResultBlockParam:
        """
        Builds a tool result part dictionary.

        Args:
            tool_use_id (str): The ID of the tool use request.
            text (str): The content to include in the tool result.
            status (Literal["success", "error"]): The execution status of the tool ("success" or "error").

        Returns:
            ToolResultBlockParam: A dictionary representing the tool result block.
        """
        return {
            "tool_use_id": tool_use_id,
            "type": "tool_result",
            "content": text,
            "is_error": status == "error",
        }

    @classmethod
    async def execute_tool_requests(
        cls,
        tool_client_map: dict[str, MCPClient],
        message: Message,
    ) -> list[ToolResultBlockParam]:
        """
        Executes a list of tool requests against the provided clients.

        Args:
            tool_client_map (dict[str, MCPClient]): A mapping from tool name to the MCPClient that provides it.
            message (Message): The message containing the tool requests.

        Returns:
            list[ToolResultBlockParam]: A list of tool result blocks for each executed tool request.
        """
        tool_requests = [block for block in message.content if block.type == "tool_use"]
        tool_result_blocks: list[ToolResultBlockParam] = []

        for tool_request in tool_requests:
            tool_use_id = tool_request.id
            tool_name = tool_request.name
            tool_arguments = tool_request.input

            client = tool_client_map.get(tool_name)

            if not client:
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id, "Could not find that tool", "error"
                )
                tool_result_blocks.append(tool_result_part)
                continue

            try:
                tool_output = await client.call_tool(tool_name, args=tool_arguments)

                items = []
                if tool_output:
                    items = tool_output.content

                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                content_json = json.dumps(content_list)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    content_json,
                    "error" if tool_output and tool_output.isError else "success",
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    json.dumps({"error": error_message}),
                    "error",
                )

            tool_result_blocks.append(tool_result_part)

        return tool_result_blocks
