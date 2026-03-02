import json
import asyncio
from typing import Optional, Any
from contextlib import AsyncExitStack
from pydantic import AnyUrl
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    A client for connecting to and interacting with an MCP server.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: Optional[dict] = None,
    ):
        """
        Initialize the MCP client with the server's launch configuration.

        Args:
            command (str): The executable used to start the MCP server (e.g. 'uv').
            args (list[str]): Arguments passed to the command (e.g. ['run', 'mcp_server.py']).
            env (Optional[dict]): Optional environment variables to pass to the server process.
        """
        self._command = command
        self._args = args
        self._env = env
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

    async def connect(self):
        """
        Launch the MCP server process and establish a session with it.

        Opens stdio pipes to the server and initializes a ClientSession,
        registering both with the exit stack so they are cleaned up automatically.
        """
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )

        server_pipes = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = server_pipes

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        """
        Return the active ClientSession.

        Returns:
            ClientSession: The active session connected to the MCP server.

        Raises:
            ConnectionError: If connect() has not been called yet.
        """
        if self._session is None:
            raise ConnectionError("No active session. Call connect() first.")

        return self._session

    async def cleanup(self):
        """
        Close all open connections and reset the session.

        Calls aclose() on the exit stack, which tears down the ClientSession
        and stdio pipes in reverse order of how they were opened.
        """
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        """
        Support usage as an async context manager.
        Automatically calls connect() when entering the `async with` block.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Support usage as an async context manager.
        Automatically calls cleanup() when exiting the `async with` block.
        """
        await self.cleanup()

    async def list_tools(self) -> list[types.Tool]:
        """
        Retrieve the list of tools available on the connected MCP server.

        Returns:
            list[types.Tool]: A list of Tool objects defined by the server.
        """
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(
        self, tool_name: str, arguments: dict
    ) -> types.CallToolResult | None:
        """
        Invoke a specific tool on the MCP server with the given arguments.

        Args:
            tool_name (str): The name of the tool to call.
            arguments (dict): A dictionary of arguments to pass to the tool.

        Returns:
            types.CallToolResult | None: The result returned by the tool, or None.
        """
        return await self.session().call_tool(tool_name, arguments)

    async def list_prompts(self) -> list[types.Prompt]:
        """
        Retrieve the list of prompts available on the connected MCP server.

        Returns:
            list[types.Prompt]: A list of Prompt objects defined by the server.
        """
        # TODO: Return a list of prompts defined by the MCP server
        return []

    async def get_prompt(self, prompt_name: str, args: dict[str, str]):
        """
        Fetch a specific prompt from the MCP server, rendered with the given arguments.

        Args:
            prompt_name (str): The name of the prompt to retrieve.
            args (dict[str, str]): Key-value pairs used to fill in the prompt's variables.
        """
        # TODO: Get a particular prompt defined by the MCP server
        return []

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource exposed by the MCP server at the given URI.

        Args:
            uri (str): The URI identifier of the resource to read.

        Returns:
            Any: The parsed contents of the resource.
        """
        result = await self.session().read_resource(AnyUrl(uri))
        resource = result.contents[0]

        if isinstance(resource, types.TextResourceContents):
            if resource.mimeType == "application/json":
                return json.loads(resource.text)

            return resource.text


# For testing
async def main():
    async with MCPClient(
        command="uv",
        args=["run", "server/mcp_server.py"],
    ) as _client:
        result = await _client.list_tools()
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
