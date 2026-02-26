import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from core.tools import ToolManager
from mcp.types import TextContent, CallToolResult


@pytest.fixture
def mock_client_1():
    """Create a mock MCPClient with a single tool."""
    client = MagicMock()
    tool1 = MagicMock()
    tool1.name = "tool_1"
    tool1.description = "First tool"
    tool1.inputSchema = {"type": "object"}
    client.list_tools = AsyncMock(return_value=[tool1])
    return client


@pytest.fixture
def mock_client_2():
    """Create a secondary mock MCPClient with a different tool."""
    client = MagicMock()
    tool2 = MagicMock()
    tool2.name = "tool_2"
    tool2.description = "Second tool"
    tool2.inputSchema = {"type": "object"}
    client.list_tools = AsyncMock(return_value=[tool2])
    return client


async def test_get_all_tools(mock_client_1, mock_client_2):
    """Test retrieving and formatting all tools from multiple clients."""
    clients = {"c1": mock_client_1, "c2": mock_client_2}
    tools = await ToolManager.get_all_tools(clients)

    assert len(tools) == 2
    assert tools[0]["name"] == "tool_1"
    assert tools[1]["name"] == "tool_2"


async def test_find_client_with_tool(mock_client_1, mock_client_2):
    """Test finding a specific client that provides a given tool."""
    clients = [mock_client_1, mock_client_2]

    found_client = await ToolManager._find_client_with_tool(clients, "tool_2")
    assert found_client == mock_client_2

    not_found = await ToolManager._find_client_with_tool(clients, "tool_3")
    assert not_found is None


def test_build_tool_result_part():
    """Test that a tool result part block is formatted correctly."""
    res = ToolManager._build_tool_result_part("id_123", "some text", "success")
    assert res == {
        "tool_use_id": "id_123",
        "type": "tool_result",
        "content": "some text",
        "is_error": False,
    }

    err_res = ToolManager._build_tool_result_part("id_456", "error text", "error")
    assert err_res["is_error"] is True


async def test_execute_tool_requests_not_found(mock_client_1):
    """Test executing a tool request when the tool is not found on any client."""
    # Mock message
    message = MagicMock()
    tool_req = MagicMock()
    tool_req.type = "tool_use"
    tool_req.id = "req_1"
    tool_req.name = "unknown_tool"
    tool_req.input = {}
    message.content = [tool_req]

    clients = {"c1": mock_client_1}

    results = await ToolManager.execute_tool_requests(clients, message)

    assert len(results) == 1
    assert results[0]["is_error"] is True
    assert results[0]["content"] == "Could not find that tool"


async def test_execute_tool_requests_success(mock_client_1):
    """Test successful tool execution mapping back to the proper format."""
    # Setup mock message and request
    message = MagicMock()
    tool_req = MagicMock()
    tool_req.type = "tool_use"
    tool_req.id = "req_1"
    tool_req.name = "tool_1"
    tool_req.input = {"param": "value"}
    message.content = [tool_req]

    clients = {"c1": mock_client_1}

    # Setup the client's tool call return
    call_result = MagicMock(spec=CallToolResult)
    call_result.isError = False

    # TextContent is required by the `isinstance` check in the implementation
    mock_content = MagicMock(spec=TextContent)
    mock_content.text = "Success output"
    call_result.content = [mock_content]

    mock_client_1.call_tool = AsyncMock(return_value=call_result)

    results = await ToolManager.execute_tool_requests(clients, message)

    assert len(results) == 1
    assert results[0]["is_error"] is False
    assert results[0]["content"] == '["Success output"]'


async def test_execute_tool_requests_exception(mock_client_1):
    """Test executing a tool query that throws an unexpected connection exception."""
    # Setup mock message for tool request
    message = MagicMock()
    tool_req = MagicMock()
    tool_req.type = "tool_use"
    tool_req.id = "req_1"
    tool_req.name = "tool_1"
    tool_req.input = {}
    message.content = [tool_req]

    clients = {"c1": mock_client_1}

    # Force call_tool to raise an Exception
    mock_client_1.call_tool = AsyncMock(side_effect=Exception("Network error"))

    results = await ToolManager.execute_tool_requests(clients, message)

    assert len(results) == 1
    assert results[0]["is_error"] is True

    # Validate that we successfully caught the error and encoded it as JSON
    content_dict = json.loads(results[0]["content"])
    assert "error" in content_dict
    assert "Network error" in content_dict["error"]
