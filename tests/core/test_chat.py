import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from core.chat import Chat


DESIRED_MODEL = "claude-sonnet-4-6"


def make_message(stop_reason="end_turn", content=None):
    """Helper to create a fake Anthropic Message."""
    if content is None:
        content = [TextBlock(type="text", text="Final answer")]

    return Message(
        id="msg_123",
        model=DESIRED_MODEL,
        role="assistant",
        type="message",
        stop_reason=stop_reason,
        usage=Usage(input_tokens=10, output_tokens=10),
        content=content,
    )


@pytest.fixture
def mock_claude():
    """Create a mock Claude service."""
    claude = MagicMock()
    claude.text_from_message.return_value = "Final answer"
    return claude


@pytest.fixture
def mock_clients():
    """Create a mock MCP clients dict with one client."""
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=[])
    return {"c1": client}


async def test_run_appends_user_message(mock_claude, mock_clients):
    """Test that run() appends the user query to message history."""
    mock_claude.chat.return_value = make_message()

    chat = Chat(claude_service=mock_claude, clients=mock_clients)
    await chat.run("Hello")

    assert len(chat.messages) >= 1
    assert chat.messages[0] == {"role": "user", "content": "Hello"}


async def test_run_returns_final_text(mock_claude, mock_clients):
    """Test that run() returns the text from a non-tool-use response."""
    mock_claude.chat.return_value = make_message()
    mock_claude.text_from_message.return_value = "The answer is 42"

    chat = Chat(claude_service=mock_claude, clients=mock_clients)
    result = await chat.run("What is the answer?")

    assert result == "The answer is 42"


async def test_run_handles_tool_use_loop(mock_claude, mock_clients):
    """Test that run() processes tool use and loops back for a final response."""
    tool_use_response = make_message(
        stop_reason="tool_use",
        content=[
            TextBlock(type="text", text="Let me use a tool"),
            ToolUseBlock(type="tool_use", id="tu_1", name="tool_1", input={}),
        ],
    )
    final_response = make_message()

    mock_claude.chat.side_effect = [tool_use_response, final_response]
    mock_claude.text_from_message.return_value = "Final answer"

    with patch(
        "core.chat.ToolManager.execute_tool_requests", new_callable=AsyncMock
    ) as mock_execute:
        mock_execute.return_value = [
            {"tool_use_id": "tu_1", "type": "tool_result", "content": "result"}
        ]

        chat = Chat(claude_service=mock_claude, clients=mock_clients)
        result = await chat.run("Use a tool")

    assert result == "Final answer"
    assert mock_claude.chat.call_count == 2
    mock_execute.assert_called_once()


async def test_run_raises_on_max_iterations(mock_claude, mock_clients):
    """Test that run() raises RuntimeError when max_iterations is exceeded."""
    tool_use_response = make_message(
        stop_reason="tool_use",
        content=[
            ToolUseBlock(type="tool_use", id="tu_1", name="tool_1", input={}),
        ],
    )

    mock_claude.chat.return_value = tool_use_response
    mock_claude.text_from_message.return_value = ""

    with patch(
        "core.chat.ToolManager.execute_tool_requests", new_callable=AsyncMock
    ) as mock_execute:
        mock_execute.return_value = [
            {"tool_use_id": "tu_1", "type": "tool_result", "content": "result"}
        ]

        chat = Chat(claude_service=mock_claude, clients=mock_clients)

        with pytest.raises(RuntimeError, match="maximum iterations"):
            await chat.run("Loop forever", max_iterations=3)

    assert mock_claude.chat.call_count == 3
