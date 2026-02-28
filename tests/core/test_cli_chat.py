import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp.types import PromptMessage, TextContent
from core.cli_chat import (
    CliChat,
    _extract_text_from_content,
    _extract_text_blocks_from_list,
    convert_prompt_message_to_message_param,
    convert_prompt_messages_to_message_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_doc_client():
    """Create a mock MCP client for the document server."""
    client = MagicMock()
    client.list_prompts = AsyncMock(return_value=[])
    client.read_resource = AsyncMock()
    client.get_prompt = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_claude():
    """Create a mock Claude service."""
    return MagicMock()


@pytest.fixture
def mock_clients():
    """Create a mock MCP clients dict."""
    return {"c1": MagicMock()}


@pytest.fixture
def cli_chat(mock_doc_client, mock_clients, mock_claude):
    """Create a CliChat instance with mocked dependencies."""
    return CliChat(
        doc_client=mock_doc_client,
        clients=mock_clients,
        claude_service=mock_claude,
    )


# ---------------------------------------------------------------------------
# CliChat class tests
# ---------------------------------------------------------------------------


def test_init_stores_doc_client(mock_doc_client, mock_clients, mock_claude):
    """Test that __init__ stores doc_client and sets inherited attributes."""
    chat = CliChat(
        doc_client=mock_doc_client,
        clients=mock_clients,
        claude_service=mock_claude,
    )

    assert chat.doc_client is mock_doc_client
    assert chat.clients is mock_clients
    assert chat.claude_service is mock_claude
    assert chat.messages == []


async def test_extract_resources_finds_mentions(cli_chat, mock_doc_client):
    """Test that _extract_resources fetches content for @mentioned documents."""

    async def fake_read_resource(uri):
        if uri == "docs://documents":
            return ["report.docx", "notes.txt"]
        if uri == "docs://documents/report.docx":
            return "Report content here"
        return ""

    mock_doc_client.read_resource = AsyncMock(side_effect=fake_read_resource)

    result = await cli_chat._extract_resources("tell me about @report.docx")

    assert "report.docx" in result
    assert "Report content here" in result
    assert "notes.txt" not in result


async def test_extract_resources_no_mentions(cli_chat, mock_doc_client):
    """Test that _extract_resources returns empty string when no @mentions are found."""
    mock_doc_client.read_resource = AsyncMock(
        return_value=["report.docx", "notes.txt"]
    )

    result = await cli_chat._extract_resources("hello")

    assert result == ""


async def test_process_command_slash(cli_chat, mock_doc_client):
    """Test that _process_command handles slash commands and appends converted messages."""
    mock_prompt_msg = MagicMock(spec=PromptMessage)
    mock_prompt_msg.role = "user"
    mock_prompt_msg.content = TextContent(type="text", text="Summarize this doc")

    mock_doc_client.get_prompt = AsyncMock(return_value=[mock_prompt_msg])

    result = await cli_chat._process_command("/summarize report.docx")

    assert result is True
    mock_doc_client.get_prompt.assert_called_once_with(
        "summarize", {"doc_id": "report.docx"}
    )
    assert len(cli_chat.messages) == 1


async def test_process_command_not_slash(cli_chat, mock_doc_client):
    """Test that _process_command returns False for non-slash queries."""
    result = await cli_chat._process_command("hello")

    assert result is False
    mock_doc_client.get_prompt.assert_not_called()


async def test_process_query_slash_command(cli_chat, mock_doc_client, mock_claude):
    """Test that _process_query delegates to _process_command for slash input."""
    mock_doc_client.get_prompt = AsyncMock(return_value=[])

    await cli_chat._process_query("/summarize report.docx")

    mock_claude.add_user_message.assert_not_called()


async def test_process_query_normal(cli_chat, mock_doc_client, mock_claude):
    """Test that _process_query enriches normal queries with resources and adds user message."""
    mock_doc_client.read_resource = AsyncMock(return_value=[])

    await cli_chat._process_query("hello")

    mock_claude.add_user_message.assert_called_once()
    prompt = mock_claude.add_user_message.call_args[0][1]
    assert "hello" in prompt


# ---------------------------------------------------------------------------
# Converter function tests
# ---------------------------------------------------------------------------


def test_extract_text_from_content_dict():
    """Test extracting text from a dict content item."""
    result = _extract_text_from_content({"type": "text", "text": "hello"})
    assert result == "hello"


def test_extract_text_from_content_object():
    """Test extracting text from an object content item."""
    content = MagicMock()
    content.type = "text"
    content.text = "hello"

    result = _extract_text_from_content(content)
    assert result == "hello"


def test_extract_text_from_content_non_text():
    """Test that non-text type returns None."""
    result = _extract_text_from_content({"type": "image"})
    assert result is None


def test_extract_text_from_content_not_dict_or_object():
    """Test that a plain string returns None."""
    result = _extract_text_from_content("just a string")
    assert result is None


def test_extract_text_blocks_from_list():
    """Test extracting text blocks from a list of content items."""
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]

    result = _extract_text_blocks_from_list(content)

    assert result == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]


def test_extract_text_blocks_from_list_empty():
    """Test that an empty list returns None."""
    result = _extract_text_blocks_from_list([])
    assert result is None


def test_extract_text_blocks_from_list_not_list():
    """Test that non-list input returns None."""
    result = _extract_text_blocks_from_list("not a list")
    assert result is None


def test_convert_prompt_message_single_text():
    """Test converting a PromptMessage with single TextContent."""
    msg = MagicMock(spec=PromptMessage)
    msg.role = "user"
    msg.content = TextContent(type="text", text="hello")

    result = convert_prompt_message_to_message_param(msg)

    assert result == {"role": "user", "content": "hello"}


def test_convert_prompt_message_list_content():
    """Test converting a PromptMessage with a list of TextContent items."""
    msg = MagicMock(spec=PromptMessage)
    msg.role = "assistant"
    msg.content = [
        TextContent(type="text", text="first"),
        TextContent(type="text", text="second"),
    ]

    result = convert_prompt_message_to_message_param(msg)

    assert result == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ],
    }


def test_convert_prompt_message_fallback():
    """Test that unrecognized content falls back to empty string."""
    msg = MagicMock(spec=PromptMessage)
    msg.role = "user"
    msg.content = 12345

    result = convert_prompt_message_to_message_param(msg)

    assert result == {"role": "user", "content": ""}


def test_convert_prompt_messages_batch():
    """Test batch conversion of multiple PromptMessages."""
    msg1 = MagicMock(spec=PromptMessage)
    msg1.role = "user"
    msg1.content = TextContent(type="text", text="first")

    msg2 = MagicMock(spec=PromptMessage)
    msg2.role = "assistant"
    msg2.content = TextContent(type="text", text="second")

    results = convert_prompt_messages_to_message_params([msg1, msg2])

    assert len(results) == 2
    assert results[0] == {"role": "user", "content": "first"}
    assert results[1] == {"role": "assistant", "content": "second"}
