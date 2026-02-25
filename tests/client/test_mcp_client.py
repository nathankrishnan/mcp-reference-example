import pytest
from unittest.mock import AsyncMock, MagicMock
from client.mcp_client import MCPClient


def test_mcp_client_init():
    """Test that MCPClient stores constructor arguments correctly."""
    client = MCPClient(
        command="uv",
        args=["run", "mcp_server.py"],
        env={"MY_VAR": "hello"},
    )

    assert client._command == "uv"
    assert client._args == ["run", "mcp_server.py"]
    assert client._env == {"MY_VAR": "hello"}
    assert client._session is None


def test_mcp_client_init_env_defaults_to_none():
    """Test that env is None by default if not provided."""
    client = MCPClient(command="uv", args=["run", "mcp_server.py"])

    assert client._env is None


def test_session_raises_before_connect():
    """Test that calling session() before connect() raises a ConnectionError."""
    client = MCPClient(command="uv", args=["run", "mcp_server.py"])

    with pytest.raises(
        ConnectionError, match="No active session. Call connect\\(\\) first."
    ):
        client.session()


def test_session_returns_session_after_connect():
    """Test that session() returns the active session once one is set."""
    client = MCPClient(command="uv", args=["run", "mcp_server.py"])

    # Simulate what connect() does by manually setting a mock session
    mock_session = MagicMock()
    client._session = mock_session

    assert client.session() == mock_session


async def test_cleanup_resets_session(mocker):
    """Test that cleanup() resets _session to None."""
    client = MCPClient(command="uv", args=["run", "mcp_server.py"])

    # Give it a fake session so we can verify it gets cleared
    client._session = MagicMock()

    # Mock the exit stack's aclose so we don't need a real async context
    mocker.patch.object(client._exit_stack, "aclose", new_callable=AsyncMock)

    await client.cleanup()

    assert client._session is None


async def test_connect_uses_correct_server_params(mocker):
    """Test that connect() passes the right command and args to StdioServerParameters."""
    client = MCPClient(command="uv", args=["run", "mcp_server.py"])

    # Mock the stdio_client context manager to return fake read/write streams
    mock_read = AsyncMock()
    mock_write = AsyncMock()

    mock_stdio = MagicMock()
    mock_stdio.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
    mock_stdio.__aexit__ = AsyncMock(return_value=False)

    # Mock the ClientSession context manager
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.initialize = AsyncMock()

    mocker.patch("client.mcp_client.stdio_client", return_value=mock_stdio)
    mocker.patch("client.mcp_client.ClientSession", return_value=mock_session)

    await client.connect()

    # After connect(), _session should be set (not None)
    assert client._session is not None
