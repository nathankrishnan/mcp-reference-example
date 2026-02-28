import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prompt_toolkit.document import Document
from core.cli import CommandAutoSuggest, UnifiedCompleter, CliApp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_prompt(name, description="", arg_names=None):
    """Create a mock Prompt with the given name and argument names."""
    prompt = MagicMock()
    prompt.name = name
    prompt.description = description
    if arg_names:
        args = []
        for arg_name in arg_names:
            arg = MagicMock()
            arg.name = arg_name
            args.append(arg)
        prompt.arguments = args
    else:
        prompt.arguments = []
    return prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompts():
    """Create a list of mock Prompt objects."""
    return [
        make_prompt("summarize", "Summarize a document", ["doc_id"]),
        make_prompt("translate", "Translate a document", ["doc_id"]),
    ]


@pytest.fixture
def auto_suggester(prompts):
    """Create a CommandAutoSuggest with mock prompts."""
    return CommandAutoSuggest(prompts)


@pytest.fixture
def completer():
    """Create an empty UnifiedCompleter."""
    return UnifiedCompleter()


@pytest.fixture
def mock_agent():
    """Create a mock CliChat agent."""
    agent = MagicMock()
    agent.list_docs_ids = AsyncMock(return_value=["report.docx", "notes.txt"])
    agent.list_prompts = AsyncMock(return_value=[])
    agent.run = AsyncMock(return_value="Test response")
    return agent


@pytest.fixture
def cli_app(mock_agent):
    """Create a CliApp with a mocked agent."""
    return CliApp(agent=mock_agent)


# ---------------------------------------------------------------------------
# CommandAutoSuggest tests
# ---------------------------------------------------------------------------


def test_auto_suggest_returns_none_for_non_slash(auto_suggester):
    """Test that get_suggestion returns None for non-slash input."""
    doc = Document("hello world")
    result = auto_suggester.get_suggestion(MagicMock(), doc)
    assert result is None


def test_auto_suggest_returns_none_for_unknown_command(auto_suggester):
    """Test that get_suggestion returns None for an unknown slash command."""
    doc = Document("/unknown")
    result = auto_suggester.get_suggestion(MagicMock(), doc)
    assert result is None


def test_auto_suggest_returns_argument_name(auto_suggester):
    """Test that get_suggestion returns the first argument name for a known command."""
    doc = Document("/summarize")
    result = auto_suggester.get_suggestion(MagicMock(), doc)
    assert result is not None
    assert result.text == " doc_id"


def test_auto_suggest_returns_none_for_partial_command(auto_suggester):
    """Test that get_suggestion returns None for a partial command not in the dict."""
    doc = Document("/sum")
    result = auto_suggester.get_suggestion(MagicMock(), doc)
    assert result is None


# ---------------------------------------------------------------------------
# UnifiedCompleter tests
# ---------------------------------------------------------------------------


def test_update_prompts_stores_prompts(completer, prompts):
    """Test that update_prompts populates prompts and prompt_dict."""
    completer.update_prompts(prompts)

    assert completer.prompts is prompts
    assert "summarize" in completer.prompt_dict
    assert "translate" in completer.prompt_dict


def test_update_resources_stores_resources(completer):
    """Test that update_resources populates the resources list."""
    resources = ["report.docx", "notes.txt"]
    completer.update_resources(resources)

    assert completer.resources is resources


def test_completions_at_mention_matches(completer):
    """Test that @mention prefix yields matching resource completions."""
    completer.update_resources(["report.docx", "readme.md", "notes.txt"])

    doc = Document("tell me about @rep")
    completions = list(completer.get_completions(doc, MagicMock()))

    assert len(completions) == 1
    assert completions[0].text == "report.docx"


def test_completions_at_mention_no_match(completer):
    """Test that @mention with no matching resources yields nothing."""
    completer.update_resources(["report.docx", "notes.txt"])

    doc = Document("@xyz")
    completions = list(completer.get_completions(doc, MagicMock()))

    assert len(completions) == 0


def test_completions_slash_command_prefix(completer, prompts):
    """Test that a partial slash command yields matching prompt completions."""
    completer.update_prompts(prompts)

    doc = Document("/sum")
    completions = list(completer.get_completions(doc, MagicMock()))

    assert len(completions) == 1
    assert completions[0].text == "summarize"
    assert completions[0].display_meta_text == "Summarize a document"


def test_completions_slash_command_argument(completer, prompts):
    """Test that a complete command followed by space yields all resource IDs."""
    completer.update_prompts(prompts)
    completer.update_resources(["report.docx", "notes.txt"])

    doc = Document("/summarize ")
    completions = list(completer.get_completions(doc, MagicMock()))

    texts = [c.text for c in completions]
    assert "report.docx" in texts
    assert "notes.txt" in texts


def test_completions_plain_text_yields_nothing(completer, prompts):
    """Test that plain text input yields no completions."""
    completer.update_prompts(prompts)
    completer.update_resources(["report.docx"])

    doc = Document("hello world")
    completions = list(completer.get_completions(doc, MagicMock()))

    assert len(completions) == 0


# ---------------------------------------------------------------------------
# CliApp tests
# ---------------------------------------------------------------------------


def test_init_creates_session_and_completer(cli_app):
    """Test that __init__ sets up the completer, session, and key bindings."""
    assert cli_app.completer is not None
    assert cli_app.session is not None
    assert cli_app.kb is not None


async def test_initialize_calls_refresh_methods(cli_app):
    """Test that initialize calls both refresh_resources and refresh_prompts."""
    with patch.object(cli_app, "refresh_resources", new_callable=AsyncMock) as mock_res, \
         patch.object(cli_app, "refresh_prompts", new_callable=AsyncMock) as mock_pr:
        await cli_app.initialize()

    mock_res.assert_called_once()
    mock_pr.assert_called_once()


async def test_refresh_resources_updates_completer(cli_app, mock_agent):
    """Test that refresh_resources fetches doc IDs and updates the completer."""
    await cli_app.refresh_resources()

    mock_agent.list_docs_ids.assert_called_once()
    assert cli_app.completer.resources == ["report.docx", "notes.txt"]


async def test_refresh_resources_handles_error(cli_app, mock_agent, capsys):
    """Test that refresh_resources catches exceptions and prints an error."""
    mock_agent.list_docs_ids = AsyncMock(side_effect=RuntimeError("connection failed"))

    await cli_app.refresh_resources()

    captured = capsys.readouterr()
    assert "Error refreshing resources" in captured.out


async def test_refresh_prompts_updates_completer_and_suggester(cli_app, mock_agent, prompts):
    """Test that refresh_prompts updates the completer and auto-suggester."""
    mock_agent.list_prompts = AsyncMock(return_value=prompts)

    await cli_app.refresh_prompts()

    mock_agent.list_prompts.assert_called_once()
    assert cli_app.completer.prompts is prompts
    assert isinstance(cli_app.session.auto_suggest, CommandAutoSuggest)


async def test_refresh_prompts_handles_error(cli_app, mock_agent, capsys):
    """Test that refresh_prompts catches exceptions and prints an error."""
    mock_agent.list_prompts = AsyncMock(side_effect=RuntimeError("connection failed"))

    await cli_app.refresh_prompts()

    captured = capsys.readouterr()
    assert "Error refreshing prompts" in captured.out


async def test_run_processes_input_and_prints(cli_app, mock_agent, capsys):
    """Test that run() reads input, calls agent.run, and prints the response."""
    cli_app.session.prompt_async = AsyncMock(
        side_effect=["hello", KeyboardInterrupt()]
    )

    await cli_app.run()

    mock_agent.run.assert_called_once_with("hello")
    captured = capsys.readouterr()
    assert "Test response" in captured.out


async def test_run_skips_empty_input(cli_app, mock_agent):
    """Test that run() skips empty input without calling agent.run."""
    cli_app.session.prompt_async = AsyncMock(
        side_effect=["", "   ", KeyboardInterrupt()]
    )

    await cli_app.run()

    mock_agent.run.assert_not_called()


async def test_run_exits_on_keyboard_interrupt(cli_app, mock_agent):
    """Test that run() exits cleanly on KeyboardInterrupt."""
    cli_app.session.prompt_async = AsyncMock(side_effect=KeyboardInterrupt())

    await cli_app.run()

    mock_agent.run.assert_not_called()


async def test_run_exits_on_colon_q(cli_app, mock_agent):
    """Test that run() exits when user types :q."""
    cli_app.session.prompt_async = AsyncMock(return_value=":q")

    await cli_app.run()

    mock_agent.run.assert_not_called()


async def test_run_exits_on_colon_quit(cli_app, mock_agent):
    """Test that run() exits when user types :quit."""
    cli_app.session.prompt_async = AsyncMock(return_value=":quit")

    await cli_app.run()

    mock_agent.run.assert_not_called()
