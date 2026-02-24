from anthropic.types import Message, TextBlock, Usage
from core.claude import Claude

DESIRED_MODEL = "claude-sonnet-4-6"


def test_claude_init():
    """Test that the Claude class inits with a provided model string."""
    client = Claude(model=DESIRED_MODEL)

    assert client.model == DESIRED_MODEL
    assert client.client is not None


def test_add_user_message():
    """Test that user messages are formatted and appended correctly."""
    client = Claude(model=DESIRED_MODEL)
    messages = []

    client.add_user_message(messages, "Hello world")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello world"


def test_text_from_message():
    """Test extracting text from an Anthropic Message object."""
    client = Claude(model=DESIRED_MODEL)

    # Create a perfectly typed fake Anthropic Message object
    fake_message = Message(
        id="msg_123",
        model=DESIRED_MODEL,
        role="assistant",
        type="message",
        usage=Usage(input_tokens=10, output_tokens=10),
        content=[TextBlock(type="text", text="This is a mocked response.")],
    )

    extracted_text = client.text_from_message(fake_message)
    assert extracted_text == "This is a mocked response."


def test_chat_method_mocked(mocker):
    """
    Test the chat method by mocking the Anthropic API call.
    This ensures we don't actually hit the network/use credits.
    """
    client = Claude(model=DESIRED_MODEL)

    # Mock message we want the API to return
    mock_return_message = Message(
        id="msg_456",
        model=DESIRED_MODEL,
        role="assistant",
        type="message",
        usage=Usage(input_tokens=5, output_tokens=5),
        content=[TextBlock(type="text", text="Mocked chat response")],
    )

    # Tell pytest to intercept the Anthropic `Messages.create()` method
    # and return our mock message instead.
    mock_create = mocker.patch(
        "anthropic.resources.messages.Messages.create", return_value=mock_return_message
    )

    result = client.chat(messages=[{"role": "user", "content": "Hi"}])

    # Verify our method returned the mock response
    assert result == mock_return_message

    # Verify that our Claude class actually called Anthropic's `Messages.create()`
    # with the correct parameters.
    mock_create.assert_called_once()

    assert mock_create.call_args.kwargs["model"] == DESIRED_MODEL
    assert mock_create.call_args.kwargs["max_tokens"] == 8000
